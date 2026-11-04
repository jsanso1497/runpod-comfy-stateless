"""One GPU child. Upstream code is pinned by the image build, not downloaded here."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from .video_process import atomic_json, parent_death_signal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    parent_death_signal(int(os.environ.get('WB_PARENT_PID', os.getppid())))
    config = json.loads((args.job / 'worker_config.json').read_text())
    status = args.job / 'worker_status.json'
    state = {'state': 'starting', 'phase': 'model_import', 'pid': os.getpid(), 'chunks_completed': 0}
    def record(**values):
        state.update(values)
        atomic_json(status, state)
    def capture_failure(exc):
        # Upstream main() prints exceptions then raises SystemExit(1).
        # Capture before that boundary so the real phase and CUDA OOM survive.
        if state.get('state') == 'failed' and state.get('error_type'):
            return
        message = str(exc).lower()
        is_oom = (('cuda' in message and 'out of memory' in message)
                  or 'cublas_status_alloc_failed' in message
                  or type(exc).__name__ == 'OutOfMemoryError')
        record(state='failed', error_type=type(exc).__name__, error=str(exc),
               oom=is_oom, traceback=traceback.format_exc())
    try:
        import torch
        # Built-in fused SDPA is available without installing flash-attn.
        # Refuse an unbounded math-attention fallback in the GPU child.
        fused_only = config.get('fused_only', True)
        if fused_only:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        cli = Path(config['cli'])
        sys.path.insert(0, str(cli.parent))
        sys.argv = [str(cli)] + config['args']
        spec = importlib.util.spec_from_file_location('_wb_seedvr_cli', cli)
        if spec is None or spec.loader is None:
            raise RuntimeError('Unable to load the installed SeedVR2 CLI')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for symbol, label in (
            ('prepare_runner', 'model_preparation'),
            ('encode_all_batches', 'vae_encode'),
            ('upscale_all_batches', 'dit_restore'),
            ('decode_all_batches', 'vae_decode'),
            ('postprocess_all_batches', 'color_and_overlap'),
        ):
            original = getattr(module, symbol)
            def make_wrapper(fn, phase):
                def wrapped(*a, **kw):
                    values = {'state': 'running', 'phase': phase}
                    if torch.cuda.is_available():
                        free, total = torch.cuda.mem_get_info()
                        values.update(gpu_free_bytes=free, gpu_total_bytes=total,
                                      allocated_bytes=torch.cuda.memory_allocated(),
                                      reserved_bytes=torch.cuda.memory_reserved())
                    record(**values)
                    try:
                        result = fn(*a, **kw)
                    except BaseException as exc:
                        capture_failure(exc)
                        raise
                    if phase == 'color_and_overlap':
                        record(chunks_completed=state['chunks_completed'] + 1)
                    return result
                return wrapped
            setattr(module, symbol, make_wrapper(original, label))
        original_process = module.process_single_file
        def guarded_process(*a, **kw):
            try:
                return original_process(*a, **kw)
            except BaseException as exc:
                capture_failure(exc)
                raise
        module.process_single_file = guarded_process
        record(phase='starting_inference', state='running')
        module.main()
        record(state='complete', phase='inference_complete')
        return 0
    except BaseException as exc:
        capture_failure(exc)
        record(exit_exception_type=type(exc).__name__)
        traceback.print_exc()
        return 42 if state.get('oom') else 1
    finally:
        # The process exits even if cleanup fails. No model can outlive it.
        sys.stdout.flush()
        sys.stderr.flush()

if __name__ == '__main__':
    raise SystemExit(main())
