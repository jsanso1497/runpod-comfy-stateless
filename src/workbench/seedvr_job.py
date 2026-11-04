"""CPU supervisor for a disk-backed SeedVR2 job; no model imports in this process."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import traceback
from .video_repair_catalog import MODEL, STANDARD, VAE
from .video_process import atomic_json, parent_death_signal, run_logged, clean_environment
from .video_media import source_plan, storage_required, prepare, finish, probe, video_stream


def cli_arguments(job: Path, request: dict, attempt: int = 0) -> list[str]:
    mode = request['mode']
    short, long = (2160, 3840) if mode == '4k' else (1080, 1920)
    batch = 5 if mode == '4k' or attempt else 9
    chunk = 9 if mode == '4k' or attempt else 33
    tile = 256 if attempt else 512
    swaps = 36 if mode == '4k' or attempt else 24
    model = MODEL if request['model'] == 'sharp' else STANDARD
    return [str(job / 'prepared.mkv'), '--output', str(job / 'restored_frames'),
            '--output_format', 'png', '--model_dir', request['model_dir'], '--dit_model', model,
            '--cuda_device', '0', '--resolution', str(short), '--max_resolution', str(long),
            '--chunk_size', str(chunk), '--batch_size', str(batch), '--uniform_batch_size',
            '--temporal_overlap', '3', '--prepend_frames', '0', '--seed', str(request['seed']),
            '--color_correction', 'wavelet', '--input_noise_scale', '0', '--latent_noise_scale', '0',
            '--dit_offload_device', 'cpu', '--vae_offload_device', 'cpu', '--tensor_offload_device', 'cpu',
            '--blocks_to_swap', str(swaps), '--swap_io_components', '--attention_mode', 'sdpa',
            '--vae_encode_tiled', '--vae_encode_tile_size', str(tile), '--vae_encode_tile_overlap', '128',
            '--vae_decode_tiled', '--vae_decode_tile_size', str(tile), '--vae_decode_tile_overlap', '128',
            '--debug']


def run_job(job: Path) -> None:
    request = json.loads((job / 'request.json').read_text())
    state = {'state': 'running', 'phase': 'preflight', 'pid': os.getpid(), 'job_id': job.name}
    status = job / 'status.json'
    def record(**kw):
        state.update(kw)
        atomic_json(status, state)
    record()
    source = Path(request['source'])
    plan = source_plan(source, request['mode'], request['start'], request['duration'])
    plan['crf'] = request.get('crf', 16)
    atomic_json(job / 'plan.json', plan)
    output = Path(request['output'])
    if output.exists():
        raise FileExistsError('This job already has a completed output')
    required = storage_required(plan)
    if shutil.disk_usage(job).free < required:
        raise RuntimeError(f'Need about {required/1024**3:.1f} GiB free for lossless intermediate frames; nothing deleted')
    models = Path(request['model_dir'])
    model = MODEL if request['model'] == 'sharp' else STANDARD
    for name in (model, VAE):
        if not (models / name).is_file():
            raise FileNotFoundError('Required model not prepared: ' + str(models / name))
    cli = Path(request['cli'])
    text = cli.read_text()
    for marker in ('--chunk_size', '--model_dir', '--output_format', '--swap_io_components'):
        if marker not in text:
            raise RuntimeError('Installed SeedVR2 CLI has an unexpected interface')
    record(phase='source_trim_and_crop')
    prepared, count = prepare(source, job, plan)
    record(phase='inference', frames=count)
    succeeded = False
    for attempt in range(2 if request.get('oom_retry', True) else 1):
        config = {'cli': str(cli), 'args': cli_arguments(job, request, attempt),
                  'fused_only': request.get('fused_only', True), 'attempt': attempt + 1}
        atomic_json(job / 'worker_config.json', config)
        atomic_json(job / f'worker_config_{attempt+1}.json', config)
        record(phase='inference', attempt=attempt+1)
        env = clean_environment()
        env['WB_PARENT_PID'] = str(os.getpid())
        rc = run_logged([sys.executable, '-u', '-m', 'workbench.seedvr_worker', '--job', str(job)],
                        job / f'inference_{attempt+1}.log', env=env, cwd=cli.parent)
        worker_state = {}
        try:
            worker_state = json.loads((job / 'worker_status.json').read_text())
            atomic_json(job / f'worker_result_{attempt+1}.json', worker_state)
        except (OSError, ValueError):
            pass
        if rc == 0 and worker_state.get('state') == 'complete':
            succeeded = True
            break
        # Retry only a confirmed CUDA-style OOM in a child that has exited.
        # Keep model, precision, target size and noise unchanged. No 1-frame fallback.
        if rc != 42 or not worker_state.get('oom') or attempt > 0 or not request.get('oom_retry', True):
            detail = worker_state.get('error', f'Worker exited {rc}; -9 can indicate an OS/cgroup kill')
            phase = worker_state.get('phase', 'worker_startup')
            raise RuntimeError(f'SeedVR2 failed in {phase}: {detail}. Logs: {job}')
        record(phase='retry_after_confirmed_oom', failed_phase=worker_state.get('phase'))
        # Only this job's incomplete frames; source/models/older outputs are untouched.
        shutil.rmtree(job / 'restored_frames', ignore_errors=True)
    if not succeeded:
        raise RuntimeError('Inference did not complete')
    # The CUDA process is gone before encoding begins.
    record(phase='cpu_encode')
    finish(job, output, plan, count)
    atomic_json(output.parent / 'settings.json', {'request': request, 'plan': plan, 'worker': config})
    record(state='complete', phase='complete', output=str(output), frames=count,
           final_size=plan['final_size'], gpu_worker_exited=True)
    if not request.get('keep_intermediates', False):
        shutil.rmtree(job / 'restored_frames', ignore_errors=True)
        prepared.unlink(missing_ok=True)
    print('COMPLETE: ' + str(output), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--finish-only', action='store_true', help='Validate and encode saved frames without loading a GPU model')
    args = parser.parse_args()
    parent_death_signal(int(os.environ.get('WB_PARENT_PID', os.getppid())))
    def cancelled(_signum, _frame):
        raise SystemExit(130)
    signal.signal(signal.SIGTERM, cancelled)
    try:
        if args.finish_only:
            request = json.loads((args.job / 'request.json').read_text())
            plan = json.loads((args.job / 'plan.json').read_text())
            count = int(video_stream(probe(args.job / 'prepared.mkv', count=True))['nb_read_frames'])
            output = finish(args.job, Path(request['output']), plan, count)
            atomic_json(args.job / 'status.json', {'state': 'complete', 'phase': 'complete',
                        'output': str(output), 'frames': count, 'gpu_worker_exited': True})
            print('COMPLETE: ' + str(output), flush=True)
        else:
            run_job(args.job)
        return 0
    except BaseException as exc:
        old = {}
        try:
            old = json.loads((args.job / 'status.json').read_text())
        except (OSError, ValueError):
            pass
        old.update(state='cancelled' if isinstance(exc, (SystemExit, KeyboardInterrupt)) else 'failed',
                   error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        atomic_json(args.job / 'status.json', old)
        traceback.print_exc()
        return 130 if isinstance(exc, (SystemExit, KeyboardInterrupt)) else 1

if __name__ == '__main__':
    raise SystemExit(main())
