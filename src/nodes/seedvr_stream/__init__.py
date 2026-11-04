"""ComfyUI-facing isolated video node. Only paths/configuration enter its cache."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from workbench.video_process import atomic_json, clean_environment, terminate_process
from workbench.video_memory import install_job_cleanup, preflight_gpu, release_models

install_job_cleanup()
WEB_DIRECTORY = './web'


class WBSeedVR2StreamVideo:
    CATEGORY = 'Workbench/Video'
    DESCRIPTION = ('Restores a complete video through a separate, disk-backed SeedVR2 process. '
                   'Keeps FP16 weights and wavelet color. Cancellation reaps the worker. '
                   'Saves the final MP4 and returns a file-backed VIDEO, not a frame tensor.')
    FUNCTION = 'restore'
    RETURN_TYPES = ('VIDEO', 'STRING')
    RETURN_NAMES = ('video', 'file_path')
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'video': ('VIDEO',),
            'mode': (['1080p', '4k', '2x3-source'],),
            'model': (['sharp', 'standard'],),
            'start_seconds': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 86400.0, 'step': 0.01}),
            'duration_seconds': ('FLOAT', {'default': 24.0, 'min': 0.0, 'max': 86400.0, 'step': 0.01,
                                         'tooltip': '0 means the remaining source. No frame cap.'}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 4294967295, 'control_after_generate': False}),
            'crf': ('INT', {'default': 16, 'min': 0, 'max': 30}),
            'keep_intermediates': ('BOOLEAN', {'default': False,
                'tooltip': 'On success retain lossless PNGs; failed jobs retain diagnostics regardless.'}),
        }}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    async def restore(self, video, mode, model, start_seconds, duration_seconds,
                      seed, crf, keep_intermediates):
        import folder_paths
        import comfy.model_management as mm
        from comfy.utils import ProgressBar
        from comfy_api.latest import InputImpl
        from filelock import FileLock, Timeout
        source_fn = getattr(video, 'get_stream_source', None)
        if source_fn is None:
            raise ValueError('Connect Load Video directly. In-memory IMAGE batches are not accepted.')
        source_raw = source_fn()
        if not isinstance(source_raw, (str, os.PathLike)):
            raise ValueError('A video file on disk is required; no in-memory video copy will be made.')
        source = Path(source_raw).resolve()
        allowed = [Path(folder_paths.get_input_directory()).resolve(), Path(folder_paths.get_output_directory()).resolve()]
        if not source.is_file() or not any(source.is_relative_to(root) for root in allowed):
            raise ValueError('Video must exist under the configured input or output directory.')
        # Avoid silently ignoring an upstream crop or trim. The clean graphs
        # connect LoadVideo directly and expose the trim on this node.
        if getattr(video, '_VideoFromFile__crop', None) is not None:
            raise ValueError('Connect the uncropped source and use this node\'s crop mode.')
        trim = getattr(video, 'get_active_trim_window', lambda: (0, 0))()
        if trim != (0, 0) and trim != (0.0, 0.0):
            raise ValueError('Connect Load Video directly; set start/duration on this node.')
        data = Path(os.environ.get('WB_DATA_ROOT', '/workspace')).resolve()
        jobs = data / 'workbench' / 'video_jobs'
        jobs.mkdir(parents=True, exist_ok=True)
        job_id = 'seedvr_' + mode.replace('-', '_') + '_' + uuid.uuid4().hex[:16]
        job = jobs / job_id
        job.mkdir()
        output_root = Path(folder_paths.get_output_directory()).resolve()
        output = output_root / 'video' / job_id / 'restored.mp4'
        comfy_root = Path(folder_paths.base_path)
        matches = sorted(p for p in (comfy_root / 'custom_nodes').glob('*/inference_cli.py')
                         if 'seedvr2' in p.parent.name.lower())
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one pinned SeedVR2 CLI in the image.')
        request = {'source': str(source), 'output': str(output), 'cli': str(matches[0]),
                   'model_dir': str(data / 'models' / 'SEEDVR2'), 'mode': mode, 'model': model,
                   'start': float(start_seconds), 'duration': float(duration_seconds),
                   'seed': int(seed), 'crf': int(crf), 'keep_intermediates': bool(keep_intermediates),
                   'oom_retry': os.environ.get('WB_SEEDVR_OOM_RETRY', '1') == '1',
                   'fused_only': os.environ.get('WB_SEEDVR_FUSED_ONLY', '1') == '1'}
        atomic_json(job / 'request.json', request)
        lock = FileLock(str(jobs / '.gpu.lock'))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            raise RuntimeError('Another video job is running; use the normal ComfyUI queue.') from None
        proc = None
        try:
            atomic_json(job / 'preflight.json', preflight_gpu(mode))
            env = clean_environment()
            env['WB_PARENT_PID'] = str(os.getpid())
            pbar = ProgressBar(100)
            last_progress = -1
            with (job / 'supervisor.log').open('ab', buffering=0) as log:
                proc = subprocess.Popen([sys.executable, '-u', '-m', 'workbench.seedvr_job', '--job', str(job)],
                                        env=env, cwd=str(comfy_root), stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while proc.poll() is None:
                    mm.throw_exception_if_processing_interrupted()
                    progress = 0
                    try:
                        state = json.loads((job / 'status.json').read_text())
                        phase = state.get('phase')
                        if phase == 'inference':
                            worker = json.loads((job / 'worker_status.json').read_text())
                            cfg = json.loads((job / 'worker_config.json').read_text())
                            chunk = int(cfg['args'][cfg['args'].index('--chunk_size')+1])
                            total = max(1, int(state.get('frames', 1)))
                            progress = min(90, 5 + int(worker.get('chunks_completed', 0)*chunk/total*85))
                        elif phase == 'cpu_encode':
                            progress = 95
                    except (OSError, ValueError, KeyError):
                        pass
                    if progress != last_progress:
                        pbar.update_absolute(progress, 100)
                        last_progress = progress
                    await asyncio.sleep(0.35)
                state = json.loads((job / 'status.json').read_text()) if (job / 'status.json').exists() else {}
                if proc.returncode != 0 or state.get('state') != 'complete' or not output.is_file():
                    error = state.get('error', f'Worker exited {proc.returncode}. Check memory.events for an OS kill.')
                    raise RuntimeError(f'{error}\nDiagnostics: /_workbench/repair?job={job_id}')
                pbar.update_absolute(100, 100)
            rel = output.relative_to(output_root)
            return {'ui': {'images': [{'filename': output.name, 'subfolder': rel.parent.as_posix(),
                                      'type': 'output'}], 'animated': (True,)},
                    'result': (InputImpl.VideoFromFile(str(output)), str(output))}
        except BaseException as exc:
            if proc is not None:
                terminate_process(proc, group=True)
            status_file = job / 'status.json'
            if not status_file.exists():
                atomic_json(status_file, {'state': 'cancelled' if not isinstance(exc, Exception) else 'failed',
                                          'phase': 'preflight', 'error': str(exc),
                                          'error_type': type(exc).__name__, 'job_id': job_id})
            raise
        finally:
            if proc is not None:
                terminate_process(proc, group=True)
            lock.release()
            # Job-boundary cleanup resets executor caches separately. Here no
            # restoration tensors exist in the ComfyUI process in the first place.
            try:
                release_models()
            except Exception as cleanup_error:
                print('[Workbench memory] Parent cleanup warning: ' + repr(cleanup_error), flush=True)

NODE_CLASS_MAPPINGS = {'WBSeedVR2StreamVideo': WBSeedVR2StreamVideo}
NODE_DISPLAY_NAME_MAPPINGS = {'WBSeedVR2StreamVideo': 'SeedVR2 FP16 Video | Isolated Streaming'}
