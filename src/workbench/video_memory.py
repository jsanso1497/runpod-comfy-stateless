"""Job-boundary cleanup and measurements. Never kill arbitrary GPU processes."""
from __future__ import annotations
import contextlib
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from .video_process import atomic_json


def available_ram() -> int:
    vals = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith(('MemAvailable:', 'MemTotal:')):
                vals[line.split(':')[0]] = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return 0
    available = vals.get('MemAvailable', 0)
    for limit, current in (
        ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ):
        try:
            raw = Path(limit).read_text().strip()
            if raw.isdigit():
                available = min(available, max(0, int(raw) - int(Path(current).read_text())))
        except (OSError, ValueError):
            pass
    return available


def snapshot() -> dict:
    result = {'timestamp': time.time(), 'pid': os.getpid(), 'available_ram_bytes': available_ram()}
    for name in ('memory.current', 'memory.max', 'memory.events'):
        try:
            result[name] = Path('/sys/fs/cgroup/' + name).read_text().strip()
        except OSError:
            pass
    try:
        result['gpu_processes'] = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'],
            text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        result['gpu_processes'] = 'unavailable'
    return result


def release_models() -> list[str]:
    actions = []
    import comfy.model_management as mm
    mm.unload_all_models()
    actions.append('unloaded Comfy-managed models')
    gc.collect()
    mm.soft_empty_cache()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            actions.append('released unused CUDA allocator cache')
    except Exception as exc:
        actions.append('CUDA cleanup warning: ' + type(exc).__name__)
    return actions


def preflight_gpu(mode: str) -> dict:
    release_models()
    result = snapshot()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('SeedVR2 inference requires a CUDA GPU')
    free, total = torch.cuda.mem_get_info()
    result.update(gpu_free_bytes=free, gpu_total_bytes=total)
    minimum = float(os.environ.get('WB_SEEDVR_MIN_FREE_VRAM_GB', '32' if mode == '4k' else '20')) * 1024**3
    minimum_ram = float(os.environ.get('WB_SEEDVR_MIN_FREE_RAM_GB', '24')) * 1024**3
    if free < minimum:
        raise RuntimeError(f'Only {free/1024**3:.1f} GiB GPU memory free before SeedVR2; need {minimum/1024**3:.1f} GiB for this preflight. Check the Diagnostics page for another GPU process. No render started.')
    if result['available_ram_bytes'] and result['available_ram_bytes'] < minimum_ram:
        raise RuntimeError(f'Only {result["available_ram_bytes"]/1024**3:.1f} GiB container RAM available. Full FP16 needs more headroom. No render started.')
    return result


def install_job_cleanup() -> None:
    if os.environ.get('WB_VIDEO_REPAIR', '0') != '1' or os.environ.get('WB_CLEAN_AFTER_JOB', '1') != '1':
        return
    import execution
    old = execution.PromptExecutor.execute
    if getattr(old, '_wb_cleanup', False):
        return
    def wrapped(self, *args, **kwargs):
        try:
            return old(self, *args, **kwargs)
        finally:
            # reset() also resets success/messages. Preserve those for main.py's
            # task_done/history code: a failure must never turn into a success.
            preserved = {k: getattr(self, k) for k in ('success', 'status_messages', 'history_result') if hasattr(self, k)}
            try:
                self.reset()
                for key, value in preserved.items():
                    setattr(self, key, value)
                actions = release_models()
                report = snapshot()
                report['actions'] = actions
                root = Path(os.environ.get('WB_DATA_ROOT', '/workspace')) / 'workbench'
                atomic_json(root / 'last-job-memory.json', report)
                print('[Workbench memory] End-of-job caches reset; managed models unloaded.', flush=True)
            except Exception as exc:
                # Preserve output and the original failure, and expose cleanup failure.
                for key, value in preserved.items():
                    setattr(self, key, value)
                print('[Workbench memory] Cleanup failed: ' + repr(exc), flush=True)
    wrapped._wb_cleanup = True
    execution.PromptExecutor.execute = wrapped
