"""Small CPU-only utilities. No model imports in ComfyUI or the supervisor."""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODEL = 'seedvr2_ema_7b_fp16.safetensors'
VAE = 'ema_vae_fp16.safetensors'
VERSION = '2026-11-03.faithful-video-1'


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def child_env() -> dict[str, str]:
    # Credentials never enter the GPU/FFmpeg environment or settings.json.
    keys = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_HOME',
            'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES',
            'TMPDIR', 'PYTHONPATH')
    result = {k: os.environ[k] for k in keys if k in os.environ}
    result.update(PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  PYTORCH_CUDA_ALLOC_CONF='backend:cudaMallocAsync', OMP_NUM_THREADS='4')
    return result


def parent_death_signal(parent: int | None = None) -> None:
    if sys.platform != 'linux':
        return
    expected = parent if parent is not None else os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'prctl(PR_SET_PDEATHSIG) failed')
    if os.getppid() != expected:
        os.kill(os.getpid(), signal.SIGKILL)


def terminate(proc: subprocess.Popen, group: bool = False) -> None:
    # A process group may still have descendants after its leader exits.
    if not group and proc.poll() is not None:
        return
    try:
        if group:
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            if group:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=8)
    if group:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_logged(command: list[str], log: Path, cwd: Path | None = None) -> None:
    with log.open('ab', buffering=0) as f:
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=f,
                             stderr=subprocess.STDOUT, env=child_env(), cwd=cwd)
        try:
            rc = p.wait()
            if rc:
                tail = log.read_text(errors='replace')[-3500:]
                raise RuntimeError(f'Process exited {rc}. See {log.name}\n{tail}')
        finally:
            terminate(p)
