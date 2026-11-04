"""Child lifetime and logging. No CUDA imports and no shell commands."""
from __future__ import annotations
import contextlib
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def parent_death_signal(expected_parent: int | None = None) -> None:
    """Linux child dies even if ComfyUI is killed instead of cancelling cleanly."""
    if not sys.platform.startswith('linux'):
        return
    parent = expected_parent if expected_parent is not None else os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot set child parent-death signal')
    if os.getppid() != parent:
        raise SystemExit('Parent exited during child startup')


def terminate_process(proc: subprocess.Popen, *, group: bool = False) -> None:
    if proc.poll() is not None:
        return
    def send(sig):
        with contextlib.suppress(ProcessLookupError):
            if group and os.name == 'posix':
                os.killpg(proc.pid, sig)
            else:
                proc.send_signal(sig)
    send(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        proc.wait(timeout=10)


def run_logged(cmd: list[str], log: Path, *, cwd: Path | None = None,
               env: dict | None = None, cancelled=None) -> int:
    """Wait on one child, always reap it, and never hold an entire stdout pipe."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab', buffering=0) as handle:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            while proc.poll() is None:
                if cancelled:
                    cancelled()
                time.sleep(0.15)
            return int(proc.returncode)
        finally:
            terminate_process(proc)


def clean_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in ('HF_TOKEN', 'hf_token', 'CIVITAI_TOKEN', 'civit_token',
                'WB_PASSWORD', 'WB_LORA_URLS', 'WB_CHECKPOINTS'):
        env.pop(key, None)
    env.update(PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1')
    # Do not replace the parent's allocator. This applies only to the child.
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    return env
