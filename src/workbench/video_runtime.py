"""Optional runtime additions for the combined Qwen + video image."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import threading
from .video_process import atomic_json


def launch_args(comfy: Path, family: str) -> list[str]:
    if family != 'qwen' or os.environ.get('WB_VIDEO_REPAIR', '0') != '1':
        return []
    text = (comfy / 'comfy/cli_args.py').read_text()
    args = ['--cache-none', '--disable-smart-memory', '--reserve-vram', '2']
    for flag in ('--cache-none', '--disable-smart-memory', '--reserve-vram'):
        if flag not in text:
            raise RuntimeError('Pinned ComfyUI is missing required memory option: ' + flag)
    # Preserve the existing frontend/assets settings. The independent saved-file
    # browser does not depend on an optional asset database or history retention.
    return args


def spawn_comfy(args: list[str], *, cwd: str, env: dict):
    if os.environ.get('WB_VIDEO_REPAIR', '0') != '1':
        return subprocess.Popen(args, cwd=cwd, env=env)
    from logging.handlers import RotatingFileHandler
    import logging
    logroot = Path(os.environ.get('WB_DATA_ROOT', '/workspace')) / 'workbench' / 'logs'
    logroot.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(logroot / 'comfy.log', maxBytes=8*1024**2, backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
    def drain():
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                record = logging.LogRecord('comfy', logging.INFO, '', 0, line.rstrip('\n'), (), None)
                handler.emit(record)
        finally:
            proc.stdout.close()
            handler.close()
    threading.Thread(target=drain, name='workbench-comfy-log', daemon=True).start()
    atomic_json(logroot / 'comfy-process.json', {'pid': proc.pid, 'args': args})
    return proc
