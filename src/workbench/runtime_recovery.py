"""Explicit crash recovery for the one managed ComfyUI child; no restart loops."""
from __future__ import annotations
import asyncio
import os
from pathlib import Path
import socket
import subprocess
from aiohttp import ClientTimeout, web


def launch_options(comfy_root: Path, family: str):
    raw = os.environ.get('WB_DISABLE_COMFY_COMPILER', '1' if family in ('qwen', 'krea') else '0')
    if raw not in ('0', '1'):
        raise ValueError('WB_DISABLE_COMFY_COMPILER must be 0 or 1.')
    if raw == '0':
        return []
    path = Path(comfy_root) / 'comfy/cli_args.py'
    if not path.is_file() or '--disable-comfy-compiler' not in path.read_text(encoding='utf-8'):
        raise ValueError('This ComfyUI build does not support the requested compiler switch.')
    return ['--disable-comfy-compiler']


def guard_single_instance(runtime):
    if runtime.comfy is not None and runtime.comfy.poll() is None:
        raise ValueError('ComfyUI is already running. Use the managed restart command, not a second main.py.')
    try:
        connection = socket.create_connection(('127.0.0.1', 8189), timeout=0.3)
    except ConnectionRefusedError:
        return
    except OSError as exc:
        raise ValueError('Cannot verify that the managed ComfyUI port is free.') from exc
    else:
        connection.close()
        raise ValueError('Port 8189 is already occupied. No second ComfyUI process was started.')


async def restart(runtime, request):
    if not hasattr(runtime, '_restart_lock'):
        runtime._restart_lock = asyncio.Lock()
    if runtime._restart_lock.locked():
        return web.json_response({'error': 'A ComfyUI restart is already in progress.'}, status=409)
    async with runtime._restart_lock:
        if runtime.job and not runtime.job.done():
            return web.json_response({'error': 'Wait for model preparation to finish before restarting.'}, status=409)
        proc = runtime.comfy
        if proc is not None and proc.poll() is None:
            try:
                async with runtime.session.get('http://127.0.0.1:8189/queue', timeout=ClientTimeout(total=5)) as response:
                    if response.status != 200:
                        raise RuntimeError('Queue unavailable')
                    queue = await response.json()
                if not isinstance(queue, dict) or 'queue_running' not in queue or 'queue_pending' not in queue:
                    raise RuntimeError('Unknown queue state')
            except Exception:
                return web.json_response({'error': 'ComfyUI is alive but its queue cannot be verified. No process was killed.'}, status=409)
            if queue['queue_running'] or queue['queue_pending']:
                return web.json_response({'error': 'ComfyUI has running or queued work. Finish or cancel it before restart.'}, status=409)
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait, timeout=10)
        runtime.comfy = None
        runtime.start_comfy()
        return web.json_response({'state': 'starting', 'internal_port': 8189, 'public_port': 8188}, status=202)
