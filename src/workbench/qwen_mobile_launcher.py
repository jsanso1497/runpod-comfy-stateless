"""Qwen-only Jupyter sidecar supervisor; the existing Workbench runtime is unchanged."""
from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SECRET_KEYS = frozenset({
    'HF_TOKEN', 'CIVITAI_TOKEN', 'hf_token', 'civit_token', 'WB_PASSWORD',
    'WB_LORA_URLS', 'WB_CHECKPOINTS', 'JUPYTER_TOKEN', 'JUPYTER_TOKEN_FILE',
})


def jupyter_enabled(env: dict) -> bool:
    if env.get('WB_WORKSPACE', 'qwen') != 'qwen' or env.get('WB_APPROVED_ONLY', '0') == '1':
        return False
    value = env.get('WB_JUPYTER', '1')
    if value not in ('0', '1'):
        raise ValueError('WB_JUPYTER must be 0 or 1.')
    return value == '1'


def resolve_token(env: dict) -> str:
    # Explicit but invalid values must not silently fall back.
    value = env['JUPYTER_TOKEN'] if 'JUPYTER_TOKEN' in env else env.get('WB_PASSWORD', '')
    if not isinstance(value, str) or len(value) < 16 or not value.strip() or value.lstrip().startswith('{{'):
        raise ValueError('Jupyter needs JUPYTER_TOKEN or WB_PASSWORD resolving to at least 16 characters.')
    if any(ch in value for ch in ('\n', '\r', '\x00')):
        raise ValueError('Jupyter token must be a single line.')
    return value


def jupyter_environment(env: dict, token: str, root: Path) -> dict:
    # No model credentials, library URLs, Workbench password or Comfy PYTHONPATH in Jupyter children.
    clean = {k: v for k, v in env.items()
             if k not in SECRET_KEYS and k not in ('PYTHONPATH', 'PYTHONHOME')
             and not k.startswith('RUNPOD_SECRET_') and not k.startswith('JUPYTER_')}
    clean['JUPYTER_TOKEN'] = token
    clean['WB_DATA_ROOT'] = str(root)
    clean['JUPYTER_CONFIG_DIR'] = '/tmp/workbench-jupyter/config'
    clean['JUPYTER_RUNTIME_DIR'] = '/tmp/workbench-jupyter/runtime'
    clean['JUPYTERLAB_SETTINGS_DIR'] = str(root / '.jupyter/lab/user-settings')
    clean['JUPYTERLAB_WORKSPACES_DIR'] = str(root / '.jupyter/lab/workspaces')
    return clean


def comfy_environment(env: dict) -> dict:
    return {k: v for k, v in env.items() if k not in ('JUPYTER_TOKEN', 'JUPYTER_TOKEN_FILE')}


def private_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Refusing a symlinked Jupyter log.')
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(str(path), flags, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, 'ab', buffering=0)


def write_status(path: Path, state: str, **fields) -> None:
    # This record intentionally never contains a token, password or URL with credentials.
    import tempfile
    data = {'service': 'jupyterlab', 'state': state, 'port': 8888, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.jupyter-status-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(data, out, indent=2)
            out.write('\n')
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def signal_group(process, signum) -> None:
    if process is not None and process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signum)


def stop_process(process, timeout: float = 10) -> None:
    if process is None:
        return
    signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        signal_group(process, signal.SIGKILL)
        process.wait(timeout=5)


def run_supervisor(env: dict) -> int:
    root = Path(env.get('WB_DATA_ROOT', '/workspace')).expanduser()
    if not root.is_absolute():
        raise ValueError('WB_DATA_ROOT must be absolute.')
    root.mkdir(parents=True, exist_ok=True)
    state_dir = root / 'workbench/jupyter'
    state_dir.mkdir(parents=True, exist_ok=True)
    if not state_dir.resolve().is_relative_to(root.resolve()):
        raise ValueError('Jupyter state directory escapes WB_DATA_ROOT.')
    status = state_dir / 'status.json'
    jupyter = None
    runtime = None
    log = None
    received_signal = None
    old_handlers = {}
    try:
        try:
            enabled = jupyter_enabled(env)
            if enabled:
                token = resolve_token(env)
                python = Path('/opt/jupyter-venv/bin/python')
                config = Path(env.get('WB_ROOT', '/opt/workbench')) / 'deploy/jupyter_workbench_config.py'
                if not python.is_file() or not config.is_file():
                    raise ValueError('Jupyter is not installed in this image; rebuild Qwen HQ.')
                child_env = jupyter_environment(env, token, root)
                for key in ('JUPYTER_CONFIG_DIR', 'JUPYTER_RUNTIME_DIR'):
                    path = Path(child_env[key])
                    if path.is_symlink():
                        raise ValueError('Refusing a symlinked Jupyter runtime directory.')
                    path.mkdir(parents=True, exist_ok=True, mode=0o700)
                    path.chmod(0o700)
                log = private_file(state_dir / 'server.log')
                jupyter = subprocess.Popen(
                    [str(python), '-m', 'jupyterlab', '--config=' + str(config)],
                    cwd=str(root), env=child_env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                write_status(status, 'starting', pid=jupyter.pid, log='workbench/jupyter/server.log')
                print('[QWEN JUPYTER] Starting token-protected JupyterLab on HTTP 8888.', flush=True)
                token = None
                child_env = None
            else:
                write_status(status, 'disabled')
        except (ValueError, OSError) as exc:
            # Jupyter must never fail open, and its failure must not take ComfyUI down.
            write_status(status, 'unavailable', reason=str(exc))
            print('[QWEN JUPYTER] Not started: ' + str(exc), flush=True)

        def forward(signum, _frame):
            nonlocal received_signal
            if received_signal is None:
                received_signal = (signum, time.monotonic())
            signal_group(runtime, signum)
            signal_group(jupyter, signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, forward)
        runtime = subprocess.Popen(
            [sys.executable, '-m', 'workbench.runtime'], env=comfy_environment(env),
            start_new_session=True,
        )
        # A termination signal arriving between handler registration and spawn is not lost.
        if received_signal:
            signal_group(runtime, received_signal[0])
        jupyter_exit_reported = False
        jupyter_alive_reported = False
        start = time.monotonic()
        while runtime.poll() is None:
            if jupyter is not None:
                code = jupyter.poll()
                if code is not None and not jupyter_exit_reported:
                    write_status(status, 'exited', exit_code=code, log='workbench/jupyter/server.log')
                    print('[QWEN JUPYTER] Process exited; ComfyUI remains running. See workbench/jupyter/server.log.', flush=True)
                    jupyter_exit_reported = True
                elif code is None and not jupyter_alive_reported and time.monotonic() - start >= 3:
                    # Process liveness is deliberately not labeled authenticated HTTP readiness.
                    write_status(status, 'process-running', pid=jupyter.pid, log='workbench/jupyter/server.log')
                    jupyter_alive_reported = True
            if received_signal and time.monotonic() - received_signal[1] >= 20:
                signal_group(runtime, signal.SIGKILL)
                signal_group(jupyter, signal.SIGKILL)
            time.sleep(0.25)
        return runtime.returncode if runtime.returncode >= 0 else 128 - runtime.returncode
    finally:
        stop_process(runtime)
        stop_process(jupyter)
        if log is not None:
            log.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    env = dict(os.environ)
    if env.get('WB_WORKSPACE', 'qwen') != 'qwen':
        os.execve(sys.executable, [sys.executable, '-m', 'workbench.runtime'], comfy_environment(env))
    return run_supervisor(env)


if __name__ == '__main__':
    raise SystemExit(main())
