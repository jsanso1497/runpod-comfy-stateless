#!/usr/bin/env python3
"""Password-protected JupyterLab file manager and two-service supervisor.

The file browser opens /workspace but is NOT a sandbox: a logged-in owner can
edit code and open a terminal in this container. No unauthenticated mode exists.
No keys/passwords are printed or passed as command-line arguments.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler

DEFAULT_PORT = 8888
STATE_ROOT = Path('/run/runpod-file-manager')
STOP = False


def password_from_environment() -> str:
    password = os.environ.get('FILEBROWSER_PASSWORD', '')
    if not password or len(password) < 12:
        raise ValueError('Set FILEBROWSER_PASSWORD to your RunPod secret, at least 12 characters. The file manager was NOT opened without authentication.')
    if len(password) > 1024 or '\n' in password or '\r' in password or '\0' in password:
        raise ValueError('FILEBROWSER_PASSWORD must be one line, 12-1024 characters.')
    if 'RUNPOD_SECRET_' in password or password.strip().startswith('{{'):
        raise ValueError('FILEBROWSER_PASSWORD still contains a secret placeholder. Create/select the filebrowser_password secret in RunPod.')
    return password


def checked_port(value: str | int) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('FILEBROWSER_PORT must be a numeric port; use 8888.') from exc
    if not 1024 <= port <= 65535 or port in (8188, 11434):
        raise ValueError('File manager port must be 1024-65535 and separate from ComfyUI (8188) and Ollama (11434). Use 8888.')
    return port


def safe_environment(state: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if re.search(r'(PASSWORD|TOKEN|SECRET|API_KEY|PRIVATE_KEY|CREDENTIAL)', key, re.I):
            env.pop(key, None)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PIP_CONSTRAINT', 'JUPYTER_CONFIG_PATH', 'JUPYTER_TOKEN_FILE'):
        env.pop(key, None)
    env.update({
        'JUPYTER_CONFIG_DIR': str(state / 'config'),
        'JUPYTER_RUNTIME_DIR': str(state / 'runtime'),
        'JUPYTER_DATA_DIR': str(state / 'data'),
        'JUPYTERLAB_SETTINGS_DIR': str(state / 'lab-settings'),
        'JUPYTERLAB_WORKSPACES_DIR': str(state / 'lab-workspaces'),
        'PYTHONUNBUFFERED': '1',
    })
    return env


def write_config(root: Path, state: Path, port: int) -> Path:
    from jupyter_server.auth import passwd
    password = password_from_environment()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # State must remain outside the directory being browsed.
    state = state.resolve()
    if state == root or state.is_relative_to(root):
        raise ValueError('File manager private state must be outside its browsable root.')
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    for name in ('config', 'runtime', 'data', 'lab-settings', 'lab-workspaces'):
        (state / name).mkdir(exist_ok=True, mode=0o700)
        os.chmod(state / name, 0o700)
    config = {
        'ServerApp': {
            'ip': '0.0.0.0', 'port': checked_port(port), 'port_retries': 0,
            'root_dir': str(root), 'open_browser': False, 'allow_root': True,
            'allow_remote_access': True, 'trust_xheaders': True,
            'allow_origin': '', 'disable_check_xsrf': False,
            'default_url': '/lab', 'terminals_enabled': True,
            'max_body_size': 512 * 1024**2, 'max_buffer_size': 512 * 1024**2,
            'cookie_secret_file': str(state / 'runtime/cookie_secret'),
            'log_level': 'WARN',
        },
        'PasswordIdentityProvider': {
            'hashed_password': passwd(password, algorithm='argon2'),
            'password_required': True, 'allow_password_change': False,
        },
        # Password stays enabled even though URL tokens are disabled.
        'IdentityProvider': {'token': ''},
        'ContentsManager': {'allow_hidden': True},
        'LabApp': {'extension_manager': 'readonly'},
    }
    path = state / 'config/jupyter_server_config.json'
    tmp = path.with_suffix('.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(config, f)
    os.replace(tmp, path)
    os.chmod(path, 0o600)
    return path


def run_server(root: Path, state: Path, port: int) -> None:
    config = write_config(root, state, port)
    env = safe_environment(state.resolve())
    command = [sys.executable, '-m', 'jupyterlab', '--config=' + str(config), '--no-browser']
    os.execve(sys.executable, command, env)


def get_status(port: int, path: str) -> int:
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request(f'http://127.0.0.1:{port}{path}', headers={'Accept': 'application/json'}), timeout=2) as response:
            return response.status
    except HTTPError as exc:
        return exc.code


def ready(port: int) -> bool:
    try:
        # A login page alone is not proof that file access is protected.
        return get_status(port, '/login') == 200 and get_status(port, '/api/contents') in (401, 403)
    except (OSError, URLError):
        return False


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def supervise(app: list[str]) -> int:
    global STOP
    if app and app[0] == '--':
        app = app[1:]
    if not app or not Path(app[0]).is_file():
        raise ValueError('File manager supervisor requires the existing application start script.')
    password_from_environment()  # Fail BEFORE model downloads when a secret is missing.
    port = checked_port(os.environ.get('FILEBROWSER_PORT', '8888'))
    STATE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(STATE_ROOT, 0o700)
    STOP = False
    def interrupted(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    browser = application = None
    logpath = STATE_ROOT / 'server.log'
    with open(logpath, 'a', encoding='utf-8') as log:
        os.chmod(logpath, 0o600)
        try:
            browser = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), 'run', '--port', str(port)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 90
            while not STOP and time.monotonic() < deadline:
                if browser.poll() is not None:
                    raise RuntimeError(f'File manager exited before startup. Read {logpath}; model downloads were not started.')
                if ready(port):
                    break
                time.sleep(0.5)
            else:
                if STOP:
                    return 0
                raise RuntimeError(f'File manager did not become ready within 90 seconds. Read {logpath}.')
            print(f'FILE MANAGER READY: port {port} | /workspace | password required | JupyterLab', flush=True)
            env = dict(os.environ)
            env.pop('FILEBROWSER_PASSWORD', None)
            # Jupyter is isolated; do not activate its environment for ComfyUI.
            application = subprocess.Popen(app, env=env, start_new_session=True)
            while not STOP:
                rc = application.poll()
                if rc is not None:
                    return rc if rc >= 0 else 128 - rc
                rc = browser.poll()
                if rc is not None:
                    print(f'FILE MANAGER EXITED ({rc}); stopping the application. Read {logpath}.', file=sys.stderr, flush=True)
                    return 1
                time.sleep(0.5)
            return 0
        finally:
            stop_process(application)
            stop_process(browser)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--root', type=Path, default=Path('/workspace'))
    run.add_argument('--state', type=Path, default=STATE_ROOT)
    run.add_argument('--port', type=int, default=DEFAULT_PORT)
    sup = sub.add_parser('supervise')
    sup.add_argument('app', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command == 'run':
            run_server(args.root, args.state, args.port)
            return 0
        return supervise(args.app)
    except (ValueError, RuntimeError) as exc:
        print('FILE MANAGER ERROR: ' + str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
