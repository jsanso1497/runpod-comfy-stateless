"""Start login/files immediately; supervise independent Jupyter and Comfy backends."""
from __future__ import annotations
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from access_gateway import password_hash
from origin_policy import public_origins
from launch import validate_password
from recovery_launcher import terminal_environment, stop_process

PACKAGE = Path(__file__).resolve().parents[1]


def private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    if path.is_symlink():
        raise ValueError('Refusing a symlinked runtime config.')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
    path.chmod(0o600)
    return path


def jupyter_settings(runtime, workspace, token, port=8889):
    # A new base URL and workspace avoid stale /lab and /api responses/settings.
    # Password auth lives at the gateway; private token auth still protects loopback.
    return {
        'ServerApp': {
            'ip': '127.0.0.1', 'port': port, 'port_retries': 0,
            'open_browser': False, 'allow_root': True, 'allow_remote_access': True,
            'trust_xheaders': True, 'root_dir': str(workspace),
            'base_url': '/jupyter/', 'default_url': '/lab/workspaces/h3-studio-v2',
            'terminals_enabled': True, 'disable_check_xsrf': False,
            'terminado_settings': {'shell_command': ['/bin/bash', '-l']},
            'tornado_settings': {'headers': {'Cache-Control': 'no-store'}},
        },
        'IdentityProvider': {'token': token},
        'PasswordIdentityProvider': {'password_required': False},
        'LabApp': {'workspaces_dir': str(runtime / 'lab-workspaces'),
                   'user_settings_dir': str(runtime / 'lab-settings')},
    }


def write_jupyter_proxy_config(runtime, workspace, token, port=8889):
    settings = jupyter_settings(runtime, workspace, token, port)
    path = runtime / 'jupyter_server_config.py'
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ValueError('Refusing symlinked Jupyter config.')
    text = "import json\nc = get_config()\nsettings = json.loads(%r)\n" % json.dumps(settings)
    text += "for section, values in settings.items():\n    for name, value in values.items():\n        setattr(getattr(c, section), name, value)\n"
    text += "def private_page_config(handler, config):\n    result = dict(config)\n    result['token'] = ''\n    result['buildAvailable'] = False\n    result['buildCheck'] = False\n    return result\n"
    text += "c.ServerApp.tornado_settings = {'headers': {'Cache-Control':'no-store'}, 'page_config_hook':private_page_config}\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd,'w') as stream:
        stream.write(text)
    path.chmod(0o600)
    return path


def make_config(password, runtime, workspace):
    token = secrets.token_urlsafe(48)
    settings = dict(password_hash=password_hash(password), session_key=secrets.token_hex(48),
                    workspace=str(workspace), secure_cookies=True,
                    comfy_upstream='http://127.0.0.1:8189', jupyter_upstream='http://127.0.0.1:8889',
                    jupyter_token=token, comfy_port=8188, jupyter_port=8888,
                    public_origins=public_origins())
    gateway = private_json(runtime / 'gateway.json', settings)
    jupyter = write_jupyter_proxy_config(runtime, workspace, token)
    return gateway, jupyter


def main():
    password = os.environ.get('COMFY_PASSWORD', '')
    validate_password(password)
    workspace = Path('/workspace')
    workspace.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ.get('H3_STILL_DATA_ROOT', '/workspace/h3-still')).resolve()
    if root == workspace or not root.is_relative_to(workspace):
        raise ValueError('H3_STILL_DATA_ROOT must be inside /workspace.')
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    runtime = Path('/run/h3-studio-v2')
    gate_config, jupyter_config = make_config(password, runtime, workspace)
    env = terminal_environment()
    env['JUPYTER_RUNTIME_DIR'] = str(runtime / 'jupyter-runtime')
    env['JUPYTER_CONFIG_DIR'] = str(runtime)
    env['JUPYTER_CONFIG_PATH'] = str(runtime)
    env['SHELL'] = '/bin/bash'
    comfy_env = dict(os.environ)
    comfy_env.pop('COMFY_PASSWORD', None)
    children, files = [], []
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def start(label, command, child_env):
        log = (logs / (label + '.log')).open('a', buffering=1)
        log.write('\n=== H3 Studio v2 startup %s ===\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
        files.append(log)
        child = subprocess.Popen(command, cwd=workspace, env=child_env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        children.append((label, child))
        return child
    try:
        gateway = start('access', [sys.executable, str(PACKAGE / 'runtime/access_gateway.py'),
                                   '--config', str(gate_config)], env)
        start('jupyter', [sys.executable, '-m', 'jupyterlab', '--config', str(jupyter_config)], env)
        start('comfy', [sys.executable, str(PACKAGE / 'runtime/media_backend.py')], comfy_env)
        password = ''
        comfy_env.clear()
        for key in list(os.environ):
            if key not in env and key.upper().endswith(('_TOKEN', '_PASSWORD', '_SECRET', '_API_KEY')):
                os.environ.pop(key, None)
        print('H3 Studio v2: 8188 = ComfyUI sign-in; 8888 = Files/Jupyter sign-in. Use comfy_password; no username.', flush=True)
        print('Fresh access path on either service: /__h3/login-v3 . No manual UTM parameter is required by this app.', flush=True)
        print('The file browser stays available while models load. Logs: /workspace/h3-still/logs/', flush=True)
        print('This does not stop GPU billing. Stop the Pod in RunPod when finished.', flush=True)
        reported = set()
        while not stopping:
            if gateway.poll() is not None:
                raise RuntimeError('Access gateway exited. Read /workspace/h3-still/logs/access.log in RunPod logs/storage.')
            for name, child in children:
                if child.poll() is not None and name not in reported:
                    print('%s backend exited with %s. Login/files remain available; inspect logs/%s.log. GPU billing continues.' %
                          (name, child.returncode, name), file=sys.stderr, flush=True)
                    reported.add(name)
            time.sleep(1)
        return 0
    finally:
        for _, child in reversed(children):
            stop_process(child)
        for stream in files:
            stream.close()

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('H3 Studio startup failed:', str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
