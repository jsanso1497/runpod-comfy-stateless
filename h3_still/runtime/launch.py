"""Single-user authenticated RunPod entrypoint for the isolated H3 Still image."""
from __future__ import annotations
import grp
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from bootstrap import PACKAGE, FACTORY, prepare

NGINX = r'''
user www-data;
worker_processes 1;
pid /run/h3-still/nginx.pid;
error_log /dev/stderr warn;
events { worker_connections 1024; }
http {
  include /etc/nginx/mime.types;
  default_type application/octet-stream;
  access_log off;
  server_tokens off;
  map $http_upgrade $connection_upgrade { default upgrade; '' close; }
  server {
    listen 8188;
    client_max_body_size 250m;
    auth_basic "H3 Still Workspace";
    auth_basic_user_file /run/h3-still/passwords;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy same-origin always;
    add_header Cache-Control "no-store" always;
    location / {
      proxy_pass http://127.0.0.1:8189;
      proxy_http_version 1.1;
      proxy_set_header Host $http_host;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection $connection_upgrade;
      proxy_set_header X-Forwarded-Proto https;
      proxy_set_header Authorization "";
      proxy_read_timeout 3600s;
      proxy_send_timeout 3600s;
      proxy_buffering off;
      proxy_request_buffering off;
    }
  }
}
'''


def validate_password(password: str) -> None:
    if not 16 <= len(password.encode('utf-8')) <= 72 or any(c in password for c in ('\r', '\n', '\x00')):
        raise ValueError('COMFY_PASSWORD must contain 16-72 UTF-8 bytes with no newlines or nulls.')
    if '{{' in password or 'RUNPOD_SECRET' in password:
        raise ValueError('COMFY_PASSWORD has an unresolved RunPod secret reference.')


def clean_environment() -> dict[str, str]:
    result = dict(os.environ)
    for key in ('COMFY_PASSWORD', 'WB_PASSWORD', 'HF_TOKEN', 'hf_token', 'CIVITAI_TOKEN',
                'civit_token', 'WB_LORA_URLS', 'WB_CHECKPOINTS'):
        result.pop(key, None)
    return result


def prepare_auth(password: str) -> Path:
    runtime = Path('/run/h3-still')
    runtime.mkdir(parents=True, exist_ok=True)
    group = grp.getgrnam('www-data').gr_gid
    os.chown(runtime, 0, group)
    runtime.chmod(0o750)
    result = subprocess.run(['htpasswd', '-niB', 'workbench'], input=password+'\n',
                            capture_output=True, text=True, check=True)
    auth = runtime / 'passwords'
    auth.write_text(result.stdout, encoding='utf-8')
    os.chown(auth, 0, group)
    auth.chmod(0o640)
    config = runtime / 'nginx.conf'
    config.write_text(NGINX, encoding='utf-8')
    return config


def main() -> int:
    password = os.environ.get('COMFY_PASSWORD', '')
    validate_password(password)
    for key in ('HF_TOKEN',):
        if '{{' in os.environ.get(key, ''):
            raise ValueError(key + ' contains an unresolved secret reference.')
    root = Path(os.environ.get('H3_STILL_DATA_ROOT', '/workspace/h3-still')).resolve()
    if root == Path('/workspace') or not root.is_relative_to(Path('/workspace')):
        raise ValueError('H3_STILL_DATA_ROOT must be a subdirectory inside /workspace.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select a GPU host compatible with this CUDA 13.0 image.')
    print('H3 Still: developer INT8/ConvRot weights, native single-frame generation.', flush=True)
    print('Required Qwen3-VL encoder is not a prompt enhancer. No Ollama is installed.', flush=True)
    print('First startup prepares about 40 GB of weights. Public port opens after readiness checks.', flush=True)
    print('Stop the Pod through RunPod when finished. This image does not stop GPU billing.', flush=True)
    prepare(root)
    config = prepare_auth(password)
    child_env = clean_environment()
    password = ''
    os.environ.pop('COMFY_PASSWORD', None)
    os.environ.pop('HF_TOKEN', None)
    command = [sys.executable, '/opt/ComfyUI/main.py', '--listen', '127.0.0.1', '--port', '8189',
               '--input-directory', str(root / 'input'), '--output-directory', str(root / 'output'),
               '--user-directory', str(root / 'user'), '--temp-directory', str(root / 'temp'),
               '--extra-model-paths-config', str(root / 'extra_model_paths.yaml'),
               '--preview-method', 'none', '--disable-auto-launch']
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        comfy = subprocess.Popen(command, cwd='/opt/ComfyUI', env=child_env)
        children.append(comfy)
        deadline = time.monotonic() + 300
        info = None
        while not stopping and time.monotonic() < deadline:
            if comfy.poll() is not None:
                raise RuntimeError('ComfyUI exited during startup; check the preceding logs.')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8189/object_info', timeout=10) as response:
                    info = json.load(response)
                break
            except (OSError, ValueError):
                time.sleep(2)
        if stopping:
            return 0
        if info is None:
            raise RuntimeError('ComfyUI did not become ready.')
        sys.path.insert(0, str(PACKAGE))
        from validate_live import validate
        issues = validate(info)
        if issues:
            raise RuntimeError('Installed workflow compatibility check failed:\n' + '\n'.join(issues))
        subprocess.run(['nginx', '-t', '-c', str(config)], check=True, env=child_env)
        children.append(subprocess.Popen(['nginx', '-c', str(config), '-g', 'daemon off;'], env=child_env))
        print('Ready. RunPod HTTPS 8188. Username: workbench. Workflows > ' + FACTORY, flush=True)
        while not stopping:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('A required service exited; stopping the other service.')
            time.sleep(1)
        return 0
    finally:
        stop()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('H3 Still startup failed:', str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
