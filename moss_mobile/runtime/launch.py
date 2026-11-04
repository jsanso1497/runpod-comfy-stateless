"""Authenticated single-user ComfyUI. Does not stop RunPod billing when idle."""
from __future__ import annotations
import json
import grp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from bootstrap import PACKAGE, prepare

NGINX = r'''
user www-data;
worker_processes 1;
pid /run/moss-mobile/nginx.pid;
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
    client_max_body_size 100m;
    auth_basic "MOSS Mobile";
    auth_basic_user_file /run/moss-mobile/passwords;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy same-origin always;
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
        raise ValueError('COMFY_PASSWORD must contain 16-72 UTF-8 bytes, with no newlines. Use a RunPod secret.')
    if '{{' in password or 'RUNPOD_SECRET' in password:
        raise ValueError('COMFY_PASSWORD contains an unresolved RunPod secret reference.')


def main() -> int:
    password = os.environ.get('COMFY_PASSWORD', '')
    validate_password(password)
    root = Path(os.environ.get('MOSS_DATA_ROOT', '/workspace/moss-mobile')).resolve()
    if not str(root).startswith('/workspace/'):
        raise ValueError('MOSS_DATA_ROOT must be inside the persistent /workspace mount.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU is available. Start this Pod with a GPU, not CPU-only mode.')
    print('MOSS Mobile HQ. Use the HTTPS RunPod proxy on port 8188. Username: workbench.', flush=True)
    print('First start downloads about 30.3 GB. Later starts reuse persistent verified files.', flush=True)
    print('GPU billing continues while this Pod runs. Stop the Pod in RunPod when finished.', flush=True)
    prepare(root)
    runtime = Path('/run/moss-mobile')
    runtime.mkdir(parents=True, exist_ok=True, mode=0o750)
    group = grp.getgrnam('www-data').gr_gid
    os.chown(runtime, 0, group)
    runtime.chmod(0o750)
    result = subprocess.run(['htpasswd', '-niB', 'workbench'], input=password+'\n', text=True,
                            capture_output=True, check=True)
    auth = runtime / 'passwords'
    auth.write_text(result.stdout, encoding='utf-8')
    os.chown(auth, 0, group)
    auth.chmod(0o640)
    config = runtime / 'nginx.conf'
    config.write_text(NGINX, encoding='utf-8')
    # Nginx does not inherit the plaintext password or model service token.
    child_env = os.environ.copy()
    child_env.pop('COMFY_PASSWORD', None)
    child_env.pop('HF_TOKEN', None)
    os.environ.pop('COMFY_PASSWORD', None)
    password = ''
    comfy_root = Path('/opt/ComfyUI')
    command = [sys.executable, str(comfy_root/'main.py'), '--listen', '127.0.0.1', '--port', '8189',
               '--input-directory', str(root/'input'), '--output-directory', str(root/'output'),
               '--user-directory', str(root/'user'), '--temp-directory', str(root/'temp'),
               '--extra-model-paths-config', str(root/'extra_model_paths.yaml'),
               '--preview-method', 'none', '--disable-auto-launch']
    # Do not force low-VRAM/offload, quantization, compilation or approximate attention.
    # Native ComfyUI memory management handles the selected GPU.
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
        comfy = subprocess.Popen(command, cwd=comfy_root, env=child_env)
        children.append(comfy)
        deadline = time.monotonic() + 240
        while not stopping and time.monotonic() < deadline:
            if comfy.poll() is not None:
                raise RuntimeError('ComfyUI exited during startup. Read the preceding container logs.')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8189/object_info', timeout=3) as response:
                    object_info = json.load(response)
                break
            except (OSError, ValueError):
                time.sleep(2)
        else:
            raise RuntimeError('ComfyUI did not become ready.')
        # Fail closed before exposing a broken node installation to mobile users.
        sys.path.insert(0, str(PACKAGE))
        from validate_live import validate
        issues = validate(object_info)
        if issues:
            raise RuntimeError('Node/model compatibility check failed:\n' + '\n'.join(issues))
        subprocess.run(['nginx', '-t', '-c', str(config)], check=True, env=child_env)
        proxy = subprocess.Popen(['nginx', '-c', str(config), '-g', 'daemon off;'], env=child_env)
        children.append(proxy)
        print('Ready: open HTTPS 8188, sign in, then Workflows > 00_MOSS_Mobile_HQ_v1.', flush=True)
        while not stopping:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('A service exited. Stopping the remaining service; review logs.')
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
        print('MOSS Mobile startup failed:', str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
