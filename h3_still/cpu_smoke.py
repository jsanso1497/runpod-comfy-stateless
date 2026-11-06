"""Real ComfyUI CPU startup and authenticated proxy check, without model weights.

The empty model-name fixtures are only for loader dropdown/schema validation.
Native image resizing is submitted; no model inference is submitted. This test must never be described as a GPU test.
"""
from __future__ import annotations
import base64
import http.client
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from validate_live import validate

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'runtime'))
from launch import prepare_auth, clean_environment


def read(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read()


def main():
    import yaml
    children = []
    with tempfile.TemporaryDirectory(prefix='h3-still-cpu-') as folder:
        root = Path(folder)
        for name in ('models', 'input', 'output', 'user', 'temp'):
            (root / name).mkdir()
        for row in json.loads((PACKAGE / 'models.lock.json').read_text())['models']:
            dest = root / 'models' / row['destination']
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.touch()
        paths = root / 'extra.yaml'
        paths.write_text(yaml.safe_dump({'cpu_fixture': {'base_path': str(root / 'models'),
            'is_default': True, **{key: key for key in ('diffusion_models', 'text_encoders', 'vae', 'loras')}}}))
        log_path = root / 'comfy.log'
        try:
            with log_path.open('w') as log:
                command = [sys.executable, '/opt/ComfyUI/main.py', '--cpu', '--listen', '127.0.0.1',
                    '--port', '8189', '--input-directory', str(root / 'input'),
                    '--output-directory', str(root / 'output'), '--user-directory', str(root / 'user'),
                    '--temp-directory', str(root / 'temp'), '--extra-model-paths-config', str(paths),
                    '--disable-auto-launch', '--preview-method', 'none']
                comfy = subprocess.Popen(command, cwd='/opt/ComfyUI', stdout=log,
                                         stderr=subprocess.STDOUT, env=clean_environment())
                children.append(comfy)
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    if comfy.poll() is not None:
                        raise RuntimeError('CPU ComfyUI process exited before readiness.')
                    try:
                        _, body = read('http://127.0.0.1:8189/object_info')
                        info = json.loads(body)
                        break
                    except (OSError, ValueError):
                        time.sleep(2)
                else:
                    raise RuntimeError('CPU startup timed out.')
                errors = validate(info)
                if errors:
                    raise RuntimeError('\n'.join(errors))
                from core_resize_smoke import run as test_native_resize
                resize_report = test_native_resize(root)
                from studio_smoke import run as test_studio
                studio_report = test_studio(root, info)
                password = 'ci-only-temporary-h3-still-credential'
                config = prepare_auth(password)
                subprocess.run(['nginx', '-t', '-c', str(config)], check=True)
                proxy = subprocess.Popen(['nginx', '-c', str(config), '-g', 'daemon off;'], env=clean_environment())
                children.append(proxy)
                token = base64.b64encode(('workbench:' + password).encode()).decode()
                for _ in range(30):
                    try:
                        read('http://127.0.0.1:8188/system_stats', {'Authorization': 'Basic ' + token})
                        break
                    except OSError:
                        time.sleep(0.2)
                else:
                    raise RuntimeError('Authenticated proxy never became ready.')
                for path in ('/', '/object_info', '/?utm_source=chatgpt.com', '/ws?clientId=auth-test'):
                    for headers in ({}, {'Authorization': 'Basic ' + base64.b64encode(b'workbench:wrong-password').decode()}):
                        try:
                            read('http://127.0.0.1:8188' + path, headers)
                        except urllib.error.HTTPError as error:
                            if error.code != 401:
                                raise RuntimeError('Expected a 401 for unauthenticated path: ' + path)
                        else:
                            raise RuntimeError('Authentication was bypassed: ' + path)
                status, _ = read('http://127.0.0.1:8188/object_info', {'Authorization': 'Basic ' + token})
                if status != 200:
                    raise RuntimeError('Authenticated request failed.')
                connection = http.client.HTTPConnection('127.0.0.1', 8188, timeout=10)
                try:
                    connection.request('GET', '/ws?clientId=h3-still-ci', headers={
                        'Authorization': 'Basic ' + token, 'Connection': 'Upgrade',
                        'Upgrade': 'websocket', 'Sec-WebSocket-Version': '13',
                        'Sec-WebSocket-Key': base64.b64encode(os.urandom(16)).decode()})
                    if connection.getresponse().status != 101:
                        raise RuntimeError('Authenticated WebSocket upgrade failed.')
                finally:
                    connection.close()
                result = {'cpu_startup': 'passed', 'node_schema': 'passed', 'authenticated_proxy': 'passed', 'websocket_upgrade': 'passed',
                          'weights_loaded': False, 'gpu_inference': 'not tested', 'image_quality': 'not tested', 'native_resize':resize_report, 'studio_inputs':studio_report}
                (PACKAGE / 'cpu-smoke-report.json').write_text(json.dumps(result, indent=2)+'\n')
                print(json.dumps(result, indent=2))
        except Exception:
            if log_path.exists():
                print(log_path.read_text(errors='replace')[-16000:], file=sys.stderr)
            raise
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == '__main__':
    main()
