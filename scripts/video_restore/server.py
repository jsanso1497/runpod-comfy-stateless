"""Single-user, password-protected restoration UI using Python's standard library.

Designed for a disposable RunPod test, not a multi-tenant/public production service.
The RunPod HTTPS proxy provides TLS. Never publish the raw HTTP port to the internet.
"""
from __future__ import annotations
import base64
from concurrent.futures import ThreadPoolExecutor
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse
import uuid

import engine

WEB = Path(__file__).resolve().parent / 'web'


class State:
    def __init__(self, initialize=True):
        self.lock = threading.RLock()
        self.csrf = secrets.token_urlsafe(32)
        self.ready = False
        self.stage = 'Starting hardware checks'
        self.error = None
        self.hardware = {}
        self.weights = {}
        self.jobs = {}
        self.active = None
        self.cancel = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='restore')
        for name in ['uploads', 'jobs', 'models']:
            (engine.HOME / name).mkdir(parents=True, exist_ok=True)
        if initialize:
            threading.Thread(target=self.initialize, daemon=True).start()

    def initialize(self):
        try:
            result = subprocess.run([sys.executable, str(engine.APP / 'hardware.py')],
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            print(result.stdout, flush=True)
            lines = [line for line in result.stdout.splitlines() if line.startswith('HARDWARE_JSON=')]
            if not lines:
                raise RuntimeError('Hardware check failed. Inspect the Pod log.')
            self.hardware = json.loads(lines[-1].split('=', 1)[1])
            if result.returncode or self.hardware.get('error'):
                raise RuntimeError(self.hardware.get('error', 'CUDA hardware check failed.'))
            self.weights = engine.ensure_models(os.environ.get('RESTORE_MODEL', '7b'), self.set_stage)
            with self.lock:
                self.ready = True
                self.stage = 'Ready. Start with a three-second preview.'
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.stage = 'Startup failed'
            print(f'STARTUP ERROR: {exc}', flush=True)

    def set_stage(self, text):
        with self.lock:
            self.stage = text
        print(text, flush=True)

    def snapshot(self):
        with self.lock:
            active = self.jobs.get(self.active) if self.active else None
            return {'csrf': self.csrf, 'ready': self.ready, 'stage': self.stage,
                    'error': self.error, 'hardware': self.hardware,
                    'model': os.environ.get('RESTORE_MODEL', '7b'), 'active': active}

    def submit(self, upload_id, options):
        if not re.fullmatch(r'[0-9a-f]{32}', upload_id or ''):
            raise ValueError('Invalid upload ID.')
        metadata = engine.HOME / 'uploads' / f'{upload_id}.json'
        if not metadata.exists():
            raise ValueError('Upload not found. Upload the clip again.')
        info = json.loads(metadata.read_text())
        source = Path(info['path'])
        with self.lock:
            if not self.ready:
                raise ValueError('Startup checks/model downloads have not completed.')
            if self.active and self.jobs[self.active]['status'] == 'running':
                raise ValueError('One job is already running. Wait or cancel it first.')
            job_id = uuid.uuid4().hex
            job = engine.HOME / 'jobs' / job_id
            job.mkdir()
            self.cancel = threading.Event()
            self.active = job_id
            self.jobs[job_id] = {'id': job_id, 'status': 'running', 'outputs': [], 'error': None}
            self.executor.submit(self._run, job_id, source, job, options, self.cancel)
            return job_id

    def _run(self, job_id, source, job, options, cancel):
        try:
            report = engine.run_test(source, job, options, self.weights, self.hardware,
                                     cancel=cancel, progress=self.set_stage)
            with self.lock:
                self.jobs[job_id].update(status='complete', outputs=report['outputs'], report=report)
        except Exception as exc:
            status = 'cancelled' if isinstance(exc, engine.Cancelled) else 'error'
            report_path = job / 'report.json'
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            report.update(status=status, error=str(exc))
            report_path.write_text(json.dumps(report, indent=2))
            with self.lock:
                self.jobs[job_id].update(status=status, error=str(exc),
                    outputs=[p.name for p in job.iterdir() if p.is_file() and
                             (p.name.endswith('.mp4') or p.name in {'report.json', 'run.log'})])
            self.set_stage(f'{status.capitalize()}. The Pod remains running and billable.')

    def get_job(self, job_id):
        with self.lock:
            item = self.jobs.get(job_id)
            if not item:
                return None
            result = dict(item)
        log = engine.HOME / 'jobs' / job_id / 'run.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size - 16000))
                result['log_tail'] = f.read().decode(errors='replace')
        else:
            result['log_tail'] = ''
        return result


class Handler(BaseHTTPRequestHandler):
    server_version = 'VideoRestore/1.0'
    protocol_version = 'HTTP/1.1'

    @property
    def state(self):
        return self.server.state

    def security_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")

    def respond(self, value, code=200):
        body = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.security_headers()
        self.end_headers()
        self.wfile.write(body)

    def authenticated(self):
        expected = f'{self.server.username}:{self.server.password}'
        header = self.headers.get('Authorization', '')
        try:
            value = base64.b64decode(header[6:], validate=True).decode() if header.startswith('Basic ') else ''
        except Exception:
            value = ''
        if hmac.compare_digest(value.encode(), expected.encode()):
            return True
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="Private video restoration", charset="UTF-8"')
        self.send_header('Content-Length', '0')
        self.security_headers()
        self.end_headers()
        return False

    def file_response(self, path, download=False):
        size = path.stat().st_size
        start, end = 0, size - 1
        partial = False
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
            if not match:
                self.respond({'error': 'Unsupported byte range'}, 416); return
            start = int(match[1]); end = min(int(match[2]) if match[2] else end, end)
            if start > end or start < 0:
                self.respond({'error': 'Invalid byte range'}, 416); return
            partial = True
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        if partial:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{path.name}"')
        self.security_headers()
        self.end_headers()
        if self.command == 'HEAD':
            return
        with path.open('rb') as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                block = f.read(min(1024 * 1024, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route == '/healthz':
            self.respond({'alive': True, 'ready': self.state.ready}); return
        if not self.authenticated():
            return
        try:
            static = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            if route in static:
                self.file_response(WEB / static[route]); return
            if route == '/api/state':
                self.respond(self.state.snapshot()); return
            match = re.fullmatch(r'/api/job/([0-9a-f]{32})', route)
            if match:
                item = self.state.get_job(match[1])
                self.respond(item or {'error': 'Job not found'}, 200 if item else 404); return
            match = re.fullmatch(r'/files/([0-9a-f]{32})/([A-Za-z0-9_.-]+)', route)
            if match:
                item = self.state.get_job(match[1])
                if not item or match[2] not in item.get('outputs', []):
                    self.respond({'error': 'Output not found'}, 404); return
                path = engine.HOME / 'jobs' / match[1] / match[2]
                self.file_response(path, download='download' in parse_qs(parsed.query)); return
            self.respond({'error': 'Not found'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.respond({'error': str(exc)}, 500)

    def do_POST(self):
        if not self.authenticated():
            self.close_connection = True
            return
        if not hmac.compare_digest(self.headers.get('X-CSRF-Token', ''), self.state.csrf):
            self.respond({'error': 'Refresh this page before submitting.'}, 403)
            self.close_connection = True
            return
        route = urlparse(self.path)
        try:
            content_length = int(self.headers.get('Content-Length', '-1'))
            if content_length < 0:
                raise ValueError('Content-Length is required.')
            if route.path == '/api/upload':
                self.upload(route, content_length); return
            if content_length > 65536:
                raise ValueError('Request is too large.')
            body = json.loads(self.rfile.read(content_length) or b'{}')
            if route.path == '/api/run':
                job_id = self.state.submit(body.get('upload_id'), body.get('options', {}))
                self.respond({'id': job_id}, 202); return
            if route.path == '/api/cancel':
                self.state.cancel.set()
                self.respond({'message': 'Cancellation requested. This does not stop Pod billing.'}); return
            if route.path == '/api/delete':
                job_id = body.get('id', '')
                with self.state.lock:
                    item = self.state.jobs.get(job_id)
                    if not item or item['status'] == 'running':
                        raise ValueError('Cannot delete this job while it is running.')
                    shutil.rmtree(engine.HOME / 'jobs' / job_id)
                    del self.state.jobs[job_id]
                    if self.state.active == job_id:
                        self.state.active = None
                self.respond({'deleted': True}); return
            self.respond({'error': 'Not found'}, 404)
        except Exception as exc:
            self.respond({'error': str(exc)}, 400)
            self.close_connection = True

    def upload(self, route, size):
        max_bytes = int(os.environ.get('RESTORE_MAX_UPLOAD_GB', '5')) * 2**30
        if not 0 < size <= max_bytes:
            raise ValueError('Upload must be nonempty and no larger than the configured 5 GiB limit.')
        name = Path(parse_qs(route.query).get('name', ['video.mp4'])[0]).name
        extension = Path(name).suffix.lower()
        if extension not in engine.ALLOWED_EXTENSIONS:
            raise ValueError('Unsupported video file extension.')
        if shutil.disk_usage(engine.HOME).free < size + 10 * 2**30:
            raise ValueError('Not enough free container disk for this upload.')
        uid = uuid.uuid4().hex
        target = engine.HOME / 'uploads' / f'{uid}{extension}'
        temporary = target.with_suffix(extension + '.part')
        remaining = size
        self.connection.settimeout(180)
        try:
            with temporary.open('wb') as f:
                while remaining:
                    block = self.rfile.read(min(4 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError('Upload was interrupted.')
                    f.write(block); remaining -= len(block)
            temporary.replace(target)
            metadata = engine.validate_source(target)
            info = {'id': uid, 'name': name, 'path': str(target), 'duration': metadata['duration'],
                    'fps': metadata['fps'], 'warnings': metadata['warnings']}
            (engine.HOME / 'uploads' / f'{uid}.json').write_text(json.dumps(info))
            self.respond({k: v for k, v in info.items() if k != 'path'})
        except Exception:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise

    def log_message(self, fmt, *args):
        # Never log authorization headers or request bodies.
        print(f'HTTP {self.address_string()} {fmt % args}', flush=True)


def make_server(host='0.0.0.0', port=8188, initialize=True):
    password = os.environ.get('RESTORE_PASSWORD', '')
    if len(password) < 12:
        raise RuntimeError('Set RESTORE_PASSWORD to a private password of at least 12 characters using a RunPod Secret.')
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.username = os.environ.get('RESTORE_USERNAME', 'james')
    server.password = password
    server.state = State(initialize=initialize)
    return server

if __name__ == '__main__':
    os.umask(0o077)
    instance = make_server(port=int(os.environ.get('RESTORE_PORT', '8188')))
    print('Video restoration UI listening. Use the RunPod HTTPS connection for port 8188.', flush=True)
    print('Login username: ' + instance.username, flush=True)
    print('All input, output and model files are on disposable container storage.', flush=True)
    try:
        instance.serve_forever()
    except KeyboardInterrupt:
        instance.state.cancel.set()
    finally:
        instance.server_close()
