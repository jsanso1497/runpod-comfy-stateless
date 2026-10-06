#!/usr/bin/env python3
"""Live local HTTP smoke test, no GPU or external network needed.

Run with the file-manager venv Python during image build. Exercises the real
Jupyter server: anonymous denial, password login, CSRF protection, file upload,
download, edit and delete. Uses only a disposable temporary workspace.
"""
from __future__ import annotations
import importlib.metadata
import http.cookiejar
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import build_opener, HTTPCookieProcessor, ProxyHandler, Request


def main():
    print('FILE MANAGER TEST VERSIONS: JupyterLab '+importlib.metadata.version('jupyterlab')+'; Jupyter Server '+importlib.metadata.version('jupyter_server'),flush=True)
    with tempfile.TemporaryDirectory(prefix='file-manager-test-') as t:
        base=Path(t);root=base/'workspace';root.mkdir()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        password=secrets.token_urlsafe(24)
        env=dict(os.environ,FILEBROWSER_PASSWORD=password)
        jar=http.cookiejar.CookieJar();opener=build_opener(ProxyHandler({}),HTTPCookieProcessor(jar))
        def request(path, data=None, method=None, csrf=False):
            headers={'Referer':f'http://127.0.0.1:{port}/lab'}
            if csrf:
                headers['X-XSRFToken']=next(c.value for c in jar if c.name=='_xsrf')
            if isinstance(data,dict):
                data=json.dumps(data).encode();headers['Content-Type']='application/json'
            return opener.open(Request(f'http://127.0.0.1:{port}'+path,data=data,headers=headers,method=method),timeout=10)
        with open(base/'server.log','w+') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('service.py')),'run','--root',str(root),'--state',str(base/'private'),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                deadline=time.monotonic()+60
                while time.monotonic()<deadline:
                    if proc.poll() is not None:raise RuntimeError('File manager exited before readiness')
                    try:
                        with request('/login') as r:
                            assert r.status==200
                        break
                    except (URLError,OSError):time.sleep(.2)
                else:raise RuntimeError('File manager startup timeout')
                try:
                    request('/api/contents')
                    raise AssertionError('Anonymous file access was allowed')
                except HTTPError as e:assert e.code in (401,403)
                xsrf=next(c.value for c in jar if c.name=='_xsrf')
                data=urlencode({'password':password,'_xsrf':xsrf}).encode()
                with request('/login?next=%2Flab',data=data,method='POST') as r:
                    assert r.status==200
                with request('/api/contents') as r:assert r.status==200
                payload={'type':'file','format':'text','content':'upload check\n'}
                try:
                    request('/api/contents/no-csrf.txt',payload,'PUT')
                    raise AssertionError('Write without CSRF token was allowed')
                except HTTPError as e:assert e.code==403
                with request('/api/contents/probe.txt',payload,'PUT',csrf=True) as r:assert r.status==201
                with request('/files/probe.txt') as r:assert r.read()==b'upload check\n'
                payload['content']='edited\n'
                with request('/api/contents/probe.txt',payload,'PUT',csrf=True) as r:assert r.status==200
                assert (root/'probe.txt').read_text()=='edited\n'
                with request('/api/contents/probe.txt',method='DELETE',csrf=True) as r:assert r.status==204
                assert not (root/'probe.txt').exists()
                print('FILE MANAGER HTTP SMOKE PASS: login, anonymous denial, CSRF, upload, download, edit, delete.',flush=True)
            except Exception:
                log.flush();log.seek(0)
                print(log.read().replace(password,'[REDACTED]'),file=sys.stderr)
                raise
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)


if __name__=='__main__':main()
