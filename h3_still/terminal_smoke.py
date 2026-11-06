"""Start real Jupyter, verify password/XSRF auth, then execute a terminal command.

Runs without CUDA or model files. Uses only a temporary directory and test secret.
"""
from __future__ import annotations
import importlib.metadata as md
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE/'runtime'))
from recovery_launcher import write_jupyter_config,terminal_environment,stop_process


def main() -> None:
    import requests
    import websocket
    with tempfile.TemporaryDirectory(prefix='h3-jupyter-test-') as temp:
        root=Path(temp);work=root/'work';work.mkdir()
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        password='ci-only-H3-terminal-test-secret'
        cfg=write_jupyter_config(password,root/'runtime',work,port,'127.0.0.1',False)
        assert password not in cfg.read_text()
        assert cfg.stat().st_mode&0o777==0o600
        env=terminal_environment();env['JUPYTER_RUNTIME_DIR']=str(root/'runtime/run')
        env['JUPYTER_CONFIG_DIR']=str(root/'runtime');env['JUPYTER_CONFIG_PATH']=str(root/'runtime')
        base=f'http://127.0.0.1:{port}';log=root/'log.txt'
        child=None
        try:
            with log.open('w') as stream:
                child=subprocess.Popen([sys.executable,'-m','jupyterlab','--config',str(cfg)],
                                       cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                client=requests.Session();client.trust_env=False
                until=time.monotonic()+90
                while time.monotonic()<until:
                    if child.poll() is not None:raise RuntimeError('Jupyter exited')
                    try:
                        r=client.get(base+'/login',timeout=3)
                        if r.status_code==200:break
                    except requests.RequestException:pass
                    time.sleep(.5)
                else:raise RuntimeError('Jupyter readiness timed out')
                for suffix in ('/api/terminals','/api/contents','/api/terminals?utm_source=chatgpt.com'):
                    r=client.get(base+suffix,allow_redirects=False,timeout=5)
                    assert r.status_code in (302,401,403),(suffix,r.status_code)
                xsrf=client.cookies.get('_xsrf').strip('"')
                client.post(base+'/login?next=%2Flab',data={'password':'wrong-password','_xsrf':xsrf},timeout=10)
                assert client.get(base+'/api/terminals',allow_redirects=False,timeout=5).status_code in (302,401,403)
                client.post(base+'/login?next=%2Flab',data={'password':password,'_xsrf':xsrf},timeout=15)
                assert client.get(base+'/api/terminals',timeout=5).status_code==200
                assert client.post(base+'/api/terminals',json={},timeout=5).status_code==403
                headers={'X-XSRFToken':client.cookies.get('_xsrf').strip('"')}
                result=client.post(base+'/api/terminals',json={},headers=headers,timeout=10)
                assert result.status_code==200,result.text
                name=result.json()['name']
                cookie='; '.join(f'{k}={v}' for k,v in client.cookies.get_dict().items())
                ws=websocket.create_connection(f'ws://127.0.0.1:{port}/terminals/websocket/{name}',
                                               cookie=cookie,origin=base,timeout=10)
                try:
                    # Confirm a command ran, not just echoed input.
                    ws.send(json.dumps(['stdin',"printf '%s' h3-terminal-ok > h3-terminal-check.txt\r"]))
                    end=time.monotonic()+12
                    while time.monotonic()<end:
                        if (work/'h3-terminal-check.txt').is_file():break
                        ws.settimeout(1)
                        try:ws.recv()
                        except websocket.WebSocketTimeoutException:pass
                    assert (work/'h3-terminal-check.txt').read_text()=='h3-terminal-ok'
                finally:ws.close()
                client.delete(base+'/api/terminals/'+name,headers=headers,timeout=5).raise_for_status()
                assert client.get(base+'/api/contents/h3-terminal-check.txt',timeout=5).status_code==200
                client.close()
            report={'jupyterlab':md.version('jupyterlab'),'jupyter_server':md.version('jupyter_server'),
                    'unauthenticated_requests_denied':True,'wrong_password_denied':True,
                    'correct_password_login':True,'xsrf_enforced':True,'real_terminal_command':True,
                    'file_browser_read':True,'gpu_inference':False}
            (PACKAGE/'terminal-smoke-report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
        except Exception:
            if log.exists():print(log.read_text()[-10000:],file=sys.stderr)
            raise
        finally:
            if child is not None:stop_process(child)

if __name__=='__main__':main()
