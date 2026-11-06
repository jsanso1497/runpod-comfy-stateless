"""Start an authenticated Jupyter terminal before starting model preparation.

This is separate from RunPod's managed Web Terminal service. If ComfyUI fails,
the diagnostic terminal stays available and the container healthcheck fails.
It does not stop GPU billing. No model, sampler or authentication bypass is added.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from launch import validate_password

PACKAGE = Path(__file__).resolve().parents[1]


def terminal_environment() -> dict[str, str]:
    """Do not inherit model-service credentials into Jupyter or its terminals."""
    result = dict(os.environ)
    exact = {'WB_LORA_URLS','WB_CHECKPOINTS','JUPYTER_CONFIG_PATH','JUPYTER_TOKEN_FILE'}
    for name in list(result):
        upper = name.upper()
        if name in exact or upper.endswith(('_TOKEN','_PASSWORD','_SECRET','_API_KEY')) or upper.startswith('RUNPOD_SECRET_'):
            result.pop(name, None)
    result['TERM'] = 'xterm-256color'
    return result


def write_jupyter_config(password: str, runtime: Path, root: Path,
                         port: int = 8888, ip: str = '0.0.0.0', secure: bool = True) -> Path:
    validate_password(password)
    from jupyter_server.auth import passwd
    runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime.chmod(0o700)
    path = runtime / 'jupyter_server_config.json'
    if path.is_symlink():
        raise ValueError('Refusing a symlinked Jupyter config.')
    settings = {
        'ServerApp': {
            'ip':ip, 'port':port, 'port_retries':0, 'open_browser':False,
            'allow_root':True, 'allow_remote_access':True, 'trust_xheaders':True,
            'root_dir':str(root), 'default_url':'/lab', 'terminals_enabled':True,
            'disable_check_xsrf':False,
            'terminado_settings':{'shell_command':['/bin/bash','-l']},
            'cookie_options':{'secure':secure,'samesite':'Lax'},
        },
        'IdentityProvider':{'token':''},
        'PasswordIdentityProvider':{'hashed_password':passwd(password), 'password_required':True},
    }
    fd=os.open(path, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as stream:
        json.dump(settings,stream,indent=2);stream.write('\n')
    path.chmod(0o600)
    return path


def stop_process(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    # Child sessions include their shell/Comfy/nginx descendants.
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()


def main() -> int:
    password=os.environ.get('COMFY_PASSWORD','')
    validate_password(password)
    workspace=Path('/workspace')
    workspace.mkdir(parents=True,exist_ok=True)
    runtime=Path('/run/h3-still-jupyter')
    config=write_jupyter_config(password,runtime,workspace)
    env=terminal_environment()
    env['JUPYTER_RUNTIME_DIR']=str(runtime/'runtime')
    env['JUPYTER_CONFIG_DIR']=str(runtime)
    env['JUPYTER_CONFIG_PATH']=str(runtime)
    children=[];stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        jupyter=subprocess.Popen([sys.executable,'-m','jupyterlab','--config',str(config)],
                                 cwd=workspace,env=env,start_new_session=True)
        children.append(jupyter)
        deadline=time.monotonic()+120
        while not stopping and time.monotonic()<deadline:
            if jupyter.poll() is not None:
                raise RuntimeError('Jupyter exited before readiness. Check the container logs.')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8888/login',timeout=3) as response:
                    if response.status==200:break
            except OSError:time.sleep(1)
        else:
            if stopping:return 0
            raise RuntimeError('Jupyter login service did not become ready.')
        print('Recovery terminal ready: RunPod HTTPS port 8888 > log in with your comfy_password > Launcher > Terminal.',flush=True)
        print('ComfyUI starts separately on HTTPS 8188 after model checks. GPU billing continues until you stop the Pod.',flush=True)
        comfy_env=dict(os.environ)
        comfy=subprocess.Popen([sys.executable,str(PACKAGE/'runtime/launch.py')],
                               cwd=workspace,env=comfy_env,start_new_session=True)
        children.append(comfy)
        password='';comfy_env.clear()
        for name in list(os.environ):
            if name not in env and name.upper().endswith(('_TOKEN','_PASSWORD','_SECRET','_API_KEY')):
                os.environ.pop(name,None)
        reported=False
        while not stopping:
            if jupyter.poll() is not None:
                raise RuntimeError('Jupyter exited. Stopping this container; inspect its logs.')
            if comfy.poll() is not None and not reported:
                print('COMFYUI STOPPED (exit %s). Jupyter on 8888 remains available for diagnosis. '
                      'This does not stop GPU charges. Check RunPod Logs, then stop/restart the Pod after correcting the error.' % comfy.returncode,
                      file=sys.stderr,flush=True)
                reported=True
            time.sleep(1)
        return 0
    finally:
        for child in reversed(children):stop_process(child)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print('H3 recovery startup failed:',str(error),file=sys.stderr,flush=True)
        raise SystemExit(1)
