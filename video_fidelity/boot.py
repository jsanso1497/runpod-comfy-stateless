"""Only two locked weight files. Reuse matching existing SeedVR2 files when present."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
from fv_common import ROOT, atomic_json, sha256, terminate
from gateway import validate_password,create_app


def ensure_models(data: Path) -> None:
    from huggingface_hub import hf_hub_download
    models=data/'models'
    models.mkdir(parents=True,exist_ok=True)
    lock=json.loads((ROOT/'models.lock.json').read_text())
    for row in lock['models']:
        target=models/row['filename']
        if not target.exists():
            previous=Path('/workspace/models/SEEDVR2')/row['filename']
            if previous.is_file() and sha256(previous)==row['sha256']:
                target.symlink_to(previous)
                print('Reusing verified existing weight: '+row['filename'],flush=True)
            else:
                print('Preparing '+row['filename'],flush=True)
                hf_hub_download(repo_id=row['repo'],filename=row['filename'],revision=row['revision'],
                                local_dir=str(models),token=os.environ.get('HF_TOKEN') or None)
        if sha256(target)!=row['sha256']:
            raise RuntimeError('Model checksum mismatch; file was not deleted: '+str(target))
    atomic_json(data/'verified_models.json',lock)


def main() -> None:
    password=validate_password(os.environ.get('COMFY_PASSWORD',''))
    data=Path(os.environ.get('FAITHFUL_DATA_ROOT','/workspace/faithful-video')).resolve()
    for name in ('input','output','user','temp','jobs','logs'):
        (data/name).mkdir(parents=True,exist_ok=True)
    ensure_models(data)
    workflows=data/'user/default/workflows/00_Faithful_Video_v1'
    workflows.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'workflows').glob('*.json'):
        dest=workflows/source.name
        if not dest.exists():shutil.copy2(source,dest)
    # Logs persist, with bounded rotation between launches.
    log=data/'logs/comfy.log'
    if log.exists() and log.stat().st_size>10*1024**2:
        log.replace(log.with_suffix('.previous.log'))
    env=os.environ.copy()
    for key in list(env):
        if 'TOKEN' in key or key.endswith('PASSWORD') or 'SECRET' in key:
            env.pop(key,None)
    env.update(FAITHFUL_DATA_ROOT=str(data),HF_HUB_OFFLINE='1')
    command=[sys.executable,'-u','/opt/ComfyUI/main.py','--listen','127.0.0.1','--port','8189',
             '--input-directory',str(data/'input'),'--output-directory',str(data/'output'),
             '--user-directory',str(data/'user'),'--temp-directory',str(data/'temp'),
             '--cache-none','--disable-smart-memory','--disable-auto-launch']
    with log.open('ab',buffering=0) as handle:
        child=subprocess.Popen(command,env=env,cwd='/opt/ComfyUI',stdout=handle,
                               stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for _ in range(240):
                if child.poll() is not None:
                    raise RuntimeError(f'ComfyUI exited {child.returncode}. Inspect {log}')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8189/object_info',timeout=2) as response:
                        schema=json.load(response)
                    if not {'LoadVideo','FaithfulVideoRestore'} <= schema.keys():
                        raise RuntimeError('Required node schema missing. Inspect '+str(log))
                    break
                except (OSError,TimeoutError):time.sleep(1)
            else:
                raise RuntimeError('ComfyUI startup did not become ready. Inspect '+str(log))
            print('Faithful Video ready on HTTP 8188. Use the RunPod HTTPS proxy. Username: workbench.',flush=True)
            from aiohttp import web
            web.run_app(create_app(password,child),host='0.0.0.0',port=8188,access_log=None)
        finally:
            terminate(child,group=True)

if __name__=='__main__':main()
