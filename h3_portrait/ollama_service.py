#!/usr/bin/env python3
"""Supervise the one private loopback Ollama daemon used by this template."""
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

# Import only our own small package; ComfyUI/CUDA imports in it are lazy.
from node import ollama_client as oc
from node.logic import settings

ROOT=Path('/workspace/h3-portrait')

def main():
    ROOT.mkdir(parents=True,exist_ok=True);cfg=settings();child=None
    env={k:v for k,v in os.environ.items() if not k.endswith(('_TOKEN','_KEY','_PASSWORD','_SECRET')) and k.upper() not in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','OLLAMA_ORIGINS')}
    env.update(OLLAMA_HOST='127.0.0.1:11434',OLLAMA_MODELS=str(ROOT/'ollama-models'),OLLAMA_NO_CLOUD='1',OLLAMA_KEEP_ALIVE='0',OLLAMA_NUM_PARALLEL='1',OLLAMA_MAX_LOADED_MODELS='1')
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        with oc.session() as s, (ROOT/'ollama.log').open('a',buffering=1) as log:
            try:oc.api(s,'/api/version',timeout=2)
            except Exception:pass
            else:raise RuntimeError('Port 11434 is already occupied; this template must own its Ollama daemon.')
            child=subprocess.Popen(['/usr/bin/ollama','serve'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+90
            while True:
                if child.poll() is not None:raise RuntimeError('Ollama exited; inspect ollama.log.')
                try:oc.api(s,'/api/version',timeout=3);break
                except Exception:
                    if time.monotonic()>=deadline:raise RuntimeError('Ollama did not start.')
                    time.sleep(1)
            for model in dict.fromkeys((cfg['analysis_model'], cfg['ollama_model'])):
                print('H3 PORTRAIT: downloading prompt model '+model,flush=True)
                subprocess.run(['/usr/bin/ollama','pull',model],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=cfg['pull_timeout_seconds'],check=True)
            info=oc.pipeline_info(s,cfg)
            (ROOT/'ollama-status.json').write_text(json.dumps(info,indent=2))
            print('H3 PORTRAIT OLLAMA READY: analysis='+cfg['analysis_model']+' | director='+cfg['ollama_model']+' | separate models; sequential loading',flush=True)
            child.wait();raise RuntimeError('Ollama stopped.')
    except KeyboardInterrupt:pass
    finally:
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGKILL)
                child.wait()

if __name__=='__main__':main()
