#!/usr/bin/env python3
"""Check graph wiring and a real CPU ComfyUI node registry. No weights/inference."""
import argparse
import contextlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import requests

HERE=Path(__file__).resolve().parent
WORKFLOW=HERE/'workflows/H3_Portrait_Auto.json'


def validate_graph(workflow):
    ns={n['id']:n for n in workflow['nodes']};seen=set();parents={i:set() for i in ns}
    for i,a,sa,b,sb,t in workflow['links']:
        assert i not in seen;seen.add(i)
        assert a in ns and b in ns
        src=ns[a]['outputs'][sa];dest=ns[b]['inputs'][sb]
        assert i in src['links'] and dest['link']==i
        assert src['type'] in (t,'*') and dest['type'] in (t,'*')
        parents[b].add(a)
    for n in ns.values():
        for inp in n.get('inputs',[]):
            assert inp.get('link') is None or inp['link'] in seen
        for out in n.get('outputs',[]):assert set(out.get('links',[])).issubset(seen)
    done=set()
    while len(done)<len(ns):
        new={n for n,p in parents.items() if n not in done and p.issubset(done)}
        assert new,'Graph contains a cycle';done|=new
    assert not any(n['type'] in ('OmniNode','OmniNodeAPI') or 'RefMod' in n['type'] or 'Rebalance' in n['type'] for n in ns.values())
    return True


def validate_registry(registry,workflow):
    for n in workflow['nodes']:
        if n['type']=='Note':continue
        if n['type'] not in registry:raise ValueError('Missing installed node: '+n['type'])
        info=registry[n['type']];defs={}
        for group in ('required','optional'):defs.update(info.get('input',{}).get(group,{}))
        for slot in n['inputs']:
            if slot['name'] not in defs:raise ValueError(f'{n["type"]} input not registered: {slot["name"]}')
        for i,slot in enumerate(n['outputs']):
            if i>=len(info.get('output',[])):raise ValueError('Missing output on '+n['type'])
            typ=info['output'][i]
            if typ not in (slot['type'],'*'):raise ValueError(f'{n["type"]} output changed: {typ} != {slot["type"]}')
    # Wrappers call these actual pinned native methods. Check their registry and
    # callable interfaces in the separate CPU test command below, not by guessing.
    for name in ('UNETLoader','CLIPLoader','VAELoader','MiniMaxH3ReferenceToVideo','BasicScheduler','RandomNoise','KSamplerSelect','SamplerCustomAdvanced','BasicGuider'):
        if name not in registry:raise ValueError('Required native H3 support is missing: '+name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');ap.add_argument('--port',type=int,default=8188);ap.add_argument('--wait',type=int,default=240);a=ap.parse_args()
    graph=json.loads(WORKFLOW.read_text());validate_graph(graph)
    port=18188 if a.build else a.port;child=None
    session=requests.Session();session.trust_env=False
    log_path=HERE/'build-smoke.log'
    try:
        if a.build:
            log=log_path.open('w')
            child=subprocess.Popen([sys.executable,'main.py','--cpu','--listen','127.0.0.1','--port',str(port),'--disable-auto-launch','--use-pytorch-cross-attention'],cwd='/opt/comfy-bundle',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+a.wait
        while True:
            if child is not None and child.poll() is not None:raise RuntimeError('CPU ComfyUI exited during smoke test.')
            try:
                response=session.get(f'http://127.0.0.1:{port}/object_info',timeout=5)
                if response.status_code==200:
                    registry=response.json();break
            except requests.RequestException:pass
            if time.monotonic()>deadline:raise RuntimeError('ComfyUI did not become ready for node-schema validation.')
            time.sleep(1)
        validate_registry(registry,graph)
        print('H3 PORTRAIT SCHEMA PASS: actual ComfyUI registry and workflow connections. No GPU generation tested.',flush=True)
    except Exception:
        if a.build and log_path.exists():print(log_path.read_text(errors='replace')[-18000:],file=sys.stderr)
        raise
    finally:
        session.close()
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGKILL)
                child.wait()

if __name__=='__main__':main()
