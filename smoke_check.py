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
            if slot['name'] not in defs:
                parent, _, child = slot['name'].partition('.')
                prefix, limit = {'ref_images':('ref_image_',9),'ref_videos':('ref_video_',3),'ref_video_audios':('ref_video_audio_',3),'ref_audios':('ref_audio_',3)}.get(parent,('',0))
                dynamic = n['type']=='MiniMaxH3ReferenceToVideo' and limit>0 and child.startswith(prefix) and child[len(prefix):].isdigit() and 0 <= int(child[len(prefix):]) < limit
                if not dynamic:raise ValueError(f'{n["type"]} input not registered: {slot["name"]}')
        for i,slot in enumerate(n['outputs']):
            if i>=len(info.get('output',[])):raise ValueError('Missing output on '+n['type'])
            typ=info['output'][i]
            if typ not in (slot['type'],'*'):raise ValueError(f'{n["type"]} output changed: {typ} != {slot["type"]}')
    # Wrappers call these actual pinned native methods. Check their registry and
    # callable interfaces in the separate CPU test command below, not by guessing.
    for name in ('UNETLoader','CLIPLoader','VAELoader','MiniMaxH3ReferenceToVideo','BasicScheduler','RandomNoise','KSamplerSelect','SamplerCustomAdvanced','BasicGuider'):
        if name not in registry:raise ValueError('Required native H3 support is missing: '+name)


def validate_director_widgets(registry, workflow):
    """Count the FRONTEND seed companion, not only Python inputs."""
    import math
    for node in workflow['nodes']:
        if node['type'] != 'H3PortraitDirector':continue
        info=registry[node['type']];widgets=[]
        for group in ('required','optional'):
            for name,spec in info.get('input',{}).get(group,{}).items():
                typ=spec[0];opts=spec[1] if len(spec)>1 else {}
                if isinstance(typ,list) or typ in ('STRING','INT','FLOAT','BOOLEAN'):
                    widgets.append((name,typ))
                    if opts.get('control_after_generate',name in ('seed','noise_seed')):
                        widgets.append(('control_after_generate',['fixed','increment','decrement','randomize']))
        values=node.get('widgets_values',[])
        if len(values)!=len(widgets):raise ValueError('Director serialized widget count does not match the real schema plus frontend seed control.')
        for (name,typ),value in zip(widgets,values):
            valid=(value in typ if isinstance(typ,list) else type(value) is int if typ=='INT' else type(value) is bool if typ=='BOOLEAN' else isinstance(value,str) if typ=='STRING' else type(value) in (int,float) and math.isfinite(value))
            if not valid:raise ValueError('Invalid serialized Director control: '+name)


def export_probe(session, port, home):
    """Real CPU server execution of native CreateVideo -> export -> final frame.

    Runs only during Docker build. A tiny generated fixture, no model downloads.
    """
    import uuid
    from PIL import Image
    tag='_h3_export_probe_'+uuid.uuid4().hex
    home=Path(home);input_dir=home/'input';input_dir.mkdir(parents=True,exist_ok=True)
    fixture=input_dir/(tag+'.png');output_dir=home/'output'/tag
    Image.new('RGB',(32,48),(60,100,180)).save(fixture)
    prompt={
        '1':{'class_type':'LoadImage','inputs':{'image':fixture.name}},
        '2':{'class_type':'CreateVideo','inputs':{'images':['1',0],'fps':24.0,'bit_depth':'auto','color_space':'sRGB','codec':'none'}},
        '3':{'class_type':'H3PortraitExportVideo','inputs':{'video':['2',0],'filename_prefix':tag+'/probe','crf':18}},
        '4':{'class_type':'H3PortraitSaveLastFrame','inputs':{'completed_export':['3',0]}},
    }
    try:
        submitted=session.post(f'http://127.0.0.1:{port}/prompt',json={'prompt':prompt,'client_id':tag},timeout=10)
        submitted.raise_for_status();reply=submitted.json()
        if 'prompt_id' not in reply:raise RuntimeError('CPU export probe rejected: '+str(reply)[:800])
        key=reply['prompt_id'];deadline=time.monotonic()+120
        while True:
            response=session.get(f'http://127.0.0.1:{port}/history/{key}',timeout=5);response.raise_for_status()
            record=response.json().get(key)
            if record:
                if record.get('status',{}).get('status_str')=='error':
                    raise RuntimeError('CPU export probe failed: '+str(record.get('status',{}))[-1600:])
                if record.get('status',{}).get('completed'):break
            if time.monotonic()>deadline:raise RuntimeError('CPU export probe did not finish.')
            time.sleep(.25)
        mp4s=list(output_dir.glob('*.mp4'));pngs=list(output_dir.glob('*_last.png'))
        if len(mp4s)!=1 or len(pngs)!=1 or pngs[0].stem!=mp4s[0].stem+'_last':
            raise RuntimeError('CPU export probe did not produce a matched MP4/PNG pair.')
        details=json.loads(pngs[0].with_suffix('.json').read_text())
        if details['decoded_frames']!=1 or details['last_frame_index']!=0:
            raise RuntimeError('CPU export probe final-frame index is incorrect.')
        with Image.open(pngs[0]) as image:
            if image.size!=(32,48):raise RuntimeError('CPU export probe changed frame dimensions.')
        print('H3 EXPORT CPU SMOKE PASS: native CreateVideo -> H.264 export -> exact final-frame PNG.',flush=True)
    finally:
        fixture.unlink(missing_ok=True)
        import shutil
        shutil.rmtree(output_dir,ignore_errors=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');ap.add_argument('--port',type=int,default=8188);ap.add_argument('--wait',type=int,default=240);a=ap.parse_args()
    import shutil
    for tool in ('ffmpeg','ffprobe'):
        if not shutil.which(tool):raise RuntimeError('Missing export tool: '+tool)
    graphs=[json.loads(p.read_text()) for p in (HERE/'workflows').glob('*.json')]
    for graph in graphs:validate_graph(graph)
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
        for graph in graphs:
            validate_registry(registry,graph)
            validate_director_widgets(registry,graph)
        if a.build:export_probe(session,port,'/opt/comfy-bundle')
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
