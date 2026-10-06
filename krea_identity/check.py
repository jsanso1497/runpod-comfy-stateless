#!/usr/bin/env python3
"""Static graph checks and optional real CPU ComfyUI node-schema smoke test.

Never downloads weights, queues GPU inference, or claims visual validation.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
WEIGHTS='1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0'


def validate_graph(g):
    ns={n['id']:n for n in g['nodes']};ls={l[0]:l for l in g['links']}
    if len(ns)!=len(g['nodes']) or len(ls)!=len(g['links']):raise ValueError('Duplicate graph IDs')
    for lid,a,sa,b,sb,typ in ls.values():
        source,dest=ns[a]['outputs'][sa],ns[b]['inputs'][sb]
        if source['type']!=typ or dest['type']!=typ or lid not in source['links'] or dest['link']!=lid:
            raise ValueError('Broken graph link: '+str(lid))
        if ns[a]['order']>=ns[b]['order']:raise ValueError('Cycle or non-topological node order')
    for n in ns.values():
        if n.get('mode',0)!=0:raise ValueError('Shipped nodes must be active; use the explicit rebalancer switch')
        if 'widgets_values_named' in n and list(n['widgets_values_named'].values())!=n['widgets_values']:
            raise ValueError('Widget serialization drift')
        for i,inp in enumerate(n.get('inputs',[])):
            lid=inp.get('link')
            if lid is not None and (lid not in ls or ls[lid][3:5]!=[n['id'],i]):raise ValueError('Dangling input')
        for i,out in enumerate(n.get('outputs',[])):
            for lid in out.get('links',[]):
                if lid not in ls or ls[lid][1:3]!=[n['id'],i]:raise ValueError('Dangling output')
    def one(typ):
        values=[n for n in ns.values() if n['type']==typ]
        if len(values)!=1:raise ValueError('Expected exactly one '+typ)
        return values[0]
    def origin(n,field):
        inp=next(i for i in n['inputs'] if i['name']==field)
        return tuple(ls[inp['link']][1:3]) if inp.get('link') else None
    recipe=g['extra']['krea_identity']['recipe']
    if recipe=='upscale':
        up=one('SeedVR2VideoUpscaler')['widgets_values_named']
        if up['batch_size']!=1 or up['temporal_overlap']!=0 or up['input_noise_scale']!=0 or up['latent_noise_scale']!=0:
            raise ValueError('Single-image upscale defaults drifted')
        if any('Video' in n['type'] and n['type']!='SeedVR2VideoUpscaler' for n in ns.values()):raise ValueError('Unexpected video conversion')
        return
    model=one('UNETLoader')['widgets_values_named']
    if model['unet_name']!='krea2_turbo_bf16.safetensors' or model['weight_dtype']!='default':raise ValueError('Quality baseline must be Turbo BF16')
    if one('CLIPLoader')['widgets_values_named']['type']!='krea2':raise ValueError('Wrong vision encoder type')
    sampler=one('KSampler');s=sampler['widgets_values_named']
    if [s[k] for k in ('steps','cfg','sampler_name','scheduler','denoise')]!=[12,1.0,'euler','simple',1.0]:raise ValueError('Wrong Turbo settings')
    if s['control_after_generate']!='fixed':raise ValueError('A/B comparison requires a fixed initial seed')
    if one('LoraLoaderModelOnly')['widgets_values_named']!={'lora_name':'krea2_identity_edit_v1_2.safetensors','strength_model':1.0}:raise ValueError('Use only the required full-rank editor')
    prep=one('KreaIdentityPrepare' if recipe=='three_reference' else 'KreaIdentityPair')
    encs=[n for n in ns.values() if n['type']=='Krea2EditGroundedEncode']
    if len(encs)!=2:raise ValueError('Both conditioning branches must be image-grounded')
    for e in encs:
        if origin(e,'image')!=(prep['id'],0) or origin(e,'image_b')!=(prep['id'],1):raise ValueError('Scene/subject order drifted')
    patch=one('Krea2EditModelPatch')
    for field,slot in [('source_image',0),('source_image_b',1)]:
        if origin(patch,field)!=(prep['id'],slot):raise ValueError('Semantic and appearance references differ')
    for field,slot in [('source_latent',0),('source_latent_b',1)]:
        node=ns[origin(patch,field)[0]]
        if node['type']!='VAEEncode' or origin(node,'pixels')!=(prep['id'],slot):raise ValueError('Incorrect appearance latents')
    if origin(patch,'target_latent')!=origin(sampler,'latent_image'):raise ValueError('Pre-encode canvas must match sampler')
    control=one('KreaIdentityRebalanceControl')
    if control['widgets_values_named']['enabled']:raise ValueError('Unbenchmarked rebalancing must not be forced on')
    rb=[n for n in ns.values() if n['type']=='ConditioningKrea2Rebalance']
    if len(rb)!=2 or any(n['widgets_values_named']!={'multiplier':1.0,'per_layer_weights':WEIGHTS} for n in rb):raise ValueError('Screenshot rebalancer defaults drifted')
    if origin(sampler,'positive')!=(control['id'],0) or origin(sampler,'negative')!=(control['id'],1):raise ValueError('Rebalance switch is not wired to the sampler')
    for raw,weighted in [('positive','rebalanced_positive'),('negative','rebalanced_negative')]:
        upstream=ns[origin(control,weighted)[0]]
        if upstream['type']!='ConditioningKrea2Rebalance' or origin(upstream,'conditioning')!=origin(control,raw):raise ValueError('A/B branches do not share conditioning')
    if origin(sampler,'model')!=(patch['id'],0):raise ValueError('Sampler bypasses identity model patch')
    if one('EmptySD3LatentImage')['widgets_values_named']['batch_size']!=1:raise ValueError('Expected single image')


def validate_schema(g,registry):
    files={'unet_name','clip_name','vae_name','lora_name','model','image'}
    for n in g['nodes']:
        if n['type']=='Note':continue
        if n['type'] not in registry:raise ValueError('Missing installed node: '+n['type'])
        schema=registry[n['type']];defs={}
        for group in ('required','optional'):defs.update(schema.get('input',{}).get(group,{}))
        named=n.get('widgets_values_named',{})
        linked={i['name'] for i in n['inputs'] if i.get('link') is not None}
        for name in schema.get('input',{}).get('required',{}):
            if name not in named and name not in linked:raise ValueError(n['type']+' missing required input '+name)
        for i in n['inputs']:
            if i['name'] not in defs:raise ValueError(n['type']+' unknown socket '+i['name'])
        for name,value in named.items():
            if name in ('control_after_generate','upload'):continue
            if name not in defs:raise ValueError(n['type']+' unknown widget '+name)
            typ=defs[name][0];opts=defs[name][1] if len(defs[name])>1 else {}
            # CPU build registry need not have GPU devices or downloaded weights.
            if name in files or name in ('device','offload_device'):continue
            if isinstance(typ,list) and value not in typ:raise ValueError(n['type']+' invalid choice '+name)
            if typ=='INT' and type(value)!=int:raise ValueError('Invalid integer '+name)
            if typ=='BOOLEAN' and type(value)!=bool:raise ValueError('Invalid boolean '+name)
            if typ=='FLOAT' and (type(value) not in (float,int) or not math.isfinite(value)):raise ValueError('Invalid float '+name)
            if typ in ('INT','FLOAT') and (value<opts.get('min',-math.inf) or value>opts.get('max',math.inf)):
                raise ValueError(n['type']+' numeric value out of range: '+name)
        outs=schema.get('output',[])
        for index,out in enumerate(n['outputs']):
            if index>=len(outs) or out['type']!=outs[index]:raise ValueError(n['type']+' output mismatch')


def static():
    graphs=[json.loads(p.read_text()) for p in sorted((HERE/'workflows').glob('*.json'))]
    if len(graphs)!=4:raise ValueError('Expected four versioned Krea workflows')
    for g in graphs:validate_graph(g)
    print('KREA IDENTITY STATIC PASS: four graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.',flush=True)
    return graphs


def main():
    p=argparse.ArgumentParser();p.add_argument('--build-smoke',action='store_true')
    p.add_argument('--comfy-home',type=Path,default=Path('/opt/comfy-bundle'))
    p.add_argument('--port',type=int,default=18190);p.add_argument('--timeout',type=int,default=240)
    a=p.parse_args();graphs=static()
    if not a.build_smoke:return
    import requests
    session=requests.Session();session.trust_env=False
    log=HERE/'krea-build-smoke.log';child=None
    try:
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'main.py','--cpu','--listen','127.0.0.1','--port',str(a.port),'--disable-auto-launch','--use-pytorch-cross-attention'],
                                   cwd=a.comfy_home,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+a.timeout
        while True:
            if child.poll() is not None:raise RuntimeError('ComfyUI exited during Krea smoke test')
            try:
                response=session.get(f'http://127.0.0.1:{a.port}/object_info',timeout=10)
                response.raise_for_status();registry=response.json();break
            except (requests.RequestException,ValueError):
                if time.monotonic()>=deadline:raise RuntimeError('Timed out waiting for real ComfyUI registry')
                time.sleep(1)
        for graph in graphs:validate_schema(graph,registry)
        print('KREA IDENTITY REAL CPU SCHEMA PASS. GPU inference and image quality NOT tested.',flush=True)
    except Exception:
        if log.exists():print(log.read_text(errors='replace')[-16000:],file=sys.stderr)
        raise
    finally:
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        session.close()
        log.unlink(missing_ok=True)

if __name__=='__main__':main()
