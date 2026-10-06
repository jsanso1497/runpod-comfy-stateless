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
    if recipe in ('directed_reference','couple_single_pass','couple_protected'):
        validate_directed_graph(g, ns, origin)
        return
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
    prep=one('KreaIdentityTextSceneTwoRefs' if recipe=='described_scene_two_references' else 'KreaIdentityPrepare' if recipe=='three_reference' else 'KreaIdentityPair')
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



def validate_directed_graph(g, ns, origin):
    recipe = g['extra']['krea_identity']['recipe']
    stages = g['extra']['krea_identity'].get('stages', [])
    count = 2 if recipe == 'couple_protected' else 1
    if len(stages) != count:
        raise ValueError('Incorrect number of directed edit stages.')
    def all_nodes(kind):
        return [n for n in ns.values() if n['type'] == kind]
    def single(kind):
        found = all_nodes(kind)
        if len(found) != 1:
            raise ValueError('Expected one shared ' + kind)
        return found[0]
    model = single('UNETLoader')['widgets_values_named']
    if model != {'unet_name':'krea2_turbo_bf16.safetensors','weight_dtype':'default'}:
        raise ValueError('Directed workflows must retain Turbo BF16.')
    if single('CLIPLoader')['widgets_values_named']['type'] != 'krea2':
        raise ValueError('Directed workflows need Krea image-grounded encoding.')
    lora = single('LoraLoaderModelOnly')
    if lora['widgets_values_named'] != {'lora_name':'krea2_identity_edit_v1_2.safetensors','strength_model':1.0}:
        raise ValueError('Only the full existing Identity Edit LoRA is allowed.')
    if len(all_nodes('KSampler')) != count or len(all_nodes('Krea2EditModelPatch')) != count:
        raise ValueError('Unexpected sampler/patch count.')
    if len(all_nodes('Krea2EditGroundedEncode')) != count*2 or len(all_nodes('ConditioningKrea2Rebalance')) != count*2:
        raise ValueError('Each pass needs two matching conditioning paths.')
    previous_preps = set()
    for stage in stages:
        types = {'prepare':'KreaIdentityCoupleEdit' if recipe == 'couple_single_pass' else 'KreaIdentityDirectedEdit',
                 'patch':'Krea2EditModelPatch','sampler':'KSampler','positive':'Krea2EditGroundedEncode',
                 'negative':'Krea2EditGroundedEncode','control':'KreaIdentityRebalanceControl',
                 'empty':'EmptySD3LatentImage','decode':'VAEDecode'}
        if set(stage) != set(types) or len(set(stage.values())) != len(stage):
            raise ValueError('Invalid stage metadata.')
        for key,kind in types.items():
            if stage[key] not in ns or ns[stage[key]]['type'] != kind:
                raise ValueError('Stage metadata does not match real nodes.')
        prep,patch,sample,control,empty=(ns[stage[k]] for k in ('prepare','patch','sampler','control','empty'))
        if prep['id'] in previous_preps:
            raise ValueError('Person references must not share a prepare node.')
        previous_preps.add(prep['id'])
        for key in ('positive','negative'):
            n = ns[stage[key]]
            if origin(n,'image') != (prep['id'],0) or origin(n,'image_b') != (prep['id'],1):
                raise ValueError('Directed semantic references are reversed or cross-wired.')
            if n['widgets_values_named']['grounding_px'] != 1024:
                raise ValueError('The shipped people-grounding baseline is 1024.')
        if origin(ns[stage['positive']],'prompt') != (prep['id'],2):
            raise ValueError('The labeled replacement instruction is not connected.')
        if ns[stage['negative']]['widgets_values_named']['prompt'] != '':
            raise ValueError('Negative must retain the trained empty instruction.')
        for field,slot in [('source_image',0),('source_image_b',1)]:
            if origin(patch,field) != (prep['id'],slot):
                raise ValueError('Directed appearance/semantic images do not match.')
        for field,slot in [('source_latent',0),('source_latent_b',1)]:
            upstream = ns[origin(patch,field)[0]]
            if upstream['type'] != 'VAEEncode' or origin(upstream,'pixels') != (prep['id'],slot):
                raise ValueError('Directed VAE references do not match the selected person.')
        if origin(patch,'model') != (lora['id'],0):
            raise ValueError('Passes must start with the same UNPATCHED editor, not another person\'s patch.')
        if patch['widgets_values_named'] != {'ref_boost':4.0,'ref_boost_a':1.0,'fit_mode':'fit'}:
            raise ValueError('Scene/subject/FIT defaults changed.')
        if origin(patch,'target_latent') != (empty['id'],0) or origin(sample,'latent_image') != (empty['id'],0):
            raise ValueError('Target latent must be pre-encoded at the actual sample size.')
        if origin(empty,'width') != (prep['id'],3) or origin(empty,'height') != (prep['id'],4):
            raise ValueError('Aspect ratio must follow the prepared image.')
        if empty['widgets_values_named']['batch_size'] != 1:
            raise ValueError('Only single images are sampled.')
        settings=sample['widgets_values_named']
        if [settings[k] for k in ('steps','cfg','sampler_name','scheduler','denoise','control_after_generate')] != [12,1.0,'euler','simple',1.0,'fixed']:
            raise ValueError('Quality baseline or reproducible seed mode changed.')
        if origin(sample,'model') != (patch['id'],0):
            raise ValueError('Directed sampler bypasses the identity editor.')
        if origin(sample,'positive') != (control['id'],0) or origin(sample,'negative') != (control['id'],1):
            raise ValueError('Rebalancer switch must control this exact sampler.')
        if control['widgets_values_named']['enabled']:
            raise ValueError('Rebalancer starts off until compared with a fixed seed.')
        for raw,weighted in [('positive','rebalanced_positive'),('negative','rebalanced_negative')]:
            if origin(control,raw) != (stage[raw],0):
                raise ValueError('Another person\'s conditioning was wired into this pass.')
            rb = ns[origin(control,weighted)[0]]
            if rb['type'] != 'ConditioningKrea2Rebalance' or origin(rb,'conditioning') != (stage[raw],0):
                raise ValueError('Rebalanced branch does not share the raw conditioning.')
            if rb['widgets_values_named'] != {'multiplier':1.0,'per_layer_weights':WEIGHTS}:
                raise ValueError('Screenshot weights changed.')
        if origin(ns[stage['decode']],'samples') != (sample['id'],0):
            raise ValueError('Decode is not connected to this pass.')
    if recipe == 'couple_protected':
        a,b = (ns[s['prepare']] for s in stages)
        ra,rb = (ns[origin(p,'scene')[0]] for p in (a,b))
        if ra['type'] != 'KreaIdentityRegion' or rb['type'] != 'KreaIdentityRegion':
            raise ValueError('Protected editing requires two selected regions.')
        if origin(rb,'protect_mask') != (ra['id'],2):
            raise ValueError('Second edit must protect all first-edit pixels.')
        composites = all_nodes('KreaIdentityRegionComposite')
        if len(composites) != 2:
            raise ValueError('Both generations must be composited back.')
        ca = next(n for n in composites if origin(n,'region_geometry') == (ra['id'],1))
        cb = next(n for n in composites if origin(n,'region_geometry') == (rb['id'],1))
        if origin(ca,'original') != origin(ra,'image') or origin(rb,'image') != (ca['id'],0) or origin(cb,'original') != (ca['id'],0):
            raise ValueError('Protected passes are not sequential or lost the original scene.')
        for comp,p,stage in [(ca,a,stages[0]),(cb,b,stages[1])]:
            if origin(comp,'canvas_geometry') != (p['id'],5) or origin(comp,'generated') != (stage['decode'],0):
                raise ValueError('Region composite has the wrong generation or geometry.')
        if origin(a,'references') == origin(b,'references'):
            raise ValueError('Man and woman must have separate reference banks.')
    else:
        trim = single('KreaIdentityUnframe')
        if origin(trim,'canvas_geometry') != (stages[0]['prepare'],5) or origin(trim,'image') != (stages[0]['decode'],0):
            raise ValueError('Scene margins must be removed using the matching canvas geometry.')
    for n in ns.values():
        if any(x in n['type'] for x in ('Ollama','SeedVR','H3','Video')):
            raise ValueError('Directed image generation must stay Krea-only.')


def validate_schema(g,registry):
    files={'unet_name','clip_name','vae_name','lora_name','model','image','file'}
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
    if len(graphs)!=8:raise ValueError('Expected eight versioned Krea workflows')
    for g in graphs:validate_graph(g)
    print('KREA IDENTITY STATIC PASS: eight graphs, reference order, screenshot node, fixed seeds, native-save/optional-upscale separation.',flush=True)
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
        # The Docker recipe bundles the additive media module before this gate.
        # Reuse this REAL registry, not a mocked list of available nodes.
        media_check=HERE.parent/'h3-media/check.py'
        if not media_check.exists():media_check=HERE.parent/'h3_media/check.py'
        if media_check.exists():
            import importlib.util
            spec=importlib.util.spec_from_file_location('_h3_media_build_check',media_check)
            media=importlib.util.module_from_spec(spec);spec.loader.exec_module(media)
            for graph in media.static():validate_schema(graph,registry)
            media.validate_native_registry(registry)
            print('H3 MEDIA REAL CPU SCHEMA PASS. GPU/video generation NOT tested.',flush=True)
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
