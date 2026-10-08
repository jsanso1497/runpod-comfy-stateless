#!/usr/bin/env python3
"""Validate graphs, references, model boundaries, UI widgets, optional runtime schemas."""
from __future__ import annotations
import argparse,json,re,sys,urllib.request
from pathlib import Path
from build_workflows import S
ROOT=Path(__file__).resolve().parents[1]


def linked(value,api):
    return isinstance(value,list) and len(value)==2 and isinstance(value[0],str) and value[0] in api and isinstance(value[1],int)


def require(condition,message):
    if not condition:raise ValueError(message)


def check_graph(ui,api):
    nodes={n['id']:n for n in ui['nodes']};links={x[0]:x for x in ui['links']}
    require(len(nodes)==len(ui['nodes']),'Duplicate node IDs')
    require(len(links)==len(ui['links']),'Duplicate link IDs')
    require(set(api)=={str(n['id']) for n in nodes.values() if n['type']!='Note'},'UI/API node mismatch')
    for lid,src,slot,dst,dslot,kind in links.values():
        require(src in nodes and dst in nodes,'Dangling link')
        a,b=nodes[src],nodes[dst]
        require(0<=slot<len(a['outputs']) and 0<=dslot<len(b['inputs']),'Invalid link slot')
        require(lid in a['outputs'][slot]['links'],'Missing source link')
        require(b['inputs'][dslot]['link']==lid,'Destination link mismatch')
        left,right=a['outputs'][slot]['type'],b['inputs'][dslot]['type']
        require(left==right or '*' in (left,right),'Link type mismatch')
        require(api[str(dst)]['inputs'][b['inputs'][dslot]['name']]==[str(src),slot],'API connection mismatch')
    for n in nodes.values():
        for p in n['inputs']:
            if p['link'] is not None:require(p['link'] in links,'Unknown input link')
        for p in n['outputs']:
            for lid in p['links']:require(lid in links,'Unknown output link')
        typ=n['type'];require(typ in S,f'Unspecified node schema {typ}')
        ins,outs,widgets=S[typ]
        require(len(n['widgets_values'])==len(widgets),f'Widget count mismatch: {typ}')
        if typ=='Note':continue
        vals=api[str(n['id'])]['inputs'];require(api[str(n['id'])]['class_type']==typ,'API type mismatch')
        for name,value in zip(widgets,n['widgets_values']):
            if name not in ('upload','control_after_generate'):
                require(vals.get(name)==value,f'UI/API widget drift {typ}.{name}')
        if typ=='KSampler':
            require(vals['cfg']==vals['denoise']==1.0,'Unsupported CFG/denoise for empty edit latent')
            require(vals['sampler_name']=='euler' and vals['scheduler']=='simple','Sampler mismatch')
            require(vals['steps'] in (40,50),'Unreviewed step count')
            latent=vals['latent_image'];require(api[latent[0]]['class_type']=='Q21NEncodeReferences' and latent[1]==2,'Must sample native encoder matching latent')
        if typ=='Q21NEncodeReferences':
            slots=sorted(int(k.split('_')[-1]) for k in vals if k.startswith('image_'))
            require(slots==list(range(1,max(slots)+1)),'Reference slot gap')
            mentioned={int(x) for x in re.findall(r'<image(\d+)>',vals['prompt'])}
            require(mentioned<=set(slots),'Prompt references a missing image slot')
        if typ=='CLIPLoader':require(vals['type']=='qwen_image','Wrong text encoder architecture')
        if typ=='QwenImage21Cache':require(vals['dtype']=='default','Lossy cache unexpectedly enabled')
        if typ=='LoraLoaderModelOnly':require(api[vals['model'][0]]['class_type']=='UNETLoader','LoRA must branch from base, not stack')
    seen=set();visiting=set()
    def visit(key):
        require(key not in visiting,'Cycle in graph')
        if key in seen:return
        visiting.add(key)
        for value in api[key]['inputs'].values():
            if linked(value,api):visit(value[0])
        visiting.remove(key);seen.add(key)
    for key in api:visit(key)


def check_schema(ui,api,info,check_files=False):
    file_choices={'image','unet_name','clip_name','vae_name','lora_name'}
    for key,node in api.items():
        typ=node['class_type'];require(typ in info,f'Missing runtime node: {typ}')
        spec=info[typ];required=spec.get('input',{}).get('required',{});optional=spec.get('input',{}).get('optional',{})
        inputs={**required,**optional}
        for name in required:require(name in node['inputs'],f'{typ}: required input {name} missing')
        for name,val in node['inputs'].items():
            require(name in inputs,f'{typ}: unknown input {name}')
            want=inputs[name][0]
            if linked(val,api):
                source=api[val[0]]['class_type'];output=info[source]['output']
                require(0<=val[1]<len(output),f'{source}: invalid output slot')
                actual=output[val[1]]
                require(want==actual or '*' in (want,actual),f'{typ}.{name}: expected {want}, got {actual}')
            elif isinstance(want,list) or want=='COMBO':
                # Legacy dropdowns are represented by the list itself.
                # ComfyUI V3 io.Combo.Input serializes to ["COMBO", {"options": [...]}].
                # Both are widgets, not wiring sockets. Do not mistake a V3
                # dropdown for a missing control, and validate its choices.
                opts=inputs[name][1] if len(inputs[name])>1 and isinstance(inputs[name][1],dict) else {}
                choices=want if isinstance(want,list) else opts.get('options')
                require(isinstance(choices,list) and bool(choices),f'{typ}.{name}: missing COMBO options')
                is_file=name in file_choices or (name=='model' and typ.startswith('SeedVR2'))
                if check_files or not is_file:
                    require(val in choices,f'{typ}.{name}: {val!r} not in runtime options')
            elif want in ('INT','FLOAT','STRING','BOOLEAN'):
                valid={'INT':isinstance(val,int) and not isinstance(val,bool),
                    'FLOAT':isinstance(val,(int,float)) and not isinstance(val,bool),'STRING':isinstance(val,str),'BOOLEAN':isinstance(val,bool)}
                require(valid[want],f'{typ}.{name}: invalid {want} value')
                opts=inputs[name][1] if len(inputs[name])>1 else {}
                if want in ('INT','FLOAT'):
                    require('min' not in opts or val>=opts['min'],f'{typ}.{name}: below minimum')
                    require('max' not in opts or val<=opts['max'],f'{typ}.{name}: above maximum')
        widgets=[]
        for name,desc in inputs.items():
            t=desc[0];opts=desc[1] if len(desc)>1 and isinstance(desc[1],dict) else {}
            if not opts.get('forceInput') and (isinstance(t,list) or t in ('COMBO','INT','FLOAT','STRING','BOOLEAN')):widgets.append(name)
        expected=[x for x in S[typ][2] if x not in ('upload','control_after_generate')]
        require(widgets==expected,f'{typ}: UI widget order differs: runtime {widgets}, saved {expected}')


def selected_catalog(profile='identity',include_bfs=False):
    rows=json.loads((ROOT/'config/workflow_catalog.json').read_text())['workflows']
    return [r for r in rows if (r['requires']!='upscale' or profile=='upscale') and (r['requires']!='bfs_optional' or include_bfs)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=['identity','upscale'],default='identity')
    p.add_argument('--include-bfs',action='store_true');p.add_argument('--server');p.add_argument('--schema-file',type=Path)
    p.add_argument('--check-files',action='store_true');args=p.parse_args();info=None
    if args.schema_file:info=json.loads(args.schema_file.read_text())
    if args.server:
        with urllib.request.urlopen(args.server.rstrip('/')+'/object_info',timeout=120) as response:info=json.load(response)
    rows=selected_catalog(args.profile,args.include_bfs)
    require(bool(rows),'No workflows selected')
    for row in rows:
        path=ROOT/'workflows'/row['file'];ui=json.loads(path.read_text());api=json.loads((ROOT/'api_workflows'/(path.stem+'.api.json')).read_text())
        check_graph(ui,api)
        if row['requires']!='bfs_optional':
            require(not any(n['class_type']=='LoraLoaderModelOnly' for n in api.values()),'LoRA in native workflow')
        if info is not None:check_schema(ui,api,info,args.check_files)
        print('PASS '+row['file'])
    print(f'{len(rows)} workflow pairs validated'+(' against live/source node schemas.' if info else '.'))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as exc:print('VALIDATION FAILED: '+str(exc),file=sys.stderr);sys.exit(1)
