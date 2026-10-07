#!/usr/bin/env python3
"""Static graph validation; optionally verify every node against a running ComfyUI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def check_graph(ui: dict, api: dict) -> None:
    nodes={n['id']:n for n in ui['nodes']}
    assert len(nodes)==len(ui['nodes']), 'Duplicate node IDs'
    links={x[0]:x for x in ui['links']}
    assert len(links)==len(ui['links']), 'Duplicate links'
    for lid,src,slot,dst,dslot,kind in links.values():
        a=nodes[src];b=nodes[dst]
        assert lid in a['outputs'][slot]['links']
        assert b['inputs'][dslot]['link']==lid
        left=a['outputs'][slot]['type'];right=b['inputs'][dslot]['type']
        assert left==right or '*' in (left,right), (left,right)
        assert api[str(dst)]['inputs'][b['inputs'][dslot]['name']]==[str(src),slot]
    for n in nodes.values():
        for p in n['inputs']:
            if p['link'] is not None:assert p['link'] in links
        for p in n['outputs']:
            for lid in p['links']:assert lid in links
        if n['type']=='Note':continue
        assert str(n['id']) in api
        a=api[str(n['id'])]
        assert a['class_type']==n['type']
        if n['type']=='KSampler':
            vals=a['inputs']
            assert vals['cfg']==1.0 and vals['denoise']==1.0
            assert vals['sampler_name']=='euler' and vals['scheduler']=='simple'
            assert vals['steps']==40
            latent=vals['latent_image']
            assert api[latent[0]]['class_type']=='Q21EncodeReferences' and latent[1]==2
        if n['type']=='CLIPLoader':assert a['inputs']['type']=='qwen_image'
        if n['type']=='LoraLoaderModelOnly':
            assert api[a['inputs']['model'][0]]['class_type']=='UNETLoader', 'Adapters must branch from base, not stack'
        if n['type']=='QwenImage21Cache':assert a['inputs']['dtype']=='default'
    visited=set();stack=set()
    def visit(key):
        if key in stack:raise AssertionError('Graph cycle')
        if key in visited:return
        stack.add(key)
        for value in api[key]['inputs'].values():
            if isinstance(value,list) and len(value)==2 and isinstance(value[0],str) and value[0] in api:
                visit(value[0])
        stack.remove(key);visited.add(key)
    for key in api:visit(key)


def check_schema(ui: dict, api: dict, info: dict) -> None:
    skip_choices={'image','unet_name','clip_name','vae_name','lora_name','model','device','offload_device'}
    for key,node in api.items():
        cls=node['class_type']
        if cls not in info:raise AssertionError(f'Missing runtime node: {cls}')
        spec=info[cls]
        required=spec.get('input',{}).get('required',{})
        optional=spec.get('input',{}).get('optional',{})
        inputs={**required,**optional}
        for name in required:
            if name not in node['inputs']:raise AssertionError(f'{cls}: required input {name} missing')
        for name,value in node['inputs'].items():
            if name not in inputs:raise AssertionError(f'{cls}: unknown input {name}')
            typ=inputs[name][0]
            if isinstance(value,list) and len(value)==2 and value[0] in api:
                upstream=api[value[0]]['class_type']
                outputs=info[upstream]['output']
                assert value[1]<len(outputs), (upstream,value[1],outputs)
                actual=outputs[value[1]]
                assert actual==typ or '*' in (actual,typ), (cls,name,typ,actual)
            elif isinstance(typ,list) and name not in skip_choices:
                assert value in typ,(cls,name,value,typ)
        # A saved UI widget position must match the live node schema's widget order.
        widget_names=[]
        for name,desc in inputs.items():
            t=desc[0];opts=desc[1] if len(desc)>1 and isinstance(desc[1],dict) else {}
            if not opts.get('forceInput') and (isinstance(t,list) or t in ('INT','FLOAT','BOOLEAN','STRING')):
                widget_names.append(name)
        from build_workflows import S
        expected=[x for x in S[cls][2] if x not in ('upload','control_after_generate')]
        assert widget_names==expected,(cls,'UI widget order differs',widget_names,expected)
    print('Runtime node schemas, connections, and saved widget order verified.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['identity','upscale'],default='identity')
    p.add_argument('--server',help='Example: http://127.0.0.1:8188')
    args=p.parse_args()
    info=None
    if args.server:
        with urllib.request.urlopen(args.server.rstrip('/')+'/object_info',timeout=90) as response:
            info=json.load(response)
    paths=sorted((ROOT/'workflows').glob('*.json'))
    count=0
    for path in paths:
        if path.name.startswith('06_') and args.profile!='upscale':continue
        ui=json.loads(path.read_text());api=json.loads((ROOT/'api_workflows'/(path.stem+'.api.json')).read_text())
        check_graph(ui,api)
        if info is not None:check_schema(ui,api,info)
        print('PASS '+path.name);count+=1
    print(f'{count} workflow pairs validated.')
if __name__=='__main__':main()
