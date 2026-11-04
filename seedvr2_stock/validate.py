"""Offline graph checks; optional upstream source-contract checks. No inference."""
import argparse
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXPECTED={'LoadVideo','GetVideoComponents','SeedVR2LoadDiTModel','SeedVR2LoadVAEModel',
          'SeedVR2VideoUpscaler','CreateVideo','SaveVideo'}


def validate(path):
    g=json.loads(path.read_text())
    ns={n['id']:n for n in g['nodes']}
    assert len(ns)==len(g['nodes']), 'Duplicate node IDs'
    assert {n['type'] for n in ns.values() if n['type']!='Note'}==EXPECTED
    assert len([n for n in ns.values() if n['type']!='Note'])==7
    links={v[0]:v for v in g['links']}
    assert len(links)==len(g['links'])==8
    for ident,src,slot,dst,dslot,typ in links.values():
        o=ns[src]['outputs'][slot]; i=ns[dst]['inputs'][dslot]
        assert ident in o['links'] and i['link']==ident
        assert o['type']==i['type']==typ
    for n in ns.values():
        for slot,i in enumerate(n.get('inputs',[])):
            if i.get('link') is not None:
                assert links[i['link']][3:5]==[n['id'],slot]
        for slot,o in enumerate(n.get('outputs',[])):
            for link in o.get('links') or []:
                assert links[link][1:3]==[n['id'],slot]
    by_type={n['type']:n for n in ns.values()}
    assert by_type['SeedVR2LoadDiTModel']['widgets_values'][0]=='seedvr2_ema_7b_fp16.safetensors'
    assert by_type['SeedVR2LoadVAEModel']['widgets_values'][0]=='ema_vae_fp16.safetensors'
    w=by_type['SeedVR2VideoUpscaler']['widgets_values']
    assert len(w)==13 and w[4]>=5 and (w[4]-1)%4==0
    assert w[5:8]==[True,'lab',3]
    assert w[8:11]==[0,0.0,0.0]
    assert w[2:4] == ([1080,1920] if '1080p' in path.name else [2160,3840])
    assert links[6][1:5]==[2,1,6,1] and links[7][1:5]==[2,2,6,2]
    print('PASS graph:',path.name)
    return by_type


def upstream_check(root,by_type):
    seed=root/'custom_nodes/seedvr2_videoupscaler'
    files={'LoadVideo':root/'comfy_extras/nodes_video.py',
           'GetVideoComponents':root/'comfy_extras/nodes_video.py',
           'CreateVideo':root/'comfy_extras/nodes_video.py',
           'SaveVideo':root/'comfy_extras/nodes_video.py',
           'SeedVR2LoadDiTModel':seed/'src/interfaces/dit_model_loader.py',
           'SeedVR2LoadVAEModel':seed/'src/interfaces/vae_model_loader.py',
           'SeedVR2VideoUpscaler':seed/'src/interfaces/video_upscaler.py'}
    for typ,path in files.items():
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==typ)
        methods={n.name:n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        assert 'define_schema' in methods and 'execute' in methods, typ
        names={a.arg for a in methods['execute'].args.args}
        for i in by_type[typ]['inputs']:
            assert i['name'] in names,(typ,i['name'],'not in upstream execute signature')
        print('PASS upstream contract:',typ)
    registry=(seed/'src/utils/model_registry.py').read_text()
    assert 'seedvr2_ema_7b_fp16.safetensors' in registry
    assert 'ema_vae_fp16.safetensors' in registry
    print('PASS upstream model registry')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--upstream',type=Path)
    args=ap.parse_args()
    files=sorted((ROOT/'workflows').glob('*.json'))
    assert len(files)==2
    for p in files: by_type=validate(p)
    if args.upstream: upstream_check(args.upstream,by_type)
    print('No GPU inference or visual quality test is performed by these checks.')

if __name__=='__main__':main()
