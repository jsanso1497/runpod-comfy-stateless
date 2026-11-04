"""Build-time contracts, not a runtime node, renderer or export validator."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
NAME = '01_Video_Clarity'
ALLOWED = {'LoadVideo','VideoCrop','GetVideoComponents','GetImageSize','ComfyMathExpression',
           'SeedVR2LoadDiTModel','SeedVR2LoadVAEModel','SeedVR2VideoUpscaler',
           'CreateVideo','SaveVideo','ImageFromBatch','ImageStitch','SaveImage','Note'}


def validate(root=ROOT, upstream=None):
    graph=json.loads((root/'workflows'/f'{NAME}.json').read_text())
    api=json.loads((root/'api'/f'{NAME}.api.json').read_text())
    nodes={n['id']:n for n in graph['nodes']}
    assert len(nodes)==len(graph['nodes'])==15
    assert {n['type'] for n in nodes.values()} <= ALLOWED
    assert len(list((root/'workflows').glob('*.json')))==1, 'Only one factory workflow is shipped.'
    assert {str(n['id']) for n in nodes.values() if n['type']!='Note'} == set(api)
    links={l[0]:l for l in graph['links']}
    assert len(links)==len(graph['links'])
    for lid,src,slot,dst,in_slot,kind in graph['links']:
        assert nodes[src]['outputs'][slot]['type']==kind
        target=nodes[dst]['inputs'][in_slot]
        assert kind in target['type'].split(',')
        assert target['link']==lid and lid in nodes[src]['outputs'][slot]['links']
        assert api[str(dst)]['inputs'][target['name']]==[str(src),slot]
    for n in nodes.values():
        for inp in n['inputs']:
            assert inp['link'] is None or inp['link'] in links
        for output in n['outputs']:
            assert all(l in links for l in output['links'])
    def ancestors(k, stack=None):
        stack=set() if stack is None else set(stack)
        assert k not in stack,'Cycle in graph'
        stack.add(k)
        found=set()
        for v in api[k]['inputs'].values():
            if isinstance(v,list) and len(v)==2 and str(v[0]) in api:
                found.add(str(v[0]));found.update(ancestors(str(v[0]),stack))
        return found
    assert '8' in ancestors('5'), 'Crop must precede SeedVR2'
    assert api['5']['inputs']['resolution']==['10',1]
    assert api['10']['inputs']=={'expression':'min(a,b)','values.a':['9',0],'values.b':['9',1]}
    assert api['9']['inputs']['image']==['2',0]
    assert api['8']['inputs']['crop']=={'crop':{'x':0,'y':0,'width':0,'height':0}}
    settings=api['5']['inputs']
    assert settings['max_resolution']==0 and settings['batch_size']==9
    assert settings['uniform_batch_size'] and settings['temporal_overlap']==3
    assert settings['color_correction']=='lab' and settings['prepend_frames']==4
    assert settings['input_noise_scale']==settings['latent_noise_scale']==0
    assert settings['enable_debug'] is True
    assert api['3']['inputs']['model']=='seedvr2_ema_7b_fp16.safetensors'
    assert api['4']['inputs']['model']=='ema_vae_fp16.safetensors'
    assert api['6']['inputs']['fps']==['2',2] and api['6']['inputs']['audio']==['2',1]
    assert api['7']['inputs']['format']=='mp4' and api['7']['inputs']['format.codec']=='auto'
    assert api['11']['inputs']=={'image':['2',0],'batch_index':0,'length':1}
    assert api['12']['inputs']=={'image':['5',0],'batch_index':0,'length':1}
    assert api['13']['inputs']['direction']=='right' and not api['13']['inputs']['match_image_size']
    assert api['13']['inputs']['image1']==['11',0] and api['13']['inputs']['image2']==['12',0]
    assert api['14']['inputs']['images']==['13',0]
    # Ensure API and UI defaults have not drifted.
    assert nodes[5]['widgets_values']==[42,'fixed',1080,0,9,True,'lab',3,4,0.0,0.0,'cpu',True]
    assert nodes[7]['widgets_values']==['SeedVR2/Restored','mp4']
    for path in root.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    sources=json.loads((root/'sources.json').read_text())
    docker=(root/'Dockerfile').read_text()
    assert sources['comfy_commit'] in docker and sources['seedvr_commit'] in docker
    assert 'custom_nodes/seedvr2_videoupscaler' in docker
    assert not (root/'nodes').exists(), 'No custom processing nodes may be added.'
    if upstream is not None:
        specs={
            'SeedVR2LoadDiTModel':('custom_nodes/seedvr2_videoupscaler/src/interfaces/dit_model_loader.py','SeedVR2LoadDiTModel'),
            'SeedVR2LoadVAEModel':('custom_nodes/seedvr2_videoupscaler/src/interfaces/vae_model_loader.py','SeedVR2LoadVAEModel'),
            'SeedVR2VideoUpscaler':('custom_nodes/seedvr2_videoupscaler/src/interfaces/video_upscaler.py','SeedVR2VideoUpscaler'),
            'VideoCrop':('comfy_extras/nodes_video.py','VideoCrop'),
            'GetImageSize':('comfy_extras/nodes_images.py','GetImageSize'),
            'ImageFromBatch':('comfy_extras/nodes_images.py','ImageFromBatch'),
            'ImageStitch':('comfy_extras/nodes_images.py','ImageStitch'),
        }
        for typ,(path,classname) in specs.items():
            tree=ast.parse((upstream/path).read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
            method=next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='execute')
            params={a.arg for a in method.args.args+method.args.kwonlyargs}-{'cls','self'}
            for row in api.values():
                if row['class_type']==typ:
                    assert set(row['inputs'])<=params,(typ,set(row['inputs'])-params)
        # Math inputs are intentionally flattened by native Autogrow serialization.
        assert 'ComfyMathExpression' in (upstream/'comfy_extras/nodes_math.py').read_text()
    return {'one_workflow':True,'crop_before_model':True,'native_size':True,'comparison':True,'gpu_tested':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--upstream',type=Path)
    print(json.dumps(validate(upstream=parser.parse_args().upstream),indent=2))
