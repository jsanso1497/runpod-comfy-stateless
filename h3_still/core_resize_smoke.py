"""Execute the installed native resize node through ComfyUI's real CPU queue.

No dummy model file is executed. Two differently shaped image inputs verify
that both reference paths work independently, without COLORCODE or RMBG.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import urllib.request


def request(url: str, body=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as response:return json.load(response)


def run(root: Path, url: str = 'http://127.0.0.1:8189') -> dict:
    from PIL import Image
    from workflow_tools import PRESETS
    Image.new('RGB',(160,96),(31,79,121)).save(root/'input'/'h3-native-test-a.png')
    Image.new('RGB',(96,144),(121,79,31)).save(root/'input'/'h3-native-test-b.png')
    # Deliberately use the production 2.5 MP settings, not a tiny stand-in size.
    prompt={
        '1':{'class_type':'LoadImage','inputs':{'image':'h3-native-test-a.png'}},
        '2':{'class_type':'ImageScaleToTotalPixels','inputs':{'image':['1',0],'upscale_method':'lanczos','megapixels':2.5,'resolution_steps':32}},
        '3':{'class_type':'SaveImage','inputs':{'images':['2',0],'filename_prefix':'native_resize_a'}},
        '4':{'class_type':'LoadImage','inputs':{'image':'h3-native-test-b.png'}},
        '5':{'class_type':'ImageScaleToTotalPixels','inputs':{'image':['4',0],'upscale_method':'lanczos','megapixels':2.5,'resolution_steps':32}},
        '6':{'class_type':'SaveImage','inputs':{'images':['5',0],'filename_prefix':'native_resize_b'}},
    }
    queued=request(url+'/prompt',{'prompt':prompt})
    if queued.get('node_errors'):raise RuntimeError(str(queued['node_errors']))
    prompt_id=queued['prompt_id'];record=None
    end=time.monotonic()+120
    while time.monotonic()<end:
        record=request(url+'/history/'+prompt_id).get(prompt_id)
        if record:break
        time.sleep(.25)
    if not record or record.get('status',{}).get('status_str')!='success':
        raise RuntimeError('Native resize queue failed: '+str(record))
    dimensions=[]
    for node_id in ('3','6'):
        rows=record.get('outputs',{}).get(node_id,{}).get('images',[])
        if not rows:raise RuntimeError('No output from native resize test.')
        row=rows[0];target=(root/'output'/row.get('subfolder','')/row['filename']).resolve()
        if not target.is_relative_to((root/'output').resolve()):raise ValueError('Bad output path.')
        with Image.open(target) as image:
            w,h=image.size
            assert w%32==h%32==0
            assert abs(w*h/(1024*1024)-2.5)<.12
            dimensions.append([w,h])
    assert dimensions[0][0]>dimensions[0][1]
    assert dimensions[1][0]<dimensions[1][1]
    return {'native_resize_execution':True,'reference_dimensions':dimensions,'color_node_required':False,
            'model_inference':False,'preset_count':len(PRESETS)}
