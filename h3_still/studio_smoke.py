"""Run real optional-image and routing nodes on ComfyUI CPU before publication.

No diffusion/encoder/VAE weight file is loaded by this test. Nine references
here tests routing and input handling, not nine-reference GPU capacity.
"""
import json
from pathlib import Path
import time
from core_resize_smoke import request


def run(root, info, url='http://127.0.0.1:8189'):
    from PIL import Image
    from studio_validate import validate_studio
    validate_studio(info)
    for name,size in [('studio-a.png',(160,96)),('studio-b.png',(96,160))]:
        Image.new('RGB',size,(50,90,130)).save(root/'input'/name)
    report=[]
    cases=[('none','Auto',0,'Generate'),('edit_one','Auto',1,'Edit'),
           ('references_two','Auto',2,'References'),('references_nine','Auto',9,'References'),
           ('references_one_explicit','References',1,'References'),
           ('disabled_missing_file','Auto',0,'Generate')]
    for name,mode,count,expected in cases:
        prompt={}
        for slot in range(1,10):
            filename=('studio-a.png' if slot%2 else 'studio-b.png') if slot<=count else '(none)'
            enabled=True
            if name=='disabled_missing_file' and slot==1:
                filename='missing-deleted-photo.png';enabled=False
            prompt[str(slot)]={'class_type':'H3StudioOptionalImage','inputs':{'image':filename,'enabled':enabled}}
        settings={'mode':mode,'aspect_ratio':'1:1','generation_megapixels':4.,'edit_megapixels':2.5,
                  'preset':'Quality - 50 steps, no Turbo','reference_cap_megapixels':2.5}
        settings.update({'reference_%s'%i:[str(i),0] for i in range(1,10)})
        prompt['10']={'class_type':'H3StudioPlan','inputs':settings}
        prompt['11']={'class_type':'H3StudioAudit','inputs':{'plan':['10',0]}}
        queued=request(url+'/prompt',{'prompt':prompt})
        if queued.get('node_errors'):raise RuntimeError(str(queued['node_errors']))
        id=queued['prompt_id'];record=None;end=time.monotonic()+120
        while time.monotonic()<end:
            record=request(url+'/history/'+id).get(id)
            if record:break
            time.sleep(.2)
        if not record or record.get('status',{}).get('status_str')!='success':
            raise RuntimeError('Studio CPU routing failed for '+name+': '+str(record))
        value=json.loads(record['outputs']['11']['text'][0])
        assert value['mode']==expected and value['count']==count
        assert value['model'].endswith('_pruned_int8_convrot.safetensors')
        assert ('ref2va' in value['model'])==(expected=='References')
        assert value['width']%32==value['height']%32==0
        assert len(value['reference_mapping'])==count
        report.append({'case':name,'passed':True,'mode':expected,'references':count,
                       'model':value['model'],'dimensions':[value['width'],value['height']]})
    return {'real_comfy_cpu_queue':True,'cases':report,'weights_loaded':False,'gpu_tested':False}
