"""Loopback-only vision requests, bounded repair and confirmed model unloading."""
from __future__ import annotations
import base64
import contextlib
import io
import json
import time
import requests
from .logic import output_schema, parse_llm, ClarificationNeeded

BASE='http://127.0.0.1:11434'


def session():
    s=requests.Session();s.trust_env=False;return s


def api(s,path,payload=None,timeout=30):
    if path not in ('/api/version','/api/tags','/api/show','/api/chat','/api/generate','/api/ps'):
        raise ValueError('Unsupported local endpoint.')
    try:
        method=s.get if payload is None else s.post
        kwargs={} if payload is None else {'json':payload}
        with method(BASE+path,timeout=(5,timeout),allow_redirects=False,**kwargs) as r:
            if r.status_code != 200:
                raise RuntimeError(f'Ollama {path}: HTTP {r.status_code}. Read /workspace/h3-portrait/ollama.log.')
            return r.json()
    except requests.RequestException as e:
        raise RuntimeError('Ollama is not ready. Wait for H3 PORTRAIT OLLAMA READY in the Pod log.') from e


def canonical(model):
    return model if ':' in model.rsplit('/',1)[-1] else model+':latest'


def model_info(s,model):
    match=next((x for x in api(s,'/api/tags').get('models',[]) if canonical(x.get('name',''))==canonical(model)),None)
    if not match:
        raise RuntimeError('Selected Ollama model is not downloaded. Wait for H3 PORTRAIT OLLAMA READY.')
    show=api(s,'/api/show',{'model':model})
    if 'vision' not in show.get('capabilities',[]):
        raise RuntimeError('The selected Ollama model must support vision.')
    if show.get('remote_host') or show.get('remote_model'):
        raise RuntimeError('This template uses local Ollama only.')
    return {'model':model,'digest':match.get('digest',model),'capabilities':show.get('capabilities',[])}


def preview(image,max_edge):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    x=image[0,...,:3].detach().to(device='cpu',dtype=torch.float32)
    h,w=x.shape[:2]; scale=min(1.,max_edge/max(h,w))
    if scale<1:
        x=F.interpolate(x.permute(2,0,1)[None],size=(max(1,round(h*scale)),max(1,round(w*scale))),mode='bilinear',align_corners=False,antialias=True)[0].permute(1,2,0)
    buf=io.BytesIO();Image.fromarray(np.rint(x.clamp(0,1).numpy()*255).astype('uint8')).save(buf,format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def unload(s,model,timeout=90):
    api(s,'/api/generate',{'model':model,'stream':False,'keep_alive':0},timeout=min(timeout,60))
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        rows=api(s,'/api/ps',timeout=10).get('models',[])
        if not rows:
            return
        time.sleep(.25)
    raise RuntimeError('Ollama did not unload fully. H3 was not started; retry after stopping any other Ollama request.')


def generate(s,cfg,info,system,direction,refs,recipe,triggers,variation,interrupt=lambda:None):
    images=[preview(r['image'],cfg['image_max_edge']) for r in refs]
    ledger='\n'.join(f'Image {i}: {r["filename"]}' for i,r in enumerate(refs,1))
    content=(f'ATTACHMENTS IN ORDER:\n{ledger}\n\n'
             f'OUTPUT: {recipe["aspect"]} portrait, {recipe["length"]} frames at 24 fps ({recipe["actual_seconds"]:.3f} seconds).\n'
             f'USER BRIEF (no special syntax):\n{direction}\n\n'
             f'LoRA trigger words to retain when relevant: {triggers or "None"}\n'
             f'RETURN JSON SCHEMA:\n{json.dumps(output_schema(len(refs)))}')
    messages=[{'role':'system','content':system},{'role':'user','content':content,'images':images}]
    failure=None
    try:
        for attempt in range(2):
            body={'model':info['model'],'messages':messages,'format':output_schema(len(refs)),
                  'stream':True,'keep_alive':0,
                  'options':{'num_ctx':cfg['context_length'],'num_predict':cfg['max_output_tokens'],
                             'temperature':0.15,'seed':int(variation)}}
            if 'thinking' in info['capabilities']:
                body['think']=False
            interrupt(); pieces=[];last=None;size=0
            with s.post(BASE+'/api/chat',json=body,stream=True,allow_redirects=False,timeout=(5,cfg['request_timeout_seconds'])) as response:
                if response.status_code!=200:
                    raise RuntimeError(f'Ollama returned HTTP {response.status_code}; no video was started.')
                for line in response.iter_lines(chunk_size=1):
                    interrupt()
                    if not line:continue
                    packet=json.loads(line)
                    if packet.get('error'):raise RuntimeError('Ollama generation failed; inspect the Pod Ollama log.')
                    piece=packet.get('message',{}).get('content','');size+=len(piece)
                    if size>100000:raise ValueError('Ollama exceeded the response-size limit.')
                    pieces.append(piece)
                    if packet.get('done'):last=packet;break
            result=''.join(pieces)
            try:
                if last is None or last.get('done_reason')=='length':
                    raise ValueError('Incomplete JSON. Return shorter descriptions, without losing any reference entries.')
                return parse_llm(result,len(refs))
            except ClarificationNeeded:
                raise
            except (ValueError,TypeError,KeyError) as e:
                if attempt:raise RuntimeError('Ollama could not form its internal reference map after one repair attempt. Your notes need no special format; simplify the brief or change prompt_variation.') from e
                messages.extend([{'role':'assistant','content':result[:50000]},
                                 {'role':'user','content':f'Fix ONLY the internal JSON issue: {e}. Return the entire corrected object. Preserve the user brief and all attached image assignments.'}])
                print('H3 PORTRAIT: repairing internal Ollama JSON once.',flush=True)
    except BaseException as e:
        failure=e;raise
    finally:
        try:unload(s,info['model'],cfg['unload_timeout_seconds'])
        except Exception:
            if failure is None:raise
            print('H3 PORTRAIT: cleanup failed after an interrupted/failed draft; no video handoff occurred.',flush=True)
