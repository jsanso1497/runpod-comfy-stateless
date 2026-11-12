#!/usr/bin/env python3
"""Optional ON-POD generation check with a synthetic video, not an identity benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid
from build_workflows import build,PROFILES

URL='http://127.0.0.1:8189'

def request(path,data=None):
    body=None if data is None else json.dumps(data).encode()
    with urllib.request.urlopen(urllib.request.Request(URL+path,data=body,headers={'Content-Type':'application/json'}),timeout=60) as r:return json.load(r)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--engine',choices=['all','minimax','eros','ph3'],default='all')
    parser.add_argument('--precision',choices=['INT8 HQ','BF16 diffusion','BF16 full'],default='INT8 HQ')
    parser.add_argument('--megapixels',type=float,default=.25)
    args=parser.parse_args()
    root=Path(os.environ.get('H3_STILL_DATA_ROOT','/workspace/h3-still'))
    name='h3v_synthetic_smoke_'+uuid.uuid4().hex[:8]+'.mp4'
    video=root/'input'/name
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x192:rate=24','-t','1.5',
        '-c:v','libx264','-pix_fmt','yuv420p','-y',str(video)],check=True)
    available={'minimax':'MiniMax H3','eros':'Eros Beta 7 Fix1','ph3':'P-H3 Image'}
    selected=available.values() if args.engine=='all' else [available[args.engine]]
    results=[]
    try:
        for engine in selected:
            _,prompt=build(PROFILES[0])
            prompt['10']['inputs'].update(engine=engine,precision=args.precision,output_megapixels=args.megapixels)
            prompt['11']['inputs'].update(file=name,keyframe_count=2,duration_seconds=1.5)
            prompt['12']['inputs']['prompt']='subject_definitions:\n<Video 1> supplies colors and geometric shapes.\nCreate one still image of a simple abstract geometric sculpture.'
            prompt['16']['inputs']['filename_prefix']='H3_Video_Studio/Smoke/'+engine.replace(' ','_')
            response=request('/prompt',dict(prompt=prompt,client_id=str(uuid.uuid4())))
            if response.get('node_errors'):raise RuntimeError(json.dumps(response))
            pid=response['prompt_id'];print('Queued',engine,pid,flush=True)
            deadline=time.monotonic()+3600
            while time.monotonic()<deadline:
                history=request('/history/'+pid)
                if pid in history:break
                time.sleep(2)
            else:raise TimeoutError('Generation exceeded smoke-test polling limit; inspect ComfyUI. The script did not cancel the queued job.')
            record=history[pid];status=record['status']
            if status['status_str']!='success':raise RuntimeError(json.dumps(status))
            report=json.loads(record['outputs']['18']['text'][0]);results.append(report)
            print('Passed:',engine,'size:',report['output_actual_size'],'peak bytes:',report.get('gpu',{}).get('peak_allocated_bytes'),flush=True)
    finally:
        folder=root/'output/H3_Video_Studio';folder.mkdir(parents=True,exist_ok=True)
        (folder/'gpu-smoke-summary.json').write_text(json.dumps(dict(completed=results,precision=args.precision,
            note='Synthetic low-resolution pipeline check only, not a likeness benchmark or maximum-VRAM qualification.'),indent=2)+'\n')
        # Keep the tiny fixture so stored workflow metadata remains reproducible.
    print('Report:',folder/'gpu-smoke-summary.json')
if __name__=='__main__':main()
