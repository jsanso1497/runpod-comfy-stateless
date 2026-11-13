"""Start actual ComfyUI on CPU, verify schemas and queue preflight routes over HTTP."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
P=Path(__file__).resolve().parent
URL='http://127.0.0.1:8197'

def request(path,data=None):
    body=None if data is None else json.dumps(data).encode()
    req=urllib.request.Request(URL+path,data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as response:return json.load(response)

def main():
    with tempfile.TemporaryDirectory(prefix='h3v-cpu-') as folder:
        root=Path(folder)
        for d in ('input','output','user','temp'):(root/d).mkdir()
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x192:rate=24','-t','1.5',
            '-c:v','libx264','-pix_fmt','yuv420p','-y',str(root/'input/smoke.mp4')],check=True)
        command=[sys.executable,'/opt/ComfyUI/main.py','--cpu','--listen','127.0.0.1','--port','8197',
            '--input-directory',str(root/'input'),'--output-directory',str(root/'output'),
            '--user-directory',str(root/'user'),'--temp-directory',str(root/'temp'),'--disable-auto-launch']
        with (P/'cpu-comfy.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd='/opt/ComfyUI')
            try:
                deadline=time.monotonic()+180
                while time.monotonic()<deadline:
                    if process.poll() is not None:raise RuntimeError('ComfyUI exited during CPU startup.')
                    try:schemas=request('/object_info');break
                    except Exception:time.sleep(1)
                else:raise TimeoutError('ComfyUI did not become ready on CPU.')
                required=['H3A40Settings','H3A40Video','H3A40Plan','H3A40Encode','H3A40Render','H3A40Audit','H3StudioOptionalImage',
                          'H3StudioTargetPhoto','FizgigH3StillLatent','FizgigH3StillDecode']
                missing=[name for name in required if name not in schemas]
                if missing:raise RuntimeError('Missing registered nodes: '+','.join(missing))
                from build_workflows import PROFILES,build
                for profile in PROFILES:
                    _,graph=build(profile)
                    for value in graph.values():
                        name=value['class_type']
                        if name not in schemas:raise RuntimeError('Unregistered class in workflow: '+name)
                        required=schemas[name]['input'].get('required',{})
                        missing=set(required)-set(value['inputs'])
                        if missing:raise RuntimeError('Missing API inputs for '+name+': '+str(missing))
                runs=[]
                for engine in ('MiniMax H3','Eros Beta 7 Fix1'):
                    _,prompt=build(next(p for p in PROFILES if p.get('preflight')))
                    prompt['10']['inputs']['engine']=engine
                    prompt['11']['inputs'].update(file='smoke.mp4',keyframe_count=3)
                    prompt['14']['inputs']['save_report']=False
                    submitted=request('/prompt',dict(prompt=prompt,client_id=str(uuid.uuid4())))
                    if submitted.get('node_errors'):raise RuntimeError(str(submitted))
                    pid=submitted['prompt_id'];deadline=time.monotonic()+180
                    while time.monotonic()<deadline:
                        history=request('/history/'+pid)
                        if pid in history:break
                        time.sleep(.25)
                    else:raise TimeoutError('Preflight timed out.')
                    item=history[pid]
                    if item['status']['status_str']!='success':raise RuntimeError(json.dumps(item['status']))
                    report=json.loads(item['outputs']['14']['text'][0])
                    representation=report['video']['representation']
                    expected='native video'
                    assert representation==expected
                    runs.append(dict(engine=engine,status='passed',representation=representation))
                result=dict(required_nodes=required,preflight_runs=runs,gpu_inference=False,model_weights_loaded=False)
                (P/'cpu-contract-report.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(result,indent=2))
            finally:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':main()
