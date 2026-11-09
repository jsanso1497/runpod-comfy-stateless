"""Actual ComfyUI HTTP/schema and non-model reference/contact-sheet execution in the built image."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from graph_checks import validate_graph
P=Path(__file__).resolve().parent

def request(path,data=None):
    raw=None if data is None else json.dumps(data).encode()
    req=urllib.request.Request('http://127.0.0.1:8399'+path,data=raw,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)

def main():
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='h3lab-cpu-') as temp:
        root=Path(temp)
        for f in ('input','output','user','temp','models'):(root/f).mkdir()
        Image.new('RGB',(160,96),(50,80,120)).save(root/'input/fixture.png')
        logpath=root/'comfy.log';child=None
        env=dict(os.environ);env['H3_STILL_DATA_ROOT']=str(root)
        # This CPU process never invokes the production downloader/launcher.
        with logpath.open('w') as log:
            child=subprocess.Popen([sys.executable,'/opt/ComfyUI/main.py','--cpu','--listen','127.0.0.1','--port','8399',
                '--input-directory',str(root/'input'),'--output-directory',str(root/'output'),
                '--user-directory',str(root/'user'),'--temp-directory',str(root/'temp'),
                '--disable-auto-launch','--preview-method','none'],cwd='/opt/ComfyUI',env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                for _ in range(150):
                    if child.poll() is not None:raise RuntimeError('ComfyUI exited during Lab CPU start.')
                    try:info=request('/object_info');break
                    except OSError:time.sleep(2)
                else:raise RuntimeError('ComfyUI CPU startup timed out.')
                errors=[]
                for path in sorted((P/'workflows').glob('*.json')):
                    errors.extend(path.name+': '+e for e in validate_graph(json.loads(path.read_text()),info))
                if errors:raise RuntimeError('\n'.join(errors))
                if (P/'build-profile.txt').read_text().strip()=='ph3' and 'PH3ImageConditioning' not in info:
                    raise RuntimeError('P-H3 native node did not register.')
                graph={
                    '1':{'class_type':'H3StudioOptionalImage','inputs':{'image':'fixture.png','enabled':True}},
                    '2':{'class_type':'H3LabReference','inputs':{'reference':['1',0],'role':'face','cap_megapixels':.25,'crop_left_percent':0.,'crop_top_percent':0.,'crop_right_percent':0.,'crop_bottom_percent':0.}},
                    '3':{'class_type':'H3LabPlan','inputs':{'mode':'Replace person','aspect_ratio':'2:3','output_megapixels':.25,'preset':'Quality - 50 steps, no Turbo','manual_quality_steps':0,'reference_size_mode':'match','protect_background':False,'reference_1':['1',0],'reference_2':['2',0],'reference_3':['2',0]}},
                    '4':{'class_type':'H3LabPrompt','inputs':{'plan':['3',0],'use_role_builder':True,'target_description':'person on the left','face_slot':2,'body_slot':3,'preserve_other_people':True,'match_lighting':False,'request':'One natural photograph.'}},
                    '5':{'class_type':'H3LabReport','inputs':{'text':['4',0],'save_report':True}},
                    '6':{'class_type':'H3LabContactSheet','inputs':{'columns':2,'cell_size':128,'reference_1':['1',0],'reference_2':['2',0]}},
                    '7':{'class_type':'SaveImage','inputs':{'images':['6',0],'filename_prefix':'cpu_contact_sheet'}}}
                result=request('/prompt',{'prompt':graph,'client_id':'h3lab-cpu'});pid=result['prompt_id']
                for _ in range(60):
                    history=request('/history/'+pid)
                    if pid in history:break
                    time.sleep(.5)
                else:raise RuntimeError('Non-model CPU graph did not finish.')
                entry=history[pid]
                if entry.get('status',{}).get('status_str')!='success':raise RuntimeError(json.dumps(entry))
                pngs=list((root/'output').rglob('*.png'))
                if not pngs:raise RuntimeError('Contact sheet was not saved.')
                size=Image.open(pngs[0]).size
                if size!=(256,158):raise RuntimeError('Contact sheet size mismatch: '+str(size))
                report={'cpu_startup':'passed','graphs_checked':len(list((P/'workflows').glob('*.json'))),
                        'registered_lab_nodes':sorted(n for n in info if n.startswith('H3Lab')),
                        'native_reference_preprocessing':'passed','literal_prompt_output':'passed','contact_sheet_output':size,
                        'real_model_weights_loaded':False,'gpu_inference':False}
                (P/'cpu-lab-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
            except Exception:
                print(logpath.read_text(errors='replace')[-18000:],file=sys.stderr);raise
            finally:
                if child and child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
if __name__=='__main__':main()
