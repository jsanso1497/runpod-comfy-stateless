"""Image-build gate: real pinned imports, ComfyUI startup and zero-AI media execution.
No weights are fetched or loaded. This does NOT test GPU restoration or its quality.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
from fv_common import ROOT,terminate
from worker import inference_args


def json_request(url,body=None):
    data=json.dumps(body).encode() if body is not None else None
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=10) as response:return json.load(response)


def main():
    locks=json.loads((ROOT/'sources.lock.json').read_text())
    for location,key in [('/opt/ComfyUI','comfy_commit'),('/opt/seedvr2','seedvr_commit')]:
        actual=subprocess.check_output(['git','-C',location,'rev-parse','HEAD'],text=True).strip()
        assert actual==locks[key],(location,actual)
    cli=Path('/opt/seedvr2/inference_cli.py')
    data=cli.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert blob==locks['seedvr_cli_git_blob'],'Upstream CLI source contract changed.'
    tree=ast.parse(data)
    core=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_process_frames_core')
    fields={x.attr for x in ast.walk(core) if isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=='args'}
    args=inference_args({'batch':9,'model_dir':'/unused'},{'output_size':[1920,1080]})
    assert fields <= vars(args).keys(), fields-vars(args).keys()
    # Real import checks the dependencies used by the GPU worker, without inference.
    sys.path.insert(0,'/opt/seedvr2')
    spec=importlib.util.spec_from_file_location('faithful_seedvr_import_check',cli)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    assert callable(module._process_frames_core)
    assert module.DEFAULT_VAE=='ema_vae_fp16.safetensors'
    with tempfile.TemporaryDirectory(prefix='faithful-smoke-') as temp:
        root=Path(temp)
        for name in ('input','output','user','temp','jobs'):(root/name).mkdir()
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=s=96x64:r=10:d=1',
                        '-f','lavfi','-i','sine=f=440:r=48000:d=1','-c:v','ffv1','-c:a','pcm_s16le',
                        str(root/'input/smoke.mkv')],check=True)
        env=os.environ.copy()
        env.update(FAITHFUL_DATA_ROOT=str(root),HF_HUB_OFFLINE='1')
        command=[sys.executable,'/opt/ComfyUI/main.py','--cpu','--listen','127.0.0.1','--port','8899',
                 '--input-directory',str(root/'input'),'--output-directory',str(root/'output'),
                 '--user-directory',str(root/'user'),'--temp-directory',str(root/'temp'),
                 '--cache-none','--disable-smart-memory','--disable-auto-launch']
        with (root/'comfy.log').open('wb') as log:
            child=subprocess.Popen(command,cwd='/opt/ComfyUI',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                for _ in range(240):
                    if child.poll() is not None:raise RuntimeError('ComfyUI exited before startup.')
                    try:
                        schema=json_request('http://127.0.0.1:8899/object_info')
                        break
                    except OSError:time.sleep(1)
                else:raise RuntimeError('ComfyUI startup timeout.')
                assert {'LoadVideo','FaithfulVideoRestore'}<=schema.keys()
                api=json.loads((ROOT/'api/01_Faithful_Video.api.json').read_text())
                api['1']['inputs']['file']='smoke.mkv'
                api['2']['inputs'].update(restoration_strength=0.0,duration_seconds=0.0)
                response=json_request('http://127.0.0.1:8899/prompt',{'prompt':api,'client_id':'faithful-cpu-smoke'})
                assert not response.get('node_errors'),response
                pid=response['prompt_id']
                for _ in range(180):
                    history=json_request('http://127.0.0.1:8899/history/'+pid)
                    if pid in history:break
                    time.sleep(1)
                else:raise RuntimeError('CPU graph did not complete.')
                state=history[pid]
                assert state['status']['status_str']=='success',state
                outputs=list((root/'output').rglob('enhanced.mp4'))
                assert len(outputs)==1
                from media import probe,video
                v=video(probe(outputs[0],True))
                assert (v['width'],v['height'],int(v['nb_read_frames']))==(96,64,10)
                print('PASS: actual pinned ComfyUI startup, node schemas, file-backed VIDEO graph and CPU encode.')
                print('NOT TESTED: CUDA inference, GPU memory, temporal perceptual quality or restored footage.')
            except BaseException:
                print((root/'comfy.log').read_text(errors='replace')[-20000:])
                raise
            finally:terminate(child,group=True)

if __name__=='__main__':main()
