"""Image-build gate: real Comfy job API, output metadata and restart recovery.

No H3 inference, model download or GPU. This validates the API contract used by
Open workflow, not a literal browser click. Run only on the assembled image.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from .disk_common import ComfyClient
from .graphs import Graph


def canvas():
    g=Graph('H3 right-click workflow CPU smoke')
    a=g.add('EmptyImage','CPU test image',0,0,outputs=[('IMAGE','IMAGE')],widgets={
        'width':32,'height':32,'batch_size':1,'color':0})
    b=g.add('H3SwapWindowPreview','H3 metadata path under test',400,0,inputs=[('images','IMAGE')],
        outputs=[('unchanged_frames','IMAGE')],widgets={'iteration_index':0,'hold_seconds':0,'hold_first_window_only':True})
    c=g.add('SaveImage','CPU output',800,0,inputs=[('images','IMAGE')],outputs=[('images','IMAGE')],
        widgets={'filename_prefix':'rightclick_smoke/result'})
    g.connect(a,0,b,'images');g.connect(b,0,c,'images')
    return g.result()


def main():
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='h3-user-path-smoke-') as d:
        root=Path(d)
        for x in ('input','output','user'): (root/x).mkdir()
        env=dict(os.environ,H3_SWAP_DATA=str(root),H3_SWAP_SAVE_METADATA='1')
        client=ComfyClient('http://127.0.0.1:8299',poll_seconds=.1)
        cmd=[sys.executable,'/opt/ComfyUI/main.py','--cpu','--listen','127.0.0.1','--port','8299',
             '--user-directory',str(root/'user'),'--input-directory',str(root/'input'),
             '--output-directory',str(root/'output'),'--disable-auto-launch']
        def run():
            log=(root/'comfy.log').open('a')
            proc=subprocess.Popen(cmd,cwd='/opt/ComfyUI',env=env,stdout=log,stderr=subprocess.STDOUT)
            for _ in range(180):
                if proc.poll() is not None:break
                try:
                    if 'H3SwapWindowPreview' in client.request('/object_info',timeout=2):return proc,log
                except Exception:pass
                time.sleep(1)
            proc.terminate();log.close()
            raise RuntimeError('CPU startup failed:\n'+(root/'comfy.log').read_text()[-12000:])
        def stop(proc,log):
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        graph,api=canvas();ident=None
        proc,log=run()
        try:
            ident=client.request('/prompt',{'prompt':api,'extra_data':{'extra_pnginfo':{'workflow':graph}}})['prompt_id']
            result=client.wait(ident,timeout=60)
            detail=client.request('/api/jobs/'+ident)
            assert detail['workflow']['extra_data']['extra_pnginfo']['workflow']==graph
            record=root/'h3-swap/job-history'/(ident+'.json')
            for _ in range(40):
                if record.exists():break
                time.sleep(.1)
            assert record.exists(),'History not persisted'
            images=list((root/'output/rightclick_smoke').glob('*.png'));assert images
            assert json.loads(Image.open(images[0]).info['workflow'])==graph
        finally:stop(proc,log)
        proc,log=run()
        try:
            detail=client.request('/api/jobs/'+ident)
            assert detail['workflow']['extra_data']['extra_pnginfo']['workflow']==graph
            assert not client.request('/queue')['queue_running']
            print('RIGHT-CLICK CONTRACT PASS: queued workflow, actual image metadata, restored /api/jobs record, no regeneration')
        finally:stop(proc,log)

if __name__=='__main__':main()
