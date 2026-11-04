"""Build-time CPU test of unmodified ComfyUI video I/O, not AI inference.
Custom nodes are disabled because the upstream SeedVR2 GPU selectors require CUDA.
"""
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url,data=None):
    b=None if data is None else json.dumps(data).encode()
    r=urllib.request.Request(url,data=b,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=10) as f:return json.load(f)


def main():
    with tempfile.TemporaryDirectory(prefix='seedvr2-media-test-') as d:
        r=Path(d)
        for n in ('input','output','temp','user'): (r/n).mkdir()
        subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=64x64:rate=24',
            '-f','lavfi','-i','sine=frequency=440:sample_rate=44100','-t','1',
            '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(r/'input/test.mp4')],check=True)
        command=['python','/opt/ComfyUI/main.py','--cpu','--disable-all-custom-nodes','--listen','127.0.0.1',
            '--port','8199','--input-directory',str(r/'input'),'--output-directory',str(r/'output'),
            '--temp-directory',str(r/'temp'),'--user-directory',str(r/'user')]
        with (r/'server.log').open('w+') as log:
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd='/opt/ComfyUI')
            try:
                url='http://127.0.0.1:8199'
                deadline=time.monotonic()+180
                while True:
                    if p.poll() is not None:raise RuntimeError('ComfyUI exited during CPU media check')
                    try: request(url+'/system_stats'); break
                    except (OSError,urllib.error.URLError):
                        if time.monotonic()>deadline:raise TimeoutError('ComfyUI startup timeout')
                        time.sleep(1)
                graph={
                    '1':{'class_type':'LoadVideo','inputs':{'file':'test.mp4'}},
                    '2':{'class_type':'GetVideoComponents','inputs':{'video':['1',0]}},
                    '3':{'class_type':'CreateVideo','inputs':{'images':['2',0],'fps':['2',2],'audio':['2',1]}},
                    '4':{'class_type':'SaveVideo','inputs':{'video':['3',0],'filename_prefix':'smoke',
                         'format':'auto','format.codec':'auto'}},
                }
                queued=request(url+'/prompt',{'prompt':graph,'client_id':'seedvr2-stock-cpu-check'})
                if queued.get('node_errors'):raise RuntimeError(str(queued['node_errors']))
                pid=queued['prompt_id']; deadline=time.monotonic()+90
                while True:
                    history=request(url+'/history/'+pid)
                    if pid in history:
                        entry=history[pid]
                        if entry.get('status',{}).get('status_str')!='success':raise RuntimeError(str(entry))
                        break
                    if time.monotonic()>deadline:raise TimeoutError('Video I/O execution timeout')
                    time.sleep(0.5)
                outputs=list((r/'output').glob('smoke*.mp4'))
                assert len(outputs)==1,outputs
                data=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
                    '-show_streams','-of','json',str(outputs[0])]))
                video=next(s for s in data['streams'] if s['codec_type']=='video')
                assert video['width']==video['height']==64
                assert int(video['nb_read_frames'])==24
                assert any(s['codec_type']=='audio' for s in data['streams'])
                print('PASS stock ComfyUI media I/O: 24 frames, 64x64, audio present; no AI inference')
            except Exception:
                log.flush();log.seek(0);print(log.read()[-16000:]);raise
            finally:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()

if __name__=='__main__':main()
