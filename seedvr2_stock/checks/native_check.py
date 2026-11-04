"""CI-only native ComfyUI media check THROUGH Nginx. AI deliberately bypassed.

No upstream code is patched and no model files are loaded. The CPU server disables
custom nodes because official SeedVR2 device selectors require a GPU. Static
contracts separately check the actual upstream SeedVR2 execute signatures.
"""
from __future__ import annotations
import asyncio, base64, copy, hashlib, json, os, subprocess, time, uuid
from fractions import Fraction
from pathlib import Path
import aiohttp
from PIL import Image
import numpy as np
ROOT=Path('/opt/seedvr2-stock')
DATA=Path('/workspace/seedvr2-stock')
BASE='http://127.0.0.1:8188'

def run(args):
    return subprocess.check_output(args,stderr=subprocess.STDOUT,text=True,timeout=120)

def probe(path):
    return json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(path)]))

def vstream(info):
    return next(x for x in info['streams'] if x['codec_type']=='video')

def times(path):
    rows=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))['frames']
    v=[float(x['best_effort_timestamp_time']) for x in rows]
    return [x-v[0] for x in v]

def fixture(path,rate='30',audio=True):
    cmd=['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i',f'testsrc2=size=192x108:rate={rate}:duration=1']
    if audio:cmd+=['-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=1']
    cmd+=['-c:v','libx264','-pix_fmt','yuv420p','-threads','2','-video_track_timescale',str(Fraction(rate).numerator)]
    if audio:cmd+=['-c:a','aac','-b:a','192k']
    cmd+=[str(path)];run(cmd)

def cpu_prompt(full,filename,crop,prefix):
    api=copy.deepcopy(full)
    for k in ('3','4','5'):api.pop(k)
    api['1']['inputs']['file']=filename
    api['8']['inputs']['crop']={'crop':dict(zip(('x','y','width','height'),crop))}
    api['6']['inputs']['images']=['2',0]
    api['12']['inputs']['image']=['2',0]
    api['7']['inputs']['filename_prefix']='SeedVR2/'+prefix
    api['14']['inputs']['filename_prefix']='SeedVR2/'+prefix+'_compare'
    # Force the actual native min expression to execute even though AI is bypassed.
    api['90']={'class_type':'EmptyImage','inputs':{'width':['10',1],'height':2,'batch_size':1,'color':0}}
    api['91']={'class_type':'SaveImage','inputs':{'images':['90',0],'filename_prefix':'SeedVR2/'+prefix+'_size'}}
    return api

async def main():
    password=os.environ['COMFY_PASSWORD']
    header={'Authorization':'Basic '+base64.b64encode(('workbench:'+password).encode()).decode()}
    report={'gpu_tested':False,'ai_inference_tested':False,'real_native_cpu_io':True,'cases':[]}
    full=json.loads((ROOT/'api/01_Video_Clarity.api.json').read_text())
    fixtures=DATA/'fixtures';fixtures.mkdir(exist_ok=True,parents=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
        for _ in range(180):
            try:
                async with s.get(BASE+'/object_info',headers=header) as r:
                    r.raise_for_status();schema=await r.json()
                break
            except (aiohttp.ClientError,OSError):await asyncio.sleep(1)
        else:raise RuntimeError('Native ComfyUI/Nginx startup failed')
        async with s.get(BASE+'/object_info') as r:assert r.status==401
        assert {'VideoCrop','GetVideoComponents','ComfyMathExpression','ImageStitch','SaveVideo'}<=schema.keys()
        assert 'FaithfulVideoRestore' not in schema
        source=fixtures/'normal.mp4';fixture(source)
        async def upload(path):
            form=aiohttp.FormData()
            with path.open('rb') as f:
                form.add_field('image',f,filename=path.name,content_type='video/mp4')
                form.add_field('overwrite','true')
                async with s.post(BASE+'/upload/image',data=form,headers=header) as r:
                    r.raise_for_status();return (await r.json())['name']
        uploaded=await upload(source)
        cases=[('full',(0,0,0,0)),('square_left',(0,0,108,108)),('square_center',(42,0,108,108)),
          ('square_right',(84,0,108,108)),('portrait_left',(0,0,72,108)),
          ('portrait_center',(60,0,72,108)),('portrait_right',(120,0,72,108)),('custom',(80,20,64,64))]
        client_id='clarity-ci-'+uuid.uuid4().hex;events=[]
        async with s.ws_connect(BASE+'/ws?clientId='+client_id,headers=header,heartbeat=5) as ws:
            async def reader():
                async for msg in ws:
                    if msg.type==aiohttp.WSMsgType.TEXT:events.append(json.loads(msg.data))
            task=asyncio.create_task(reader())
            try:
                async with s.patch(BASE+'/internal/logs/subscribe',headers=header,json={'clientId':client_id,'enabled':True}) as r:
                    assert r.status==200,await r.text()
                reference=None
                for name,crop,src,file_name,audio in [(n,c,source,uploaded,True) for n,c in cases]+[]:
                    result=await check_case(s,header,full,client_id,name,crop,src,file_name,audio,reference)
                    if name=='full':reference=np.array(Image.open(result['comparison_path']))[:,:192,:3]
                    report['cases'].append(result)
                for name,rate,audio in [('fractional','30000/1001',True),('odd_fps','3128000/104247',True),('silent','25',False)]:
                    src=fixtures/(name+'.mp4');fixture(src,rate,audio)
                    result=await check_case(s,header,full,client_id,name,(0,0,0,0),src,await upload(src),audio,None)
                    report['cases'].append(result)
                await ws.ping(b'still-connected');await asyncio.sleep(10)
                assert not ws.closed and not task.done(),'WebSocket did not stay open'
                assert any(x.get('type')=='executing' for x in events),'No live execution events'
                async with s.get(BASE+'/internal/logs/raw',headers=header) as r:
                    r.raise_for_status();assert 'entries' in await r.json()
                report.update(websocket_stable=True,execution_events=True,logs_available=True)
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True)
    (DATA/'native-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: native crop/size/audio/FPS/compare/export/download and Nginx transport. NO GPU inference.')

async def check_case(s,header,full,client_id,name,crop,src,filename,audio,reference):
    api=cpu_prompt(full,filename,crop,'TEST_'+name)
    async with s.post(BASE+'/prompt',headers=header,json={'prompt':api,'client_id':client_id}) as r:
        body=await r.json();assert r.status==200,body
    pid=body['prompt_id'];deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        async with s.get(BASE+'/history/'+pid,headers=header) as r:hist=await r.json()
        if pid in hist:break
        await asyncio.sleep(.25)
    else:raise RuntimeError('Native CPU prompt timed out')
    assert hist[pid]['status']['status_str']=='success',hist[pid]
    output=max((DATA/'output/SeedVR2').glob('TEST_'+name+'_*.mp4'),key=lambda x:x.stat().st_mtime_ns)
    comparison=max((DATA/'output/SeedVR2').glob('TEST_'+name+'_compare_*.png'),key=lambda x:x.stat().st_mtime_ns)
    x,y,w,h=crop;w=w or 192;h=h or 108
    before=probe(src);after=probe(output);v=vstream(after);a=vstream(before)
    assert (v['width'],v['height'],int(v['nb_read_frames']))==(w,h,int(a['nb_read_frames']))
    old,new=times(src),times(output);assert len(old)==len(new)
    assert max(abs(u-v) for u,v in zip(old,new))<.002
    assert v['codec_name']=='h264'
    tracks=[t for t in after['streams'] if t['codec_type']=='audio']
    assert len(tracks)==int(audio)
    if audio:assert int(tracks[0].get('nb_read_frames',0))>0
    size_png=max((DATA/'output/SeedVR2').glob('TEST_'+name+'_size_*.png'),key=lambda x:x.stat().st_mtime_ns)
    assert Image.open(size_png).size==(min(w,h),2),'Native automatic short-edge expression is incorrect'
    im=np.array(Image.open(comparison))[:,:,:3]
    assert im.shape[:2]==(h,2*w)
    assert np.array_equal(im[:,:w],im[:,w:]),'CPU control should compare the same image'
    if reference is not None:assert np.max(np.abs(im[:,:w].astype(int)-reference[y:y+h,x:x+w].astype(int)))<=2,'Wrong crop position'
    query={'filename':output.name,'subfolder':'SeedVR2','type':'output'}
    async with s.get(BASE+'/view',params=query,headers=header) as r:
        r.raise_for_status();data=await r.read()
    assert hashlib.sha256(data).digest()==hashlib.sha256(output.read_bytes()).digest()
    async with s.get(BASE+'/view',params=query,headers=dict(header,Range='bytes=0-1023')) as r:
        assert r.status==206 and await r.read()==data[:1024]
    result=dict(name=name,crop=crop,size=[w,h],frames=int(v['nb_read_frames']),audio=audio,download=query,
      comparison_path=str(comparison),uploaded=filename,source_fps=a['avg_frame_rate'],output_fps=v['avg_frame_rate'])
    print('PASS native CPU case:',name,flush=True);return result

if __name__=='__main__':asyncio.run(main())
