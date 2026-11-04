from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streaming import windows,OverlapWriter,source_mix
from media import make_plan,prepare,detect_cuts,finish,probe,video
from worker import inference_args,render
from fv_common import atomic_json,child_env,terminate


def request(**kw):
    result=dict(start=0.0,duration=0.0,scale=1,strength=0.5,batch=9,keep_frames=False,
                model_dir='/models',seedvr_root='/opt/seedvr2')
    result.update(kw)
    return result


def info(w=1920,h=1080,count=90,fps='30/1'):
    return {'streams':[dict(codec_type='video',width=w,height=h,avg_frame_rate=fps,
                          nb_read_frames=str(count),pix_fmt='yuv420p',sample_aspect_ratio='1:1')],
            'format':{}}


@pytest.mark.parametrize('count', [1,2,4,5,9,16,17,18,32,33,34,65,99,100,1000])
@pytest.mark.parametrize('size', [17,33,38])
def test_stream_no_lost_or_duplicate_frames(count,size):
    saved=[]
    writer=OverlapWriter(lambda index,x:saved.append((index,x.copy())))
    for win in windows(count,[2,9,35,100],size,5):
        frames=np.arange(win.start,win.stop,dtype=np.float32).reshape((-1,1,1,1))
        writer.accept(win,frames)
    writer.finish(count)
    assert [a for a,b in saved]==list(range(count))
    assert np.allclose([b.item() for a,b in saved],np.arange(count))


def test_boundary_predictions_are_actually_blended():
    saved=[]
    writer=OverlapWriter(lambda i,x:saved.append(float(x.item())))
    for i,win in enumerate(windows(50,[],33,5)):
        writer.accept(win,np.full((win.stop-win.start,1,1,1),i,dtype=np.float32))
    writer.finish(50)
    assert saved[27]==0 and saved[28]==0 and 0 < saved[30] < 1 and saved[32]==1


def test_no_blend_across_cut():
    saved=[]
    writer=OverlapWriter(lambda i,x:saved.append(float(x.item())))
    for win in windows(60,[30],33,5):
        writer.accept(win,np.full((win.stop-win.start,1,1,1),win.shot,dtype=np.float32))
    writer.finish(60)
    assert saved[:30]==[0]*30 and saved[30:]==[1]*30


def test_blend_endpoints():
    original=np.ones((4,4,3),np.float32)*0.2
    generated=np.ones_like(original)*0.8
    assert np.array_equal(source_mix(original,generated,0),original)
    assert np.allclose(source_mix(original,generated,0.5),0.5)
    assert np.allclose(source_mix(original,generated,1),generated)
    with pytest.raises(ValueError):source_mix(original,generated,1.5)


@pytest.mark.parametrize('w,h',[(1920,1080),(1584,896),(1440,1088),(1080,1920),(1366,768)])
@pytest.mark.parametrize('scale',[1,2])
def test_exact_integer_scaling(w,h,scale):
    p=make_plan(Path('x'),request(scale=scale),info(w,h))
    assert p['output_size']==[w*scale,h*scale]
    args=inference_args(request(),p)
    assert args.resolution==min(w,h)*scale and args.max_resolution==max(w,h)*scale


def test_quality_contract():
    a=inference_args(request(),make_plan(Path('x'),request(),info()))
    assert a.dit_model=='seedvr2_ema_7b_fp16.safetensors'
    assert a.color_correction=='lab' and a.batch_size==9 and a.uniform_batch_size
    assert a.input_noise_scale==a.latent_noise_scale==0
    assert a.vae_decode_tile_size==512 and a.vae_decode_tile_overlap==128
    assert a.cache_dit and a.cache_vae and not a.compile_dit
    assert a.blocks_to_swap==36 and a.tensor_offload_device=='cpu'


@pytest.mark.parametrize('field,value',[('color_transfer','smpte2084'),('pix_fmt','yuv420p10le'),
                                       ('field_order','tt'),('sample_aspect_ratio','4:3')])
def test_unsupported_sources_rejected(field,value):
    i=info();i['streams'][0][field]=value
    with pytest.raises(ValueError):make_plan(Path('x'),request(),i)


@pytest.mark.parametrize('batch',[1,3,6,8,10,34])
def test_invalid_batch_rejected(batch):
    with pytest.raises(ValueError):make_plan(Path('x'),request(batch=batch),info())


def test_trim_is_explicit():
    p=make_plan(Path('x'),request(start=1,duration=0),info())
    assert p['first']==30 and p['count']==60
    with pytest.raises(ValueError):make_plan(Path('x'),request(duration=30),info())


def test_credentials_excluded(monkeypatch):
    monkeypatch.setenv('COMFY_PASSWORD','private')
    monkeypatch.setenv('HF_TOKEN','private')
    assert 'COMFY_PASSWORD' not in child_env() and 'HF_TOKEN' not in child_env()


def generate(path, audio=True, rate='10', seconds='2'):
    cmd=['ffmpeg','-v','error','-y','-f','lavfi','-i',f'testsrc2=size=96x64:rate={rate}:duration={seconds}']
    if audio:cmd+=['-f','lavfi','-i',f'sine=frequency=440:sample_rate=48000:duration={seconds}']
    cmd+=['-c:v','ffv1','-c:a','pcm_s16le',str(path)]
    subprocess.run(cmd,check=True)


@pytest.mark.parametrize('scale',[1,2])
def test_real_cpu_pipeline_source_bypass(tmp_path,scale):
    import cv2
    src=tmp_path/'source.mkv';generate(src)
    job=tmp_path/'job';job.mkdir()
    req=request(source=str(src),output=str(tmp_path/'output/enhanced.mp4'),strength=0.0,scale=scale)
    plan=make_plan(src,req)
    atomic_json(job/'request.json',req)
    prepare(src,job,plan)
    plan['cuts']=detect_cuts(job/'prepared.mkv',job,plan['count'])
    atomic_json(job/'plan.json',plan)
    render(job)
    img=cv2.imread(str(job/'frames/frame_00000000.png'),cv2.IMREAD_UNCHANGED)
    assert img.dtype==np.uint16 and img.shape[:2]==(64*scale,96*scale)
    finish(job,Path(req['output']),plan)
    v=video(probe(Path(req['output']),True))
    assert [v['width'],v['height']]==[96*scale,64*scale]
    assert int(v['nb_read_frames'])==20 and v['avg_frame_rate']=='10/1'
    assert any(x['codec_type']=='audio' for x in probe(Path(req['output']))['streams'])


def test_real_cut_detection(tmp_path):
    src=tmp_path/'cut.mkv'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=black:s=96x64:r=10:d=1',
                    '-f','lavfi','-i','color=white:s=96x64:r=10:d=1',
                    '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0','-c:v','ffv1',str(src)],check=True)
    assert detect_cuts(src,tmp_path,20)==[10]


def test_mock_ai_real_streams_and_saves(tmp_path):
    src=tmp_path/'source.mkv';generate(src,False,seconds='5')
    job=tmp_path/'job';job.mkdir()
    req=request(source=str(src),strength=0.5)
    plan=make_plan(src,req)
    atomic_json(job/'request.json',req)
    prepare(src,job,plan)
    plan['cuts']=[20]
    atomic_json(job/'plan.json',plan)
    calls=[]
    def fake(raw):
        calls.append(len(raw))
        return np.minimum(raw.astype(np.float32)/255.0+0.1,1)
    render(job,fake)
    assert len(list((job/'frames').glob('*.png')))==50 and max(calls)<=33
    assert json.loads((job/'worker_status.json').read_text())['frames_written']==50


def test_process_group_cleanup():
    p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
    terminate(p,group=True)
    assert p.poll() is not None


@pytest.mark.parametrize('rate',['25','30000/1001'])
def test_fractional_fps_and_trim(tmp_path,rate):
    src=tmp_path/'source.mkv';generate(src,True,rate=rate,seconds='2')
    job=tmp_path/'job';job.mkdir()
    req=request(start=0.2,duration=0.5,strength=0.0,source=str(src))
    plan=make_plan(src,req)
    atomic_json(job/'request.json',req)
    prepare(src,job,plan)
    atomic_json(job/'plan.json',plan)
    render(job)
    output=tmp_path/'output/enhanced.mp4'
    finish(job,output,plan)
    from fractions import Fraction
    v=video(probe(output,True))
    assert Fraction(v['avg_frame_rate'])==Fraction(rate)
    assert int(v['nb_read_frames'])==plan['count']


def test_actual_vfr_is_rejected_without_normalizing(tmp_path):
    src=tmp_path/'vfr.mkv'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=s=96x64:r=10:d=2',
                    '-vf',"select='not(eq(n,7))'",'-fps_mode','vfr','-c:v','ffv1',str(src)],check=True)
    req=request(source=str(src))
    plan=make_plan(src,req)
    job=tmp_path/'job';job.mkdir()
    with pytest.raises(ValueError,match='Variable or discontinuous'):
        prepare(src,job,plan)
