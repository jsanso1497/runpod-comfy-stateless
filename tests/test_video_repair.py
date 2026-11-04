"""Offline repair contracts and real FFmpeg tests. Never download or load model weights."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import types
from unittest import mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from workbench import video_repair_catalog as catalog
from workbench import video_media as media
from workbench import video_memory as memory
from workbench import video_build
from workbench.seedvr_job import cli_arguments
from workbench.video_process import terminate_process, clean_environment, atomic_json


def info(w=1920, h=1080, fps='30/1', duration='25', audio=True):
    rows = [{'codec_type':'video', 'width':w, 'height':h, 'avg_frame_rate':fps,
             'duration':duration, 'sample_aspect_ratio':'1:1'}]
    if audio:
        rows.append({'codec_type':'audio'})
    return {'streams':rows, 'format':{'duration':duration}}


@pytest.mark.parametrize('mode,final,restored', [
    ('1080p',[1920,1080],[1920,1080]),
    ('4k',[3840,2160],[3840,2160]),
    ('2x3-source',[720,1080],[1080,1620]),
])
def test_geometry(mode, final, restored):
    p=media.source_plan(Path('x.mp4'),mode,0,24,info())
    assert p['final_size']==final and p['restored_size']==restored
    assert p['start']==0 and p['duration']==24


def test_768_left_crop():
    p=media.source_plan(Path('x'),'2x3-source',0,24,info(1366,768))
    assert p['prepared_size']==[512,768]
    assert p['final_size']==[512,768]


@pytest.mark.parametrize('h',range(768,780))
def test_crop_exact_even_2x3(h):
    p=media.source_plan(Path('x'),'2x3-source',0,24,info(1600,h))
    w,ch=p['prepared_size']
    assert w*3==ch*2 and w%2==ch%2==0 and 0<=h-ch<=5


@pytest.mark.parametrize('start,duration',[(0,30),(-1,2),(0,-1),(float('nan'),2)])
def test_bad_trim_rejected(start,duration):
    with pytest.raises(ValueError):media.source_plan(Path('x'),'1080p',start,duration,info())


def test_remaining_duration():
    assert media.source_plan(Path('x'),'1080p',5,0,info())['duration']==20


def test_hdr_rejected():
    i=info();i['streams'][0]['color_transfer']='smpte2084'
    with pytest.raises(ValueError,match='SDR'):media.source_plan(Path('x'),'4k',0,24,i)


def test_anamorphic_rejected():
    i=info();i['streams'][0]['sample_aspect_ratio']='4:3'
    with pytest.raises(ValueError,match='square'):media.source_plan(Path('x'),'1080p',0,24,i)


def test_too_narrow_crop_rejected():
    with pytest.raises(ValueError,match='narrow'):media.source_plan(Path('x'),'2x3-source',0,24,info(300,1080))


def test_portrait_rotation_preserved():
    i=info();i['streams'][0]['side_data_list']=[{'rotation':90}]
    p=media.source_plan(Path('x'),'4k',0,24,i)
    assert p['final_size']==[2160,3840]


def test_catalog_preserves_qwen_models_and_other_workspaces():
    base={'models':[{'id':'qwen','local_nodes':['qwen_native'],'toolbox':['sam','seedvr2'],
                     'comfy_commit':'keep','precision':'BF16'}, {'id':'h3','local_nodes':['h3']} ]}
    original=copy.deepcopy(base)
    result=catalog.apply('models.json',base)
    assert result['models'][0]['comfy_commit']=='keep'
    assert result['models'][0]['precision']=='BF16'
    assert result['models'][1]==original['models'][1]
    assert result['models'][0]['toolbox']==original['models'][0]['toolbox']
    assert result['models'][0]['local_nodes']==['qwen_native','seedvr_stream']
    assert catalog.apply('models.json',result)==result


def test_task_replacement_is_qwen_only():
    value={'tasks':[{'id':'Q01','workspaces':['qwen']}, {'id':'R03','workspaces':['qwen','h3','restoration']} ]}
    before=copy.deepcopy(value['tasks'][0])
    catalog.apply('tasks.json',value)
    assert value['tasks'][0]==before and value['tasks'][1]['workspaces']==['h3','restoration']
    task=value['tasks'][-1]
    assert task['id']=='RVID' and task['requires']==['seedvr2','seedvr2_sharp']
    assert task['workspaces']==['qwen']
    catalog.apply('tasks.json',value)
    assert sum(t['id']=='RVID' for t in value['tasks'])==1


def test_new_asset_is_fp16_pinned_by_hash_not_int8():
    value={'assets':[{'id':'qwen','group':'qwen','sha256':'unchanged'}]}
    catalog.apply('assets.json',value)
    assert value['assets'][0]['sha256']=='unchanged'
    a=value['assets'][1]
    assert a['group']=='seedvr2_sharp' and a['filename'].endswith('_sharp_fp16.safetensors')
    assert a['sha256']=='20a93e01ff24beaeebc5de4e4e5be924359606c356c9c51509fba245bd2d77dd'
    assert len(catalog.apply('assets.json',value)['assets'])==2


def arg(args,name):return args[args.index(name)+1]


@pytest.mark.parametrize('mode',['1080p','4k','2x3-source'])
def test_worker_cli_bounded_and_high_precision(mode):
    request={'mode':mode,'model':'sharp','model_dir':'/models/SEEDVR2','seed':42}
    a=cli_arguments(Path('/job'),request)
    assert int(arg(a,'--chunk_size'))>0 and int(arg(a,'--batch_size'))>=5
    assert arg(a,'--dit_model')==catalog.MODEL
    assert arg(a,'--color_correction')=='wavelet'
    assert arg(a,'--input_noise_scale')==arg(a,'--latent_noise_scale')=='0'
    assert arg(a,'--vae_encode_tile_size')=='512'
    assert arg(a,'--vae_decode_tile_overlap')=='128'
    assert '--swap_io_components' in a and '--compile_dit' not in a and '--cache_dit' not in a
    assert arg(a,'--tensor_offload_device')=='cpu'
    assert arg(a,'--output_format')=='png'


def test_oom_retry_preserves_model_resolution_and_video_context():
    request={'mode':'4k','model':'sharp','model_dir':'/models','seed':42}
    a,b=cli_arguments(Path('/j'),request),cli_arguments(Path('/j'),request,1)
    for k in ['--dit_model','--resolution','--max_resolution','--color_correction','--seed']:
        assert arg(a,k)==arg(b,k)
    assert arg(b,'--vae_encode_tile_size')=='256' and arg(b,'--batch_size')=='5'
    assert arg(b,'--blocks_to_swap')=='36'


def test_environment_has_no_model_credentials(monkeypatch):
    for key in ['WB_PASSWORD','HF_TOKEN','CIVITAI_TOKEN','WB_LORA_URLS']:
        monkeypatch.setenv(key,'private')
    e=clean_environment()
    assert not {'WB_PASSWORD','HF_TOKEN','CIVITAI_TOKEN','WB_LORA_URLS'} & set(e)
    assert e['HF_HUB_OFFLINE']=='1'


def test_owned_child_is_terminated_and_reaped():
    p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
    terminate_process(p,group=True)
    assert p.poll() is not None


def test_atomic_json(tmp_path):
    p=tmp_path/'state.json'
    atomic_json(p,{'phase':'encode'})
    assert json.loads(p.read_text())=={'phase':'encode'}
    assert len(list(tmp_path.iterdir()))==1


def test_memory_guard_preserves_failure_status_and_history(monkeypatch,tmp_path):
    class FakeExecutor:
        def reset(self):self.success=True;self.status_messages=[];self.caches={}
        def execute(self):
            self.success=False;self.status_messages=['failed'];self.history_result={'outputs':{'x':1}};self.caches={'big':'tensor'}
    fake=types.SimpleNamespace(PromptExecutor=FakeExecutor)
    monkeypatch.setitem(sys.modules,'execution',fake)
    monkeypatch.setenv('WB_VIDEO_REPAIR','1');monkeypatch.setenv('WB_DATA_ROOT',str(tmp_path))
    monkeypatch.setattr(memory,'release_models',lambda:['released'])
    monkeypatch.setattr(memory,'snapshot',lambda:{'test':True})
    memory.install_job_cleanup();memory.install_job_cleanup()
    e=FakeExecutor();e.execute()
    assert e.success is False and e.status_messages==['failed'] and e.history_result=={'outputs':{'x':1}}
    assert e.caches=={} and (tmp_path/'workbench/last-job-memory.json').is_file()


def test_memory_guard_runs_on_baseexception(monkeypatch,tmp_path):
    class Cancel(BaseException):pass
    class FakeExecutor:
        def reset(self):self.success=True;self.status_messages=[];self.caches={}
        def execute(self):
            self.success=False;self.status_messages=['cancelled'];self.history_result={};self.caches={'big':'x'}
            raise Cancel()
    fake=types.SimpleNamespace(PromptExecutor=FakeExecutor)
    monkeypatch.setitem(sys.modules,'execution',fake)
    monkeypatch.setenv('WB_VIDEO_REPAIR','1');monkeypatch.setenv('WB_DATA_ROOT',str(tmp_path))
    seen=[]
    monkeypatch.setattr(memory,'release_models',lambda:seen.append(1) or ['released'])
    monkeypatch.setattr(memory,'snapshot',lambda:{})
    memory.install_job_cleanup()
    e=FakeExecutor()
    with pytest.raises(Cancel):e.execute()
    assert seen==[1] and e.success is False and e.status_messages==['cancelled']


def test_build_patch_rejects_changed_upstream(tmp_path):
    p=tmp_path/'custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/src/interfaces/video_upscaler.py'
    p.parent.mkdir(parents=True);p.write_text('unknown source')
    with pytest.raises(RuntimeError,match='changed'):video_build.patch(tmp_path)
    assert p.read_text()=='unknown source'


def test_legacy_cancellation_patch_uses_baseexception():
    assert 'except BaseException:' in video_build.NEW
    assert 'cleanup(dit_cache=False, vae_cache=False)' in video_build.NEW
    assert 'raise e' not in video_build.NEW


def test_workflow_contracts_no_aus_or_whole_clip_image_path():
    paths=list((ROOT/'workflows/restoration/RVID_Streaming').glob('*.json'))
    assert len(paths)==4
    for p in paths:
        g=json.loads(p.read_text())
        ids={n['id']:n for n in g['nodes']}
        assert len(ids)==3
        assert {n['type'] for n in g['nodes']}=={'LoadVideo','WBSeedVR2StreamVideo','Note'}
        assert g['links']==[[1,1,0,2,0,'VIDEO']]
        assert ids[1]['outputs'][0]['links']==[1] and ids[2]['inputs'][0]['link']==1
        assert ids[2]['widgets_values'][2:4]==[0.0,24.0]
        assert len(ids[2]['widgets_values'])==7


def test_no_raw_model_weights_or_shell_exec_in_new_code():
    for p in (ROOT/'src/workbench').glob('video_*.py'):
        ast.parse(p.read_text())
        assert 'shell=True' not in p.read_text()
    for p in (ROOT/'src/workbench').glob('seedvr_*.py'):
        ast.parse(p.read_text())
        assert 'shell=True' not in p.read_text()


def test_portal_lists_saved_files_and_ignores_partial_and_symlinks(tmp_path):
    from workbench.video_portal import list_media
    out=tmp_path/'out';out.mkdir();(out/'real.mp4').write_bytes(b'x');(out/'bad.partial.mp4').write_bytes(b'x')
    (tmp_path/'private.mp4').write_bytes(b'secret')
    try:(out/'link.mp4').symlink_to(tmp_path/'private.mp4')
    except OSError:pass
    rows=list_media(out)
    assert [x['path'] for x in rows]==['real.mp4']
    assert '/_workbench/files?' in rows[0]['download']


def test_portal_escapes_names_in_client():
    from workbench.video_portal import PAGE
    assert 'textContent' in PAGE and 'innerHTML' not in PAGE


def test_output_does_not_use_frame_tensor():
    text=(ROOT/'src/nodes/seedvr_stream/__init__.py').read_text()
    assert 'get_stream_source' in text and 'get_components(' not in text
    assert 'InputImpl.VideoFromFile(str(output))' in text
    assert 'except BaseException' in text and 'terminate_process(proc, group=True)' in text


def test_fused_attention_does_not_modify_qwen():
    worker=(ROOT/'src/workbench/seedvr_worker.py').read_text()
    parent=(ROOT/'src/nodes/seedvr_stream/__init__.py').read_text()
    assert 'enable_math_sdp(False)' in worker and 'enable_math_sdp' not in parent


FFMPEG=bool(shutil.which('ffmpeg') and shutil.which('ffprobe'))


@pytest.fixture
def source24(tmp_path):
    if not FFMPEG:pytest.skip('FFmpeg unavailable')
    p=tmp_path/'source.mkv'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','testsrc2=size=96x72:rate=2:duration=24',
                    '-f','lavfi','-i','sine=frequency=440:sample_rate=8000:duration=24',
                    '-c:v','ffv1','-c:a','pcm_s16le',str(p)],check=True)
    return p


def test_real_24s_prepare_left_crop_audio_frame_count(source24,tmp_path):
    plan=media.source_plan(source24,'2x3-source',0,24)
    job=tmp_path/'job';job.mkdir()
    path,count=media.prepare(source24,job,plan)
    v=media.video_stream(media.probe(path,True))
    assert count==48 and (v['width'],v['height'])==(48,72)
    assert any(x['codec_type']=='audio' for x in media.probe(path)['streams'])


def test_real_24s_finish_mock_inference_preserves_audio(source24,tmp_path):
    from PIL import Image
    plan=media.source_plan(source24,'2x3-source',0,24)
    job=tmp_path/'job';job.mkdir()
    _,count=media.prepare(source24,job,plan)
    frames=job/'restored_frames';frames.mkdir()
    # Mock inference ONLY. Real decode/trim/crop/frame validation/encode below.
    picture=Image.new('RGB',tuple(plan['restored_size']),(90,120,150))
    for i in range(count):picture.save(frames/f'frame_{i:06}.png')
    dest=tmp_path/'output/video/job/restored.mp4'
    media.finish(job,dest,plan,count)
    final=media.probe(dest,True);v=media.video_stream(final)
    assert (v['width'],v['height'],int(v['nb_read_frames']))==(48,72,48)
    assert abs(float(v['duration'])-24)<0.01
    assert any(x['codec_type']=='audio' for x in final['streams'])


@pytest.mark.parametrize('mode,dimensions',[('1080p',[1920,1080]),('4k',[3840,2160])])
def test_real_encode_full_frame_dimensions_without_gpu(tmp_path,mode,dimensions):
    if not FFMPEG:pytest.skip('FFmpeg unavailable')
    from PIL import Image
    src=tmp_path/'src.mkv'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i','color=size=96x54:rate=2:duration=1',
                    '-c:v','ffv1',str(src)],check=True)
    plan=media.source_plan(src,mode,0,1)
    assert plan['final_size']==dimensions
    job=tmp_path/'job';job.mkdir();_,count=media.prepare(src,job,plan)
    frames=job/'restored_frames';frames.mkdir()
    for i in range(count):Image.new('RGB',tuple(dimensions),(90,100,120)).save(frames/f'frame_{i:06}.png')
    dest=tmp_path/'out/restored.mp4';media.finish(job,dest,plan,count)
    v=media.video_stream(media.probe(dest,True))
    assert [v['width'],v['height']]==dimensions and int(v['nb_read_frames'])==2


def test_incomplete_png_sequence_rejected(tmp_path):
    from PIL import Image
    for i in [0,2]:Image.new('RGB',(16,16)).save(tmp_path/f'frame_{i:06}.png')
    with pytest.raises(ValueError,match='complete'):media.find_frames(tmp_path,3)


def test_corrupt_png_rejected(tmp_path):
    p=tmp_path/'frame_000000.png';p.write_bytes(b'not png')
    with pytest.raises(ValueError,match='PNG'):media.find_frames(tmp_path,1)


def test_owned_process_group_including_child_is_reaped(tmp_path):
    if os.name!='posix':pytest.skip('POSIX process group test')
    pidfile=tmp_path/'child.pid'
    code='import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"]); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(120)'
    p=subprocess.Popen([sys.executable,'-c',code,str(pidfile)],start_new_session=True)
    try:
        for _ in range(50):
            if pidfile.exists():break
            time.sleep(.03)
        assert pidfile.exists()
        child=int(pidfile.read_text())
        terminate_process(p,group=True)
        for _ in range(50):
            status=Path(f'/proc/{child}/stat')
            if not status.exists() or status.read_text().split()[2]=='Z':break
            time.sleep(.03)
        assert not status.exists() or status.read_text().split()[2]=='Z'
    finally:terminate_process(p,group=True)


def test_parent_death_signal_prevents_orphan_worker(tmp_path):
    if not sys.platform.startswith('linux'):pytest.skip('Linux parent death test')
    pidfile=tmp_path/'child.pid'
    childcode=('from workbench.video_process import parent_death_signal; import os,time,sys,pathlib; '
               'parent_death_signal(int(sys.argv[2])); pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(120)')
    parentcode='import subprocess,sys,os,time; subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2],str(os.getpid())]); time.sleep(120)'
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
    p=subprocess.Popen([sys.executable,'-c',parentcode,childcode,str(pidfile)],env=env,start_new_session=True)
    try:
        for _ in range(100):
            if pidfile.exists():break
            time.sleep(.03)
        assert pidfile.exists()
        child=int(pidfile.read_text());p.kill();p.wait()
        for _ in range(100):
            status=Path(f'/proc/{child}/stat')
            if not status.exists() or status.read_text().split()[2]=='Z':break
            time.sleep(.03)
        assert not status.exists() or status.read_text().split()[2]=='Z'
    finally:
        with __import__('contextlib').suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGKILL)
        p.wait()


def test_supervisor_oom_retry_with_real_finish(source24,tmp_path,monkeypatch):
    from PIL import Image
    from workbench import seedvr_job as jobmod
    job=tmp_path/'supervised';job.mkdir()
    models=tmp_path/'models';models.mkdir()
    for name in (catalog.MODEL,catalog.VAE):(models/name).write_bytes(b'placeholder, not model weights')
    cli=tmp_path/'inference_cli.py';cli.write_text('--chunk_size --model_dir --output_format --swap_io_components')
    output=tmp_path/'output/restored.mp4'
    request={'source':str(source24),'output':str(output),'cli':str(cli),'model_dir':str(models),
             'mode':'2x3-source','model':'sharp','start':0,'duration':2,'seed':42,'crf':16,'oom_retry':True,
             'keep_intermediates':False,'fused_only':True}
    atomic_json(job/'request.json',request)
    calls=[]
    def fake_gpu(cmd,log,**kwargs):
        calls.append(cmd)
        cfg=json.loads((job/'worker_config.json').read_text())
        if len(calls)==1:
            atomic_json(job/'worker_status.json',{'state':'failed','oom':True,'phase':'vae_decode','error':'CUDA out of memory'})
            return 42
        plan=json.loads((job/'plan.json').read_text())
        count=int(media.video_stream(media.probe(job/'prepared.mkv',True))['nb_read_frames'])
        frames=job/'restored_frames';frames.mkdir()
        for i in range(count):Image.new('RGB',tuple(plan['restored_size']),(50,60,70)).save(frames/f'frame_{i:06}.png')
        atomic_json(job/'worker_status.json',{'state':'complete','phase':'inference_complete'})
        assert arg(cfg['args'],'--vae_encode_tile_size')=='256'
        return 0
    monkeypatch.setattr(jobmod,'run_logged',fake_gpu)
    jobmod.run_job(job)
    assert len(calls)==2 and output.exists()
    state=json.loads((job/'status.json').read_text())
    assert state['state']=='complete' and state['gpu_worker_exited']
    assert not (job/'restored_frames').exists() and source24.exists()


def test_supervisor_non_oom_is_not_retried(source24,tmp_path,monkeypatch):
    from workbench import seedvr_job as jobmod
    job=tmp_path/'badjob';job.mkdir()
    models=tmp_path/'models';models.mkdir()
    for name in (catalog.MODEL,catalog.VAE):(models/name).write_bytes(b'not real weights')
    cli=tmp_path/'inference_cli.py';cli.write_text('--chunk_size --model_dir --output_format --swap_io_components')
    request={'source':str(source24),'output':str(tmp_path/'out/restored.mp4'),'cli':str(cli),'model_dir':str(models),
             'mode':'1080p','model':'sharp','start':0,'duration':1,'seed':42,'crf':16,'oom_retry':True}
    atomic_json(job/'request.json',request)
    calls=[]
    def fake_gpu(*args,**kwargs):
        calls.append(1)
        atomic_json(job/'worker_status.json',{'state':'failed','oom':False,'phase':'dit_restore','error':'No available fused kernel'})
        return 1
    monkeypatch.setattr(jobmod,'run_logged',fake_gpu)
    with pytest.raises(RuntimeError,match='dit_restore'):jobmod.run_job(job)
    assert calls==[1] and not Path(request['output']).exists()


def test_no_model_downloads_in_worker_environment():
    text=(ROOT/'src/workbench/seedvr_job.py').read_text()
    assert 'Required model not prepared' in text
    assert 'clean_environment()' in text
    assert '--output_format' in text


def test_original_oom_survives_upstream_systemexit(tmp_path,monkeypatch):
    from workbench import seedvr_worker
    fake_torch=types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(
        enable_flash_sdp=lambda x:None,enable_mem_efficient_sdp=lambda x:None,enable_math_sdp=lambda x:None)),
        cuda=types.SimpleNamespace(is_available=lambda:False))
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    monkeypatch.setattr(seedvr_worker,'parent_death_signal',lambda *a:None)
    cli=tmp_path/'fake_cli.py'
    cli.write_text('''
def prepare_runner(): return None
def encode_all_batches(): return None
def upscale_all_batches(): raise RuntimeError("CUDA out of memory. Tried to allocate 1 GiB")
def decode_all_batches(): return None
def postprocess_all_batches(): return None
def process_single_file():
    prepare_runner()
    encode_all_batches()
    upscale_all_batches()
def main():
    try: process_single_file()
    except Exception: raise SystemExit(1)
''')
    atomic_json(tmp_path/'worker_config.json',{'cli':str(cli),'args':[],'fused_only':True})
    monkeypatch.setattr(sys,'argv',['worker','--job',str(tmp_path)])
    saved_path=list(sys.path)
    try:rc=seedvr_worker.main()
    finally:sys.path[:]=saved_path
    state=json.loads((tmp_path/'worker_status.json').read_text())
    assert rc==42 and state['oom'] is True and state['phase']=='dit_restore'
    assert state['error_type']=='RuntimeError' and 'Tried to allocate' in state['error']
    assert state['exit_exception_type']=='SystemExit'
