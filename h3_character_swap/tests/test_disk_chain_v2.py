"""No weights/GPU. Real media I/O, mocked Comfy inference/queue where indicated."""
import copy
import json
from pathlib import Path
import subprocess
import uuid
from unittest.mock import Mock
import numpy as np
import pytest
from swapkit import disk_chain as d, disk_common as c, disk_storage as s, disk_quality as q
from swapkit import disk_control as ctl


def fixture_media(tmp_path,duration=10):
    inp=tmp_path/'input';out=tmp_path/'output';inp.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    src=inp/'source.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i',
        f'testsrc2=size=160x96:rate={24 if duration > 30 else 60}:duration={duration}','-f','lavfi','-i',
        f'sine=frequency=440:sample_rate=48000:duration={duration}',
        '-c:v','libx264','-preset','ultrafast','-threads','2','-crf','16','-pix_fmt','yuv420p',
        '-c:a','aac','-t',str(duration),str(src)],check=True)
    from PIL import Image
    ref=inp/'reference.png';Image.new('RGB',(64,96),'gray').save(ref)
    return inp,out,src,ref


def new_run(tmp_path,duration=10,review='Auto check + retry',retries=2):
    inp,out,src,ref=fixture_media(tmp_path,duration)
    run=d.create_run(source=src,reference=ref,input_root=inp,output_root=out,
        run_name='test_run',duration=duration,review=review,retries=retries)
    return run


def read_rgb(path):
    info=c.verify_video(path)
    raw=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-threads','2',
        '-i',str(path),'-an','-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'])
    return np.frombuffer(raw,np.uint8).reshape(-1,info['height'],info['width'],3).astype(np.float32)/255


class FakeClient:
    """Synthetic source passthrough instead of H3. Exercises real controller/media."""
    def __init__(self,run,fail_attempts=0,crash_at=None):
        self.run=run;self.config=c.read_json(run/'config.json');self.jobs=[];self.frees=0
        self.records={};self.current=None;self.stop_file=None;self.fail_attempts=fail_attempts;self.crash_at=crash_at
    def request(self,path,data=None,timeout=90):
        if path=='/object_info':
            kinds={n['class_type'] for n in c.read_json(d.RECIPE).values()}|{'H3SwapDiskSave','H3SwapDiskHistory','GetVideoComponents','SaveImage','PreviewAny'}
            return {k:{'input':{'required':{}}} for k in kinds}
        if path=='/system_stats': return {'system':{'ram_free':1234},'devices':[]}
        if path=='/queue': return {'queue_running':[],'queue_pending':[]}
        if path.startswith('/history/'):
            ident=path.split('/')[-1];return {ident:self.records[ident]} if ident in self.records else {}
        return {}
    def wait_idle(self,*a,**k): pass
    def clear_memory(self): self.frees+=1
    def cancel_owned(self): pass
    def queue(self,graph):
        self.jobs.append(graph);ident=uuid.uuid4().hex;self.current=ident
        if self.crash_at==len(self.jobs): raise c.ChainError('Simulated process failure before rendering')
        args=graph['disk_save']['inputs'];folder=Path(self.config['output_root'])/args['folder']
        source=read_rgb(Path(self.config['input_root'])/graph['source']['inputs']['file'])
        h=args['history_frames'];n=args['new_frames']
        images=source[h:h+n].copy()
        attempt=int(folder.name.split('_')[-1])
        if h and attempt<self.fail_attempts:
            images=np.clip(images+.6,0,1)
        masks=np.ones(source.shape[:3],dtype=np.float32)
        s.save_chunk(images,source,masks,folder,n,h,'largest',0)
        self.records[ident]={'status':{'status_str':'success','completed':True},'outputs':{}}
        return ident
    def wait(self,ident,*a,**k): self.current=None;return self.records[ident]


@pytest.mark.parametrize('total',[22,56,124,125,141,247,456,2880,4320])
def test_plan_exact_coverage_and_bounded_input(total):
    plan=d.plan_segments(total,124)
    assert sum(x['new_frames'] for x in plan)==total
    covered=0
    for i,row in enumerate(plan):
        assert row['start_frame']==covered
        assert row['history_frames']==(18 if i else 0)
        assert row['input_start']==covered-row['history_frames']
        assert row['input_frames']<=141
        covered+=row['new_frames']
    if total==2880: assert len(plan)==24


def test_graph_preserves_models_sampler_and_source_alignment():
    base=c.read_json(d.RECIPE);config={'resolution':'768','target_person':'person',
        'selection':'largest','selection_index':0,'mask_expansion':64,'prompt':'same',
        'window_frames':124,'seed':904239}
    before=copy.deepcopy(base);seg=d.plan_segments(456)[1]
    graph=d.make_job(base,config,seg,'clip.mkv','ref.png','test/attempt',
        {'checkpoint':'history.npz','checkpoint_sha256':'a'*64})
    for key in ('model','text','adapter','turbo','sampler','schedule','history','edit_mask'):
        assert graph[key]==base[key]
    assert not {'start','end','video','save'}&set(graph)
    assert graph['prepare']['inputs']['duration_seconds']==0
    assert graph['plan']['inputs']['schedule']=='segment'
    assert graph['track']['inputs']['initial_mask']==['checkpoint',1]
    assert graph['checkpoint']['class_type']=='H3SwapDiskHistory'
    assert graph['disk_save']['inputs']['raw_masks']==['recover',0]
    assert all(graph[k]['inputs']['initial_history']==['checkpoint',0] for k in ('plan','window','append'))
    assert base==before
    assert c.validate_api_graph(graph)


def test_retry_seeds_keep_first_attempt_and_change_next():
    assert d.retry_seed(904239,9,0)==904239
    assert d.retry_seed(904239,9,1)!=904239
    assert d.retry_seed(2**64-1,50,3)<2**64


@pytest.mark.parametrize('name',['../x','a/b','','..','x y','a'*65])
def test_invalid_run_names(tmp_path,name):
    with pytest.raises(c.ChainError): d.resolve_run(tmp_path,name)


def test_loopback_only_client():
    with pytest.raises(c.ChainError): c.ComfyClient('https://attacker.example')


def test_npz_blocks_object_and_oversize(tmp_path):
    path=tmp_path/'bad.npz';np.savez(path,frames=np.array([object()]))
    with pytest.raises(ValueError): s.load_arrays(path)
    np.savez(path,frames=np.zeros((100,),dtype=np.float32))
    with pytest.raises(c.ChainError): s.load_arrays(path,max_bytes=10)


def test_checkpoint_preserves_float_values_and_raw_mask(tmp_path):
    rng=np.random.default_rng(5)
    f=rng.random((22,64,96,3),dtype=np.float32)
    masks=rng.random((22,64,96),dtype=np.float32)
    receipt=s.save_chunk(f,f,masks,tmp_path/'chunk',22,0,'largest',0)
    arrays=s.load_arrays(tmp_path/'chunk/history.npz')
    assert np.array_equal(arrays['frames'],f[-18:])
    assert np.array_equal(arrays['masks'],masks[-18:])
    assert receipt['history_frames']==18
    assert s.verify_receipt(tmp_path/'chunk',22)
    with pytest.raises(c.ChainError): s.save_chunk(f,f,masks,tmp_path/'chunk',22,0,'largest',0)


def qc(frames):
    masks=np.ones(frames.shape[:3],dtype=np.float32)
    return {'head':frames[:6],'tail':frames[-6:],'source_head':frames[:6],
            'source_tail':frames[-6:],'mask_head':masks[:6],'mask_tail':masks[-6:]}


def test_quality_pass_is_not_guarantee():
    f=np.full((6,64,96,3),.2,dtype=np.float32)
    result=q.assess_join(qc(f),qc(f))
    assert result['verdict']=='pass'
    assert 'does not certify' in result['limitations']


def test_quality_flags_brightness_jump():
    f=np.full((6,64,96,3),.2,dtype=np.float32)
    a,b=qc(f),qc(f)
    b['head']=np.full_like(f,.9)
    result=q.assess_join(a,b)
    assert result['verdict']=='review'
    assert 'brightness_jump' in result['flags']


def test_quality_flags_freeze_when_source_moves():
    f=np.stack([np.full((64,96,3),.2+i*.04,dtype=np.float32) for i in range(6)])
    a,b=qc(f),qc(f)
    b['head']=np.full_like(f,.4)
    assert 'possible_frozen_continuation' in q.assess_join(a,b)['flags']


def test_quality_flags_empty_mask():
    f=np.ones((6,64,96,3),dtype=np.float32)*.2
    a,b=qc(f),qc(f);a['mask_tail']*=0;b['mask_head']*=0
    assert q.assess_join(a,b)['flags']==['missing_subject_mask']


def test_real_media_ten_seconds_and_continuation(tmp_path):
    run=new_run(tmp_path,10)
    fake=FakeClient(run)
    state=d.execute(run,fake)
    assert state['status']=='complete'
    assert state['final_info']['frames']==240 and state['final_info']['audio']
    assert len(fake.jobs)==2 and fake.frees>=3
    assert fake.jobs[1]['checkpoint']['inputs']['checkpoint'].endswith('history.npz')
    assert (Path(state['chunks'][1]['folder'])/'join_review.mp4').is_file()


def test_real_media_retry_only_flagged_chunk(tmp_path):
    run=new_run(tmp_path,10)
    fake=FakeClient(run,fail_attempts=1)
    state=d.execute(run,fake)
    assert state['status']=='complete'
    assert len(fake.jobs)==3
    assert state['chunks'][0]['attempt']==0 and state['chunks'][1]['attempt']==1
    assert (run/'part_0002/attempt_00/quality.json').is_file()


def test_exhausted_retries_pause_then_approval_preserves_prefix(tmp_path,monkeypatch):
    run=new_run(tmp_path,10,retries=0)
    fake=FakeClient(run,fail_attempts=1)
    state=d.execute(run,fake)
    assert state['status']=='needs_review'
    before=c.sha256(Path(state['chunks'][0]['folder'])/'master.mkv')
    monkeypatch.setattr(ctl,'launch',lambda run:123)
    ctl.act(run.parents[2],run.name,'Approve candidate and continue')
    state=d.execute(run,fake)
    assert state['status']=='complete' and len(fake.jobs)==2
    assert c.sha256(Path(state['chunks'][0]['folder'])/'master.mkv')==before
    assert state['chunks'][1]['manual_approval']


def test_review_every_chunk_pauses_before_next_job(tmp_path):
    run=new_run(tmp_path,10,review='Review every chunk')
    fake=FakeClient(run);state=d.execute(run,fake)
    assert state['status']=='needs_review' and not state['chunks'] and len(fake.jobs)==1


def test_resume_refuses_changed_input(tmp_path):
    run=new_run(tmp_path,10)
    config=c.read_json(run/'config.json')
    Path(config['reference']).write_bytes(b'changed')
    with pytest.raises(c.ChainError,match='reference changed'):d.execute(run,FakeClient(run))


def test_resume_refuses_changed_accepted_master(tmp_path):
    run=new_run(tmp_path,10,retries=0)
    fake=FakeClient(run,fail_attempts=1);state=d.execute(run,fake)
    p=Path(state['chunks'][0]['folder'])/'master.mkv'
    with p.open('ab') as out:out.write(b'changed')
    with pytest.raises(c.ChainError,match='Accepted chunk changed'):d.execute(run,fake)


def test_stop_file_causes_clean_stop(tmp_path):
    run=new_run(tmp_path,2)
    (run/'STOP').write_text('stop')
    with pytest.raises(c.Stopped):d.execute(run,FakeClient(run))
    assert c.read_json(run/'state.json')['status']=='stopped'


def test_18_frame_history_loader_validates_and_returns_matching_initial_mask(tmp_path,monkeypatch):
    pytest.importorskip('torch', reason='Only the built image requires Torch; array I/O is tested separately')
    from swapkit import disk_nodes
    out=tmp_path/'output';p=out/'h3_character_swap/disk_chain_v2/test/part/history.npz';p.parent.mkdir(parents=True)
    f=np.random.default_rng(6).random((18,64,96,3),dtype=np.float32)
    m=np.random.default_rng(7).random((18,64,96),dtype=np.float32)
    s.save_arrays(p,frames=f,masks=m)
    monkeypatch.setattr(disk_nodes,'_inputs',lambda:(tmp_path/'input',out))
    images,mask,selection=disk_nodes.H3SwapDiskHistory().load(str(p),c.sha256(p),'leftmost',0)
    assert np.array_equal(images.numpy(),f)
    assert np.array_equal(mask.numpy(),m[:1])
    assert selection=={'selection':'leftmost','index':0}
    with pytest.raises(c.ChainError,match='checksum'):disk_nodes.H3SwapDiskHistory().load(str(p),'0'*64,'largest',0)


def test_start_node_does_not_wait_for_own_queue():
    import inspect
    from swapkit.disk_nodes import H3SwapDiskChain
    src=inspect.getsource(H3SwapDiskChain.control)
    assert 'launch(run)' in src
    assert 'execute(' not in src and '.wait(' not in src


def test_ffv1_master_preserves_encoded_16bit_rgb(tmp_path):
    f=np.random.default_rng(22).random((5,32,64,3),dtype=np.float32)
    path=tmp_path/'master.mkv';s.write_master(f,path)
    raw=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-threads','2','-i',str(path),
        '-an','-f','rawvideo','-pix_fmt','rgb48le','-threads','2','pipe:1'])
    back=np.frombuffer(raw,dtype='<u2').reshape(f.shape)
    assert np.array_equal(back,np.rint(f*65535).astype('<u2'))


def test_failed_second_job_resumes_without_resampling_first(tmp_path):
    run=new_run(tmp_path,10)
    first=FakeClient(run,crash_at=2)
    with pytest.raises(c.ChainError,match='Simulated'):d.execute(run,first)
    state=c.read_json(run/'state.json')
    assert len(state['chunks'])==1
    path=Path(state['chunks'][0]['folder'])/'master.mkv';digest=c.sha256(path)
    second=FakeClient(run)
    finished=d.execute(run,second)
    assert finished['status']=='complete' and len(second.jobs)==1
    assert finished.get('error') is None
    assert c.sha256(path)==digest


def test_runtime_signature_detects_weight_replacement(tmp_path):
    inp=tmp_path/'input';inp.mkdir()
    model=tmp_path/'models/diffusion_models/test.safetensors';model.parent.mkdir(parents=True);model.write_bytes(b'a')
    base={'model':{'class_type':'UNETLoader','inputs':{'unet_name':'test.safetensors'}}}
    a=d.runtime_signature(inp,base);model.write_bytes(b'longer')
    b=d.runtime_signature(inp,base)
    assert a!=b


def test_control_canvases_pair_with_the_api_and_use_only_registered_new_nodes():
    from swapkit import graphs,disk_nodes
    root=Path(d.__file__).parents[1]/'workflows'
    for name in ('06_H3_LowMemory_Chain_Control','08_H3_Prepare_5s_Clips'):
        canvas=c.read_json(root/(name+'.json'));api=c.read_json(root/(name+'.api.json'))
        assert graphs.validate_graph(canvas)
        for node in canvas['nodes']:
            if node['type']=='Note': continue
            assert api[str(node['id'])]['class_type']==node['type']
            if node['type'].startswith('H3Swap'):assert node['type'] in disk_nodes.NODE_CLASS_MAPPINGS


def test_internal_checkpoint_nodes_are_in_live_smoke_contract(tmp_path,monkeypatch):
    from swapkit import check
    work=tmp_path/'workflows';work.mkdir()
    (work/'06_H3_LowMemory_Chain_Control.json').write_text(json.dumps({'nodes':[{'type':'H3SwapDiskChain'}]}))
    monkeypatch.setattr(check,'ROOT',tmp_path)
    schema={k:{} for k in ('H3SwapDiskChain','H3SwapDiskSave','H3SwapDiskHistory','H3SwapClipPrep','PreviewAny')}
    assert check.inspect_schema(schema,check_models=False)['ok']
    del schema['H3SwapDiskHistory']
    assert check.inspect_schema(schema,check_models=False)['missing_nodes']==['H3SwapDiskHistory']


def test_browser_routes_require_auth_and_reject_cross_site_writes(tmp_path,monkeypatch):
    import asyncio,base64
    from aiohttp.test_utils import TestServer,TestClient
    from swapkit.gateway import make_app
    monkeypatch.setenv('H3_SWAP_DATA',str(tmp_path))
    async def exercise():
        app=make_app('0123456789abcdef',{'ready':True})
        async with TestClient(TestServer(app)) as client:
            r=await client.get('/_swap/chain');assert r.status==401
            auth={'Authorization':'Basic '+base64.b64encode(b'workbench:0123456789abcdef').decode()}
            r=await client.get('/_swap/chain',headers=auth);assert r.status==200
            assert 'not a seamless-motion' in await r.text()
            r=await client.get('/_swap/chain/runs',headers=auth);assert r.status==200 and await r.json()==[]
            r=await client.post('/_swap/chain/action',json={'run':'x','action':'Stop'},headers={**auth,'Origin':'https://attacker.example'})
            assert r.status==403
            r=await client.post('/_swap/chain/action',data='x',headers=auth);assert r.status==415
            r=await client.post('/_swap/chain/action',json={'run':'../escape','action':'Stop'},headers=auth)
            assert r.status==400
    asyncio.run(exercise())


def test_clip_prep_in_isolated_image_uses_exact_existing_engine(tmp_path):
    from swapkit import clip_prep
    inp,out,src,ref=fixture_media(tmp_path,6)
    report=clip_prep.export_clips(src,out,inp,mode='Consecutive chunks',make_zip=False,chunk_count=1)
    assert len(report['clips'])==1
    files=list(inp.glob('h3clip_*.mp4'))
    assert len(files)==1
    assert c.verify_video(files[0],124)['fps']==24


def test_two_minutes_from_three_minute_source_real_ffmpeg_not_h3(tmp_path):
    inp,out,src,ref=fixture_media(tmp_path,180)
    run=d.create_run(source=src,reference=ref,input_root=inp,output_root=out,
                     run_name='long_test',duration=120)
    fake=FakeClient(run)
    result=d.execute(run,fake)
    assert result['status']=='complete'
    assert len(result['chunks'])==24 and len(fake.jobs)==24
    assert result['final_info']['frames']==2880 and result['final_info']['audio']
    assert max(j['disk_save']['inputs']['new_frames']+j['disk_save']['inputs']['history_frames'] for j in fake.jobs)<=141


def test_tall_reference_quality_thumbnails_stay_bounded():
    a=np.zeros((1500,250,3),dtype=np.float32)
    b=s.thumb(a)
    assert max(b.shape[:2])<=160
    assert b.shape[0]>b.shape[1]


def test_checkpoint_fingerprint_cannot_read_outside_output(tmp_path,monkeypatch):
    from swapkit import disk_nodes
    monkeypatch.setattr(disk_nodes,'_inputs',lambda:(tmp_path/'input',tmp_path/'output'))
    outside=tmp_path/'secret';outside.write_text('private')
    with pytest.raises(c.ChainError):
        disk_nodes.H3SwapDiskHistory.IS_CHANGED(str(outside))


def test_low_disk_aborts_master_write_without_final_file(tmp_path,monkeypatch):
    f=np.zeros((5,32,64,3),dtype=np.float32)
    monkeypatch.setattr(s.shutil,'disk_usage',lambda p:Mock(free=0))
    with pytest.raises(c.ChainError,match='disk space'):
        s.write_master(f,tmp_path/'master.mkv')
    assert not (tmp_path/'master.mkv').exists()
    assert not (tmp_path/'master.partial.mkv').exists()


@pytest.mark.parametrize('window',[56,73,124])
def test_extraction_preserves_actual_frame_indices_not_only_counts(tmp_path,window):
    # 24 fps Matroska millisecond PTS rounds both up and down. Test the smaller
    # fallback as well as the default, against actual known pixel-frame IDs.
    ids=np.arange(456,dtype=np.int32)
    colors=np.stack([ids%256,(ids//256)*100,(ids*7)%256],axis=-1).astype(np.float32)/255
    frames=np.broadcast_to(colors[:,None,None,:],(456,32,64,3)).copy()
    master=tmp_path/'master.mkv';s.write_master(frames,master)
    source=tmp_path/'normalized.mkv'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(master),
        '-c:v','ffv1','-g','1','-pix_fmt','bgr0','-threads','2',str(source)],check=True)
    # Test the same 8-bit disk-normalized input used by the controller. The
    # separate master test above already checks exact 16-bit encoding.
    frames=read_rgb(source)
    logs=tmp_path/'logs';logs.mkdir()
    plan=d.plan_segments(len(frames),window)
    for seg in plan[1:4]:
        target=tmp_path/f"piece_{seg['index']}.mkv"
        c.extract_segment(source,target,seg,logs)
        back=read_rgb(target)
        expected=frames[seg['input_start']:seg['input_start']+seg['input_frames']]
        assert np.array_equal(np.rint(back*255).astype(np.uint8),np.rint(expected*255).astype(np.uint8))
