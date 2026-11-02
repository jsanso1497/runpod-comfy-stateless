"""Contract, CPU and real-media tests. No H3 or CUDA inference is simulated as evidence of quality."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace, ModuleType
import sys
import numpy as np
import pytest
from swapkit import independent_media as m, past_recovery as pr
from swapkit import independent_graph as ig
from swapkit.graphs import validate_graph
from swapkit.precision import transform, validate_recipe, DIFFUSION, ENCODER
from swapkit.recovery import media_metadata

ROOT=Path(__file__).resolve().parents[1]


def storage(tmp):
    (tmp/'input').mkdir(exist_ok=True);(tmp/'output').mkdir(exist_ok=True);return tmp


@pytest.mark.parametrize('total',[1,5,22,124,125,240,720,1152,2880])
@pytest.mark.parametrize('overlap',[0,12,24])
@pytest.mark.parametrize('window',[56,124])
def test_planner_and_cuts_are_exact(total,overlap,window):
    rows=m.plan_chunks(total,window,overlap)
    assert rows[0]['start_frame']==0 and rows[-1]['end_frame']==total
    assert all(0<r['frames']<=window for r in rows)
    for policy in ['Middle of overlap','Keep earlier take','Keep later take']:
        trims=m.edit_ranges([r['frames'] for r in rows],overlap,policy)
        delivered=[f for r,t in zip(rows,trims) for f in range(r['start_frame']+t['start'],r['start_frame']+t['end'])]
        assert delivered==list(range(total))


@pytest.mark.parametrize('params',[(0,124,12),(10,125,12),(10,124,99),(10,56,-1)])
def test_bad_plans_fail(params):
    with pytest.raises(ValueError):m.plan_chunks(*params)


def test_paths_reject_remote_duplicates_escape(tmp_path):
    root=storage(tmp_path);p=root/'input/a.mkv';p.write_bytes(b'x')
    assert m.local_media('input/a.mkv',root)==p
    for bad in ['https://example.com/a.mp4','a.mkv','input/../a.mp4','/etc/passwd','input/a.mkv\n']:
        with pytest.raises(ValueError):m.local_media(bad,root)
    with pytest.raises(ValueError):m.ordered_paths('input/a.mkv\ninput/a.mkv',root)
    for bad in ['../escape','bad name','x/y','']:
        with pytest.raises(ValueError):m.safe_name(bad)


def source(root,n=240,fps=24,audio=True):
    path=root/'input/source.mkv'
    frames=np.zeros((n,32,64,3),np.uint8)
    for i in range(n):
        frames[i,:,:,0]=i%251;frames[i,:,:,1]=(i*7)%251;frames[i,:,:,2]=(i*13)%251
        frames[i,:8,:8]=[(i//251)*25,70,120]
    raw=root/'frames.rgb';raw.write_bytes(frames.tobytes())
    args=['ffmpeg','-v','error','-y','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','64x32','-r',str(fps),'-i',str(raw)]
    if audio:args+=['-f','lavfi','-i','sine=frequency=441:sample_rate=48000']
    args+=['-map','0:v:0','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','1']
    if audio:args+=['-map','1:a:0','-c:a','pcm_s16le','-ar','48000','-ac','2']
    args+=['-t',str(n/fps),str(path)]
    subprocess.run(args,check=True,capture_output=True);return path,frames


def rgb(path):
    data=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(path),'-map','0:v:0','-an','-pix_fmt','rgb24','-f','rawvideo','-threads','1','pipe:1'])
    return np.frombuffer(data,np.uint8).reshape(-1,32,64,3)


@pytest.mark.parametrize('overlap,policy',[(0,'Middle of overlap'),(12,'Middle of overlap'),(12,'Keep earlier take'),(12,'Keep later take')])
def test_real_split_join_frame_exact_and_metadata(tmp_path,overlap,policy):
    root=storage(tmp_path);src,frames=source(root,n=250)
    manifest,path=m.split_source('input/source.mkv','take',0.,0.,124,overlap,root)
    assert manifest['frames']==250
    text='\n'.join(r['file'] for r in manifest['clips'])
    metadata={'workflow':{'version':.4,'nodes':[{'id':1,'type':'Note','widgets_values':['Keep caf\u00e9 = # ; \\ and\nnewlines']}],'links':[]},
              'prompt':{'1':{'class_type':'Test','inputs':{'seed':42}}}}
    result=m.join_videos(text,'joined',overlap,policy,'Original source','input/source.mkv',0.,16,metadata,root)
    np.testing.assert_array_equal(rgb(result['master']),frames)
    assert m.video_info(result['video'])['frames']==250
    tags=m.video_info(result['video'],False)['format']['tags']
    assert json.loads(tags['workflow'])==metadata['workflow']
    assert json.loads(tags['prompt'])==metadata['prompt']
    assert Path(result['video']+'.workflow.json').exists()
    assert m.video_info(result['master'],False)['audio'] is not None
    assert result['seamless_guarantee'] is False


def test_48_seconds_11_sections_complete_tail(tmp_path):
    root=storage(tmp_path);src,frames=source(root,n=1152,audio=False)
    manifest,_=m.split_source('input/source.mkv','fortyeight',48.,0.,124,12,root)
    assert len(manifest['clips'])==11 and manifest['clips'][-1]['frames']==32
    result=m.join_videos('\n'.join(r['file'] for r in manifest['clips']),'complete',12,
         'Middle of overlap','No audio','',0.,16,{},root)
    assert result['frames']==1152 and result['seconds']==48.
    np.testing.assert_array_equal(rgb(result['master']),frames)
    assert m.video_info(result['video'],False)['audio'] is None


def test_start_offset_and_one_frame_tail(tmp_path):
    root=storage(tmp_path);src,frames=source(root,n=149)
    manifest,_=m.split_source('input/source.mkv','tail',0.,1.,124,0,root)
    assert [r['frames'] for r in manifest['clips']]==[124,1]
    out=m.join_videos('\n'.join(r['file'] for r in manifest['clips']),'tail_join',0,
          'Middle of overlap','Original source','input/source.mkv',1.,16,{},root)
    assert out['frames']==125;np.testing.assert_array_equal(rgb(out['master']),frames[24:])


def test_overwrite_wrong_fps_and_duplicate_are_rejected(tmp_path):
    root=storage(tmp_path);src,_=source(root,n=125,fps=25)
    with pytest.raises(ValueError,match='24 fps'):
        m.join_videos('input/source.mkv','bad',0,'Middle of overlap','No audio','',0.,16,{},root)
    m.split_source('input/source.mkv','keep',2.,0.,56,12,root)
    with pytest.raises(ValueError,match='exists'):m.split_source('input/source.mkv','keep',2.,0.,56,12,root)


def test_cancel_does_not_publish_ready_manifest(tmp_path):
    root=storage(tmp_path);src,_=source(root,24)
    def cancel():raise RuntimeError('User cancel')
    with pytest.raises(RuntimeError,match='cancel'):
        m.split_source('input/source.mkv','cancel',1.,0.,124,12,root,cancel)
    assert not (root/'input/h3_independent/cancel/chunks.json').exists()
    assert src.exists()


def test_receipts_reject_wrong_order_overlap_source_and_profile(tmp_path):
    root=storage(tmp_path);paths=[];infos=[]
    for i,(start,end) in enumerate([(0,124),(112,236)]):
        p=root/f'output/part{i}/master.mkv';p.parent.mkdir();p.write_bytes(b'x');paths.append(p);infos.append({'frames':124})
        r={'format':'h3-independent-render-v1','model_profile':'baseline','source_clip':{
            'source_sha256':'a','set_name':'same','source_start_seconds':0.,'total_frames':236,
            'frames':124,'start_frame':start,'end_frame':end,'overlap_frames':12}}
        (p.parent/'receipt.json').write_text(json.dumps(r))
    assert m.validate_provenance(paths,infos,12).startswith('Receipt-verified')
    for ps,ov in [(paths[::-1],12),(paths,0)]:
        with pytest.raises(ValueError):m.validate_provenance(ps,infos,ov)
    r=json.loads((paths[1].parent/'receipt.json').read_text());r['model_profile']='bf16_full'
    (paths[1].parent/'receipt.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='mixed'):m.validate_provenance(paths,infos,12)


def native_fixture():
    """API-derived port skeleton for topology tests, not GPU/UI qualification."""
    api=json.loads((ROOT/'swapkit/data/akatz_disk_base.api.json').read_text())
    outputs={k:1 for k in api}
    for n in api.values():
        for v in n['inputs'].values():
            if isinstance(v,list) and len(v)==2 and v[0] in outputs:outputs[v[0]]=max(outputs[v[0]],v[1]+1)
    nodes=[]
    for i,(key,n) in enumerate(api.items()):
        widgets={k:v for k,v in n['inputs'].items() if not isinstance(v,list)}
        if key=='window':widgets.setdefault('iteration_index',0)
        if key=='history':widgets.setdefault('history_frames',0)
        node={'id':key,'type':n['class_type'],'pos':[i*400,0],'size':[350,250],'mode':0,'order':i,'flags':{},
              'properties':{'Node name for S&R':n['class_type']},'inputs':[],
              'outputs':[{'name':'out'+str(j),'type':'*','links':[]} for j in range(outputs[key])],
              'widgets_values':list(widgets.values()),'widgets_values_named':widgets}
        nodes.append(node)
    graph={'id':'fixture','version':.4,'nodes':nodes,'links':[],'groups':[],'extra':{}}
    return ig.relink(graph,api)


def test_independent_native_matches_recipe_and_disables_history():
    original,api=native_fixture();before=copy.deepcopy(api)
    graph,job=ig.independent_native(original,api)
    assert validate_graph(graph) and api==before
    assert not {'StartLoop','EndLoop','H3SwapDiskHistory','H3SwapAnalyzeClip'} & {n['class_type'] for n in job.values()}
    assert job['plan']['inputs']['history_frames']=='0' and job['plan']['inputs']['schedule']=='segment'
    assert job['window']['inputs']['iteration_index']==0 and 'previous' not in job['window']['inputs']
    for name in ('model','text','turbo','adapter','sampler','schedule','guider','expand_mask','history'):
        assert job[name]['inputs']==before[name]['inputs']
    assert len([n for n in job.values() if n['class_type']=='LoadImage'])==1
    assert job['source']['class_type']=='H3SwapIndependentSource'
    assert job['save']['class_type']=='H3SwapSaveIndependentChunk'
    bf16=transform(job);validate_recipe(bf16)
    assert bf16['model']['inputs']['unet_name']==DIFFUSION
    assert bf16['text']['inputs']['clip_name']==ENCODER
    low,_=ig.low_resolution(graph,job)
    assert next(n for n in low['nodes'] if n['type']=='H3RelayPrepareSource')['widgets_values_named']['resolution']=='480'


@pytest.mark.parametrize('factory',[ig.prepare_graph,ig.join_graph,ig.recover_graph])
def test_utility_graphs_are_reciprocal(factory):
    canvas,api=factory();assert validate_graph(canvas)
    for n in canvas['nodes']:
        assert n['widgets_values']==list(n['widgets_values_named'].values())
    assert api


def test_recover_sidecar_and_embedded_not_guessing(tmp_path):
    root=storage(tmp_path);src,_=source(root,24,audio=False)
    wf={'version':.4,'nodes':[{'id':1,'type':'Test'}],'links':[]}
    sc=Path(str(src)+'.workflow.json');sc.write_text(json.dumps(wf))
    result=pr.recover_video(src,root,lambda:{})
    assert result['workflow']==str(sc)
    sc.unlink()
    with pytest.raises(ValueError,match='No recorded workflow'):pr.recover_video(src,root,lambda:{})
    out=m.join_videos('input/source.mkv','meta',0,'Middle of overlap','No audio','',0.,16,{'workflow':wf},root)
    Path(out['video']+'.workflow.json').unlink()
    result=pr.recover_video(out['video'],root,lambda:{})
    assert json.loads(Path(result['workflow']).read_text())==wf


def test_matching_live_history_recovery_strips_cache_only(tmp_path):
    root=storage(tmp_path);src,_=source(root,24,audio=False)
    target=root/'output/old.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(src),'-c:v','libx264','-threads','1',str(target)],check=True)
    wf={'version':.4,'nodes':[{'id':1,'type':'Test'}],'links':[]}
    prompt={'1':{'class_type':'StartLoop','inputs':{'seed':42},'is_changed':[float('nan')]}}
    h={'p':{'outputs':{'save':{'videos':[{'type':'output','filename':'old.mp4','subfolder':''}]}},
        'prompt':[0,'p',prompt,{'extra_pnginfo':{'workflow':wf}}]}}
    result=pr.recover_video(target,root,lambda:h)
    assert json.loads(Path(result['workflow']).read_text())==wf
    saved=json.loads(Path(result['prompt']).read_text());assert 'is_changed' not in saved['1']
    assert 'is_changed' in prompt['1']


def test_independent_saver_real_master_receipt_and_metadata(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from swapkit import independent_nodes as nodes
    root=storage(tmp_path);src,frames=source(root,30)
    class Video:
        def __init__(self,path):self.path=str(path)
        def get_stream_source(self):return self.path
        def get_active_trim_window(self):return 0.,0.
    latest=ModuleType('comfy_api.latest');latest.InputImpl=SimpleNamespace(VideoFromFile=Video)
    monkeypatch.setitem(sys.modules,'comfy_api.latest',latest)
    cli=ModuleType('comfy.cli_args');cli.args=SimpleNamespace(disable_metadata=False)
    monkeypatch.setitem(sys.modules,'comfy.cli_args',cli)
    monkeypatch.setattr(nodes,'_preview',lambda path:{'videos':[{'filename':Path(path).name}]})
    monkeypatch.setattr(nodes,'_cancel',lambda:None)
    monkeypatch.setattr(nodes,'data_root',lambda:root)
    monkeypatch.setattr(m,'data_root',lambda:root)
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA','1')
    wf={'version':.4,'nodes':[{'id':1,'type':'Test'}],'links':[]}
    prompt={'1':{'class_type':'StartLoop','inputs':{'seed':123},'is_changed':[float('nan')]}}
    image=torch.from_numpy(frames.astype(np.float32)/255.)
    result=nodes.H3SwapSaveIndependentChunk().save(image,Video(src),'one',16,prompt,{'workflow':wf})
    master=root/result['result'][1]
    encoded=subprocess.check_output(['ffmpeg','-v','error','-threads','1','-i',str(master),'-map','0:v:0','-an','-pix_fmt','rgb48le','-f','rawvideo','-threads','1','pipe:1'])
    np.testing.assert_array_equal(np.frombuffer(encoded,dtype='<u2').reshape(frames.shape),np.rint(image.numpy()*65535).astype('<u2'))
    receipt=json.loads((master.parent/'receipt.json').read_text())
    assert receipt['frames']==30 and receipt['history'].startswith('None')
    assert receipt['metadata_saved']
    assert json.loads(Path(str(master)+'.workflow.json').read_text())==wf
    saved=json.loads(Path(str(master)+'.api.json').read_text());assert 'is_changed' not in saved['1']
    assert 'is_changed' in prompt['1']


def test_native_480_preview_and_saver_present_in_both_profiles():
    from swapkit.preview_graph import native_recoverable,native_preview
    original,api=native_fixture()
    canvas,job=native_recoverable(original,api)
    assert job['save']['class_type']=='H3SwapSaveVideo'
    canvas,job=native_preview(canvas,job)
    assert any(n['type']=='H3SwapWindowPreview' for n in canvas['nodes'])
    for prof in (lambda x:x,transform):
        c,j=ig.low_resolution(prof(canvas),prof(job))
        assert validate_graph(c)
        assert j['prepare']['inputs']['resolution']=='480'
        assert j['save']['class_type']=='H3SwapSaveVideo'
        assert j['reference_conditioning']['inputs']['prompt']=='Replace the selected person with the reference character. Preserve source performance, scene and audio timing.'


def test_recover_from_pre_restart_history_backup(tmp_path):
    root=storage(tmp_path);src,_=source(root,24,audio=False)
    target=root/'output/older.mkv';target.write_bytes(src.read_bytes())
    wf={'version':.4,'nodes':[{'id':1,'type':'PreviousNative'}],'links':[]}
    history={'old':{'outputs':{'save':{'videos':[{'type':'output','filename':'older.mkv','subfolder':''}]}},
        'prompt':[0,'old',{'1':{'class_type':'StartLoop','inputs':{},'is_changed':[float('nan')]}},
                  {'extra_pnginfo':{'workflow':wf}}]}}
    backup=root/'h3-swap/history-backups';backup.mkdir(parents=True)
    (backup/'history.json').write_text(json.dumps(history))
    def unavailable():raise OSError('Comfy history unavailable after restart')
    result=pr.recover_video(target,root,unavailable)
    assert 'backup' in result['source']
    assert json.loads(Path(result['workflow']).read_text())==wf


def test_history_backup_does_not_follow_symlinks(tmp_path):
    root=storage(tmp_path);folder=root/'h3-swap/history-backups';folder.mkdir(parents=True)
    outside=tmp_path/'private.json';outside.write_text('{"not":"history"}')
    (folder/'bad.json').symlink_to(outside)
    assert list(pr.saved_histories(root))==[]
