from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from swapkit import config, graphs, media, prompts
from swapkit.build import extract_bfs_template, metadata_lock
from swapkit.gateway import authorized, same_origin


def test_local_canvases_are_valid_and_paired():
    for canvas, api in (graphs.prompt_studio(), graphs.bfs_graph()):
        assert graphs.validate_graph(canvas)
        assert len(api)==len(canvas['nodes'])
        for n in canvas['nodes']:
            assert api[str(n['id'])]['class_type']==n['type']
        for lid,src,out,dst,inp,kind in canvas['links']:
            name=next(n for n in canvas['nodes'] if n['id']==dst)['inputs'][inp]['name']
            assert api[str(dst)]['inputs'][name]==[str(src),out]


def test_bfs_guide_is_not_a_soft_video_reference():
    _,api=graphs.bfs_graph()
    cond=next(n for n in api.values() if n['class_type']=='MiniMaxH3ReferenceToVideo')
    guide=next(n for n in api.values() if n['class_type']=='MiniMaxH3AddGuide')
    assert not any(k.startswith('ref_videos') for k in cond['inputs'])
    assert guide['inputs']['frame_idx']==0 and 'image' in guide['inputs']
    sched=next(n for n in api.values() if n['class_type']=='BasicScheduler')['inputs']
    assert (sched['steps'],sched['scheduler'],sched['denoise'])==(8,'simple',1)
    assert next(n for n in api.values() if n['class_type']=='KSamplerSelect')['inputs']['sampler_name']=='euler'


def test_bfs_uses_own_two_adapters_and_source_soundtrack():
    _,api=graphs.bfs_graph()
    loras=[n['inputs']['lora_name'] for n in api.values() if n['class_type']=='LoraLoaderModelOnly']
    assert len(loras)==2 and any('dmd' in p for p in loras)
    assert not any('fl2v' in p or 'akatz' in p for p in loras)
    prep_id=next(i for i,n in api.items() if n['class_type']=='H3SwapClipWindow')
    videos=[n for n in api.values() if n['class_type']=='CreateVideo']
    assert all(n['inputs']['audio']==[prep_id,1] for n in videos)


def test_prompt_studio_has_no_diffusion_or_loader_branch():
    _,api=graphs.prompt_studio()
    assert not {'UNETLoader','CLIPLoader','SamplerCustomAdvanced'} & {n['class_type'] for n in api.values()}


def test_graph_link_errors_are_rejected():
    g,_=graphs.prompt_studio();g['links'][0][3]=99999
    with pytest.raises(ValueError):graphs.validate_graph(g)


def test_widget_changes_update_both_representations():
    n={'type':'Example','widgets_values_named':{'a':1,'b':2},'widgets_values':[1,2]}
    graphs.set_widget(n,'b',5)
    assert n['widgets_values']==[1,5] and n['widgets_values_named']['b']==5
    n['widgets_values'][0]=3
    with pytest.raises(ValueError):graphs.set_widget(n,'a',5)


def test_preview_pruning_retains_only_dependencies():
    g=graphs.Graph('test')
    a=g.add('LoadImage','input',0,0,outputs=[('IMAGE','IMAGE')])
    b=g.add('PreviewImage','preview',1,0,inputs=[('images','IMAGE')])
    c=g.add('Other','unrelated',2,0,inputs=[('images','IMAGE')])
    g.connect(a,0,b,'images');g.connect(a,0,c,'images')
    result=graphs.prune_to_outputs(g.result()[0],[b])
    assert {n['id'] for n in result['nodes']}=={a,b}
    assert len(result['links'])==1


@pytest.mark.parametrize('value',[0,5,23,120,360,400,True,124.0])
def test_reject_invalid_frame_grid(value):
    with pytest.raises(ValueError):media.frame_count(value)


@pytest.mark.parametrize('value',[22,39,124,243,362])
def test_accept_valid_frame_grid(value):assert media.frame_count(value)==value


@pytest.mark.parametrize('shape',[(1920,1080),(1080,1920),(1024,1024),(2560,1080)])
def test_canvas_is_aligned_and_bounded(shape):
    w,h=media.canvas(*shape,768)
    assert w%32==h%32==0
    assert w*h<=768*768*1.75
    assert abs(w/h-shape[0]/shape[1])<0.1


def test_rotation_and_anamorphic_dimensions():
    assert media.displayed_size({'width':720,'height':480,'sample_aspect_ratio':'4:3'})==(960,480)
    assert media.displayed_size({'width':1920,'height':1080,'side_data_list':[{'rotation':90}]})==(1080,1920)


@pytest.mark.parametrize('path',['../escape','/absolute','a/../../b','x\\bad',''])
def test_storage_escape_rejected(tmp_path,path):
    with pytest.raises(ValueError):config.safe_path(tmp_path,path)


def test_symlink_escape_rejected(tmp_path):
    (tmp_path/'sub').symlink_to('/tmp',target_is_directory=True)
    with pytest.raises(ValueError):config.safe_path(tmp_path,'sub/escaped')


def test_basic_auth():
    import base64
    header='Basic '+base64.b64encode(b'workbench:0123456789abcdef').decode()
    assert authorized(header,'0123456789abcdef')
    assert not authorized(header,'wrong-password')
    assert not authorized('Bearer secret','0123456789abcdef')
    assert not authorized('Basic ???','0123456789abcdef')


def test_origin_protection(monkeypatch):
    monkeypatch.setenv('RUNPOD_POD_ID','abc123')
    r=SimpleNamespace(method='POST',host='internal:8188',headers={'Origin':'https://abc123-8188.proxy.runpod.net'})
    assert same_origin(r)
    r.headers['Origin']='https://attacker.invalid';assert not same_origin(r)
    r.headers={};assert same_origin(r)


def test_bfs_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv('H3_SWAP_VARIANTS','akatz,bfs')
    monkeypatch.delenv('H3_SWAP_BFS_LICENSE_REVIEWED',raising=False)
    with pytest.raises(ValueError):config.variants()
    monkeypatch.setenv('H3_SWAP_BFS_LICENSE_REVIEWED','1')
    assert config.variants()=={'akatz','bfs'}


def test_secrets_not_passed_to_comfy(monkeypatch):
    monkeypatch.setenv('WB_PASSWORD','not-for-child')
    monkeypatch.setenv('HF_TOKEN','not-for-child')
    assert 'HF_TOKEN' not in config.public_child_env()
    assert 'WB_PASSWORD' not in config.public_child_env()


def test_model_metadata_locked_and_release_hash_checked():
    info=SimpleNamespace(sha='a'*40,siblings=[SimpleNamespace(rfilename='model.safetensors',lfs=SimpleNamespace(sha256='b'*64,size=10))])
    class API:
        def model_info(self,*args,**kwargs):return info
    row={'id':'test','group':'common','repo':'org/model','revision':'main','file':'model.safetensors','destination':'loras/test.safetensors','expected_sha256':'b'*64}
    got=metadata_lock([row],API())[0]
    assert got['revision']=='a'*40 and got['bytes']==10
    row['expected_sha256']='c'*64
    with pytest.raises(ValueError):metadata_lock([row],API())


def test_bfs_prompt_only_fills_details():
    template='body_swap: <Subject 1> is the {details} shown in <Picture 1>.'
    assert prompts.bfs_prompt('person in a red shirt',template)==template.replace('{details}','person in a red shirt')
    with pytest.raises(ValueError):prompts.bfs_prompt('something <Picture 2>',template)


def test_template_extraction_rejects_changed_contract():
    with pytest.raises(ValueError):extract_bfs_template('not a template')
    template='body_swap: subject_definitions: {details} <Picture 1> '+('source performance. '*40)
    assert extract_bfs_template('Example: `'+template+'`').startswith('body_swap:')


def test_analysis_json_and_akatz_roles():
    data={'character_details':'person wearing a blue shirt','source_scene':'a kitchen','action_timeline':'0.000s: turns toward the sink','camera':'stationary medium shot'}
    got=prompts.parse_analysis(json.dumps(data))
    prompt=prompts.build_akatz(got,'the person at the sink')
    assert '<Picture 1>' in prompt and '<Video 1>' in prompt and '0.000s' in prompt
    data['camera']='Use <Picture 9>'
    with pytest.raises(ValueError):prompts.parse_analysis(json.dumps(data))


def test_contact_sheet_dimensions():
    from PIL import Image
    result=prompts.contact_sheet([Image.new('RGB',(40,60))]*8,list(range(8)))
    assert result.size==(1536,568)


def test_cache_separates_clips_and_identity(tmp_path):
    p=tmp_path/'source.mp4';p.write_bytes(b'fixture')
    a=prompts.cache_key(p,b'person-a',0,124,'performer','Reference clothing','local')
    b=prompts.cache_key(p,b'person-b',0,124,'performer','Reference clothing','local')
    c=prompts.cache_key(p,b'person-a',5.17,124,'performer','Reference clothing','local')
    assert len({a,b,c})==3


def test_no_remote_backend_or_unbounded_context():
    assert prompts.BASE=='http://127.0.0.1:11434'
    assert prompts.CONTEXT==16384
    assert '262144' not in Path(prompts.__file__).read_text()


def test_image_workflow_does_not_expose_ollama_port():
    root=Path(__file__).resolve().parents[1]
    docker=(root/'Dockerfile').read_text()
    assert 'EXPOSE 8188' in docker and 'EXPOSE 11434' not in docker
    assert 'FROM pytorch/pytorch:2.11.0' in docker


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    if not shutil.which('ffmpeg'):pytest.skip('ffmpeg is not installed')
    root=tmp_path_factory.mktemp('media');path=root/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=160x96:rate=60','-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t','7','-c:v','libx264','-preset','ultrafast','-threads','2','-c:a','aac',str(path)],check=True)
    return path


def test_actual_60fps_clip_has_exact_124_frames_and_audio(synthetic,monkeypatch):
    monkeypatch.setattr(media,'canvas',lambda w,h,short_edge:(160,96))
    images,audio,report=media.decode_clip(synthetic,start_seconds=1,frames=124,short_edge=384)
    assert images.shape==(124,96,160,3)
    assert audio.shape==(2,248000)
    assert report['seconds']==124/24 and report['source_audio']
    assert np.abs(audio).max()>0.01


def test_actual_timestamped_prompt_sampling(synthetic):
    images,times,report=media.sample_frames(synthetic,start_seconds=1,frames=124)
    assert len(images)==8 and times[0]==0 and times[-1]==123/24
    assert report['audio_analyzed'] is False


def test_end_of_video_not_silently_padded(synthetic):
    with pytest.raises(ValueError):media.window(synthetic,6,124)


@pytest.mark.parametrize('bad_json',[False,True])
def test_prompt_api_is_bounded_and_unloads_even_on_parse_failure(monkeypatch,bad_json):
    from PIL import Image
    calls=[]
    analysis={'character_details':'person in a blue shirt','source_scene':'a room','action_timeline':'0.000s: raises a hand','camera':'stationary'}
    class Resp:
        def __init__(self,obj=None,stream=None):self.obj=obj;self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def raise_for_status(self):pass
        def json(self):return self.obj
        def iter_lines(self,**kw):
            yield json.dumps({'message':{'content':'not JSON' if bad_json else json.dumps(analysis)},'done':True,'done_reason':'stop'}).encode()
    class S:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def post(self,url,**kw):
            calls.append((url,kw))
            return Resp({'capabilities':['vision']}) if url.endswith('/api/show') else Resp({})
        def get(self,url,**kw):return Resp({'models':[]})
    monkeypatch.setattr(prompts,'session',lambda:S())
    if bad_json:
        with pytest.raises(ValueError):prompts.query([Image.new('RGB',(40,40))]*2,[0],'the performer')
    else:
        assert prompts.query([Image.new('RGB',(40,40))]*2,[0],'the performer')['source_scene']=='a room'
    payload=next(kw['json'] for url,kw in calls if url.endswith('/api/chat'))
    assert payload['options']['num_ctx']==16384
    assert payload['options']['num_predict']==1800
    assert payload['think'] is False and payload['keep_alive']==0
    assert calls[-1][0].endswith('/api/generate')
    assert calls[-1][1]['json']['keep_alive']==0


def test_new_model_digest_invalidates_prompt_cache(tmp_path):
    p=tmp_path/'video';p.write_bytes(b'bytes')
    args=(p,b'identity',0,124,'performer','Reference clothing','qwen3-vl:8b')
    assert prompts.cache_key(*args,'old')!=prompts.cache_key(*args,'new')


def test_gateway_real_http_auth_and_prompt_preparation_gate():
    import asyncio
    import base64
    from aiohttp import web,ClientSession
    from swapkit.gateway import make_app
    async def exercise():
        state={'ready':False,'stage':'preparing'}
        app=make_app('0123456789abcdef',state)
        runner=web.AppRunner(app);await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        base=f'http://127.0.0.1:{port}'
        auth='Basic '+base64.b64encode(b'workbench:0123456789abcdef').decode()
        try:
            async with ClientSession() as client:
                async with client.get(base+'/healthz') as r:assert r.status==200
                async with client.get(base+'/_swap/status') as r:assert r.status==401
                async with client.get(base+'/_swap/status',headers={'Authorization':auth}) as r:
                    assert r.status==200 and (await r.json())['stage']=='preparing'
                async with client.post(base+'/prompt',json={},headers={'Authorization':auth}) as r:assert r.status==503
                async with client.post(base+'/prompt',json={},headers={'Authorization':auth,'Origin':'https://bad.invalid'}) as r:assert r.status==403
        finally:await runner.cleanup()
    asyncio.run(exercise())


def test_bfs_template_preserves_authors_double_spaces():
    template = 'body_swap: subject_definitions: {details} <Picture 1> ' + ('source performance. ' * 40) + '  summary: preserve\n detailed_description: source'
    result = extract_bfs_template('```text\n' + template + '\n```')
    assert '  summary:' in result and '  detailed_description:' in result
    assert '\n' not in result


def test_short_model_revision_resolved_before_metadata():
    full = '37d7d8b' + 'a' * 33
    class API:
        def list_repo_commits(self, repo, token=False):
            return [SimpleNamespace(commit_id=full)]
        def model_info(self, repo, revision, **kwargs):
            assert revision == full
            return SimpleNamespace(sha=full, siblings=[SimpleNamespace(rfilename='a.safetensors', lfs={'sha256':'b'*64,'size':10})])
    row={'id':'bfs','repo':'org/repo','revision':'37d7d8b','file':'a.safetensors'}
    assert metadata_lock([row],API())[0]['revision']==full
