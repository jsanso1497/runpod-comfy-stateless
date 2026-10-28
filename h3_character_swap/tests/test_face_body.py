"""CPU tests. Ollama and H3 are fixtures; no claim about generated likeness."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
import pytest
from PIL import Image
from swapkit import face_body as f, disk_chain as d, disk_common as c
from swapkit import analysis_options as opt, akatz_analysis as a, analysis_nodes as an
from swapkit import graphs, disk_nodes as dn, disk_control as ctl, check
from test_long_analysis import evidence, FakeOllama
from test_disk_chain_v2 import fixture_media, FakeClient


def options(**kwargs):
    return opt.AnalysisOptions(reference_count=2, reference_layout=opt.DUAL_LAYOUT, **kwargs).validate()


def files(tmp_path):
    inp=tmp_path/'input';out=tmp_path/'output';inp.mkdir();out.mkdir()
    src=inp/'source.mp4';src.write_bytes(b'test source, not media')
    face=inp/'face.png';body=inp/'body.png'
    Image.new('RGB',(48,64),(80,50,30)).save(face)
    Image.new('RGB',(48,96),(50,80,30)).save(body)
    return inp,out,src,face,body


def run_config(tmp_path, additional=''):
    inp,out,src,face,body=files(tmp_path)
    run=d.create_run(source=src,reference=face,body_reference=body,input_root=inp,output_root=out,
                     run_name='dual',duration=48,review='Review every chunk',prompt=additional)
    return run,c.read_json(run/'config.json')


def test_role_contract_is_complete_and_idempotent():
    text=f.compose_prompt('person in blue','Keep the jacket fasteners.')
    assert f.compose_prompt('person in blue',text)==text
    assert text.count(f.MARKER)==1 and text.count(f.CLOSING)==1
    for term in ('<Picture 1>','<Picture 2>','<Video 1>','not only the face','complete outfit',
                 'Ignore any competing face','current segment','other people'):
        assert term.lower() in text.lower()


@pytest.mark.parametrize('bad',['Use <Picture 3>','Use <Video 2>','Use <Subject 1>','\x00'])
def test_bad_reference_roles_rejected(bad):
    with pytest.raises(ValueError): f.compose_prompt('person',bad)


def test_target_mismatch_in_pasted_draft_is_not_silent():
    draft=f.compose_prompt('person in blue','Face/body details')
    with pytest.raises(ValueError,match='different target'): f.compose_prompt('person in red',draft)


def test_two_ref_prompt_uses_body_and_face_evidence_without_replaying_timeline():
    obj=evidence();clip,chain=f.build_prompts(obj,'person')
    assert 'UNIQUE REFERENCE OUTFIT' in chain
    assert obj['face_hair'] in chain and obj['body_proportions'] in chain
    for specific in ('UNIQUE ACTION','UNIQUE ROOM','UNIQUE CAMERA'):
        assert specific in clip and specific not in chain
    assert 'UNIQUE HUMAN REVIEW' not in clip+chain
    assert f.compose_prompt('person',chain)==chain


def test_default_budget_counts_both_images():
    assert options().reference_count==2
    assert options(max_output_tokens=32768,context_tokens=65536,sampled_frames=32)
    with pytest.raises(ValueError): opt.AnalysisOptions(reference_count=2).validate()
    with pytest.raises(ValueError): options(sampled_frames=17)
    with pytest.raises(ValueError): opt.AnalysisOptions(reference_count=3).validate()


def test_ollama_receives_face_then_body_then_samples(monkeypatch):
    fake=FakeOllama();monkeypatch.setattr(a.prompts,'session',lambda:fake)
    captured=[]
    def encode(image,max_edge=768):
        captured.append((image.size,max_edge));return 'image-data'
    monkeypatch.setattr(a.prompts,'jpeg_b64',encode)
    face=Image.new('RGB',(60,80));body=Image.new('RGB',(70,150));source=Image.new('RGB',(100,60))
    obj,stats=a.query([face,body]+[source]*16,[i/3 for i in range(16)],'person',options())
    payload=next(kw['json'] for url,kw in fake.calls if url.endswith('/api/chat'))
    assert payload['messages'][0]['content']==f.DUAL_SYSTEM
    assert 'Images 3 onward' in payload['messages'][1]['content']
    assert 'Separate face and body images' in payload['messages'][1]['content']
    assert len(payload['messages'][1]['images'])==18
    assert captured[:2]==[((60,80),768),((70,150),768)]
    assert all(edge==576 for size,edge in captured[2:])
    assert stats['requested']['reference_count']==2
    assert payload['options']=={'num_ctx':32768,'num_predict':16384,'temperature':0.1,'seed':42}
    assert fake.calls[-1][0].endswith('/api/generate')


def test_two_ref_cache_key_covers_both_pixels_shapes_and_order(tmp_path):
    inp,out,src,face,body=files(tmp_path)
    def key(fp=b'face',bp=b'body',fs=(20,10,3),bs=(40,10,3)):
        return a.cache_key(src,fp,fs,0,124,None,'person','Reference clothing','qwen','digest',options(),
                           body_bytes=bp,body_shape=bs)
    keys=[key(),key(fp=b'changed'),key(bp=b'changed'),key(bs=(10,40,3)),key(fs=(10,20,3)),
          key(fp=b'body',bp=b'face')]
    assert len(set(keys))==len(keys)
    with pytest.raises(ValueError):
        a.cache_key(src,b'face',(20,10,3),0,124,None,'person','Reference clothing','qwen','digest',options())


def test_config_persists_both_hashes_roles_and_effective_prompt(tmp_path):
    run,conf=run_config(tmp_path,'Reference jacket is dark green.')
    assert conf['reference_mode']=='face_body'
    assert conf['reference_roles_version']==f.CONTRACT
    assert conf['body_reference_sha256']==c.sha256(Path(conf['body_reference']))
    assert conf['reference_sha256']==c.sha256(Path(conf['reference']))
    assert (run/'effective_prompt.txt').read_text().strip()==conf['prompt']
    assert 'Reference jacket is dark green.' in conf['prompt']
    summary=ctl.summarize(run)
    assert summary['reference_files']=={'Picture 1':'face.png','Picture 2':'body.png'}


@pytest.mark.parametrize('case',['outside','missing','bad_extension'])
def test_create_run_rejects_invalid_body_file(tmp_path,case):
    inp,out,src,face,body=files(tmp_path)
    if case=='outside': body=tmp_path/'outside.png';Image.new('RGB',(3,3)).save(body)
    elif case=='missing': body=inp/'missing.png'
    else: body=inp/'not_image.txt';body.write_text('not image')
    with pytest.raises(c.ChainError,match='Body reference'):
        d.create_run(source=src,reference=face,body_reference=body,input_root=inp,output_root=out,run_name='bad')
    assert not (out/'h3_character_swap/disk_chain_v2/bad').exists()


def test_every_chunk_and_retry_has_exact_native_two_slot_connections(tmp_path):
    run,conf=run_config(tmp_path)
    base=c.read_json(d.RECIPE);original=copy.deepcopy(base)
    previous={'checkpoint':'prior/history.npz','checkpoint_sha256':'a'*64}
    plan=c.plan_segments(48*24,124);assert len(plan)==10
    for segment in plan:
        for attempt in (0,1,2):
            graph=d.make_job(base,conf,segment,'current.mkv','face.png','candidate',
                             previous if segment['history_frames'] else None,attempt,
                             body_reference_file='body.png')
            cond=graph['reference_conditioning']['inputs']
            assert cond['ref_images.ref_image_0']==['reference',0]
            assert cond['ref_images.ref_image_1']==['body_reference',0]
            assert graph['reference']['inputs']['image']=='face.png'
            assert graph['body_reference']['inputs']['image']=='body.png'
            assert cond['vae']==['video_vae',0] and cond['ref_image_size']=='match'
            assert cond['prompt']==conf['prompt']
            for name in ('model','text','adapter','turbo','sampler','schedule','guide','history','sampling_memory'):
                assert graph[name]==base[name]
            assert graph['track']['inputs']['max_objects']==1
            if segment['history_frames']:
                assert graph['plan']['inputs']['initial_history']==['checkpoint',0]
                assert graph['track']['inputs']['initial_mask']==['checkpoint',1]
            assert c.validate_api_graph(graph)
    assert base==original


def test_body_connection_cannot_silently_disappear(tmp_path):
    run,conf=run_config(tmp_path);base=c.read_json(d.RECIPE)
    with pytest.raises(c.ChainError,match='missing its body'):
        d.make_job(base,conf,c.plan_segments(124)[0],'x.mkv','face.png','output')


def test_single_reference_mode_is_unchanged_and_cannot_accept_hidden_body(tmp_path):
    inp,out,src,face,body=files(tmp_path)
    run=d.create_run(source=src,reference=face,input_root=inp,output_root=out,run_name='single')
    conf=c.read_json(run/'config.json');assert 'body_reference' not in conf
    g=d.make_job(c.read_json(d.RECIPE),conf,c.plan_segments(124)[0],'x.mkv','face.png','output')
    assert 'body_reference' not in g
    assert 'ref_images.ref_image_1' not in g['reference_conditioning']['inputs']
    with pytest.raises(c.ChainError,match='single-reference run'):
        d.make_job(c.read_json(d.RECIPE),conf,c.plan_segments(124)[0],'x.mkv','face.png','output',body_reference_file='body.png')


def test_saved_workflows_are_paired_and_do_not_replace_single_reference():
    root=Path(__file__).resolve().parents[1]
    for name,fn in [('04_Analyze_Face_Body_Build_Prompts',graphs.face_body_studio),
                    ('09_H3_LowMemory_Face_Body_Chain',graphs.face_body_chain)]:
        canvas,api=fn();assert graphs.validate_graph(canvas);assert c.validate_api_graph(api)
        assert json.loads((root/'workflows'/(name+'.json')).read_text())==canvas
        assert json.loads((root/'workflows'/(name+'.api.json')).read_text())==api
    assert (root/'workflows/06_H3_LowMemory_Chain_Control.json').exists()
    assert (root/'workflows/00_Analyze_Clip_Build_Prompts.json').exists()


def test_new_nodes_are_registered_and_required_by_smoke():
    assert 'H3SwapDiskChainFaceBody' in dn.NODE_CLASS_MAPPINGS
    assert 'H3SwapAnalyzeFaceBody' in an.NODE_CLASS_MAPPINGS
    root=Path(__file__).resolve().parents[1]
    schema={}
    for file in (root/'workflows').glob('*.json'):
        if not file.name.endswith('.api.json'):
            for kind in check.graph_nodes(json.loads(file.read_text())):
                schema[kind]={'input':{'required':{}}}
    for n in c.read_json(d.RECIPE).values(): schema[n['class_type']]={'input':{'required':{}}}
    for kind in ('H3SwapDiskSave','H3SwapDiskHistory','PreviewAny'):schema[kind]={'input':{'required':{}}}
    assert check.inspect_schema(schema,{'akatz','bfs'},False)['ok']
    del schema['H3SwapDiskChainFaceBody']
    assert not check.inspect_schema(schema,{'akatz','bfs'},False)['ok']


def test_control_defaults_and_status_ignore_new_input_dropdowns(monkeypatch):
    monkeypatch.setattr(dn,'choices',lambda ext:['example'])
    fields=dn.H3SwapDiskChainFaceBody.INPUT_TYPES()['required']
    assert 'face_image' in fields and 'body_image' in fields and 'character_image' not in fields
    assert fields['additional_instructions'][1]['default']==''
    assert fields['review_mode'][1]['default']=='Review every chunk'
    assert dn.H3SwapDiskChainFaceBody.VALIDATE_INPUTS('Status','missing','missing','missing') is True
    af=an.H3SwapAnalyzeFaceBody.INPUT_TYPES()['required']
    assert list(af)[:3]==['video','face','body']
    assert 'reference_layout' not in af and 'wardrobe' not in af
    assert af['max_output_tokens'][1]['default']==16384


@pytest.mark.parametrize('which',['face','body'])
def test_resume_rejects_changed_original_reference_before_gpu_queue(tmp_path,which):
    run,conf=run_config(tmp_path)
    file=Path(conf['reference' if which=='face' else 'body_reference'])
    file.write_bytes(file.read_bytes()+b'changed')
    client=FakeClient(run)
    with pytest.raises(c.ChainError,match='reference changed'):
        d.execute(run,client)
    assert client.jobs==[]


def test_both_refs_survive_real_media_multichunk_48_seconds(tmp_path):
    inp,out,src,face=fixture_media(tmp_path,48)
    body=inp/'body.png';Image.new('RGB',(40,80),'brown').save(body)
    run=d.create_run(source=src,reference=face,body_reference=body,input_root=inp,output_root=out,
        run_name='dual48',duration=48,review='Auto check + retry')
    client=FakeClient(run)
    result=d.execute(run,client)
    assert result['status']=='complete'
    assert c.verify_video(run/'final.mp4')['frames']==1152
    assert len(client.jobs)==10
    assert client.frees>=11
    for g in client.jobs:
        body_file=inp/g['body_reference']['inputs']['image']
        face_file=inp/g['reference']['inputs']['image']
        assert c.sha256(body_file)==c.sha256(body)
        assert c.sha256(face_file)==c.sha256(face)
        assert g['reference_conditioning']['inputs']['ref_images.ref_image_1']==['body_reference',0]
        assert g['reference_conditioning']['inputs']['prompt'].count(f.MARKER)==1


def test_reference_checks_on_resume_preserve_accepted_master(tmp_path):
    inp,out,src,face=fixture_media(tmp_path,10)
    body=inp/'body.png';Image.new('RGB',(40,80),'brown').save(body)
    run=d.create_run(source=src,reference=face,body_reference=body,input_root=inp,output_root=out,
        run_name='dual_review',duration=10,review='Review every chunk')
    client=FakeClient(run);state=d.execute(run,client)
    assert state['status']=='needs_review'
    master=Path(state['candidate']['folder'])/'master.mkv';before=c.sha256(master)
    state['candidate']['manual_approval']=True;c.atomic_json(run/'state.json',state)
    state=d.execute(run,client)
    assert len(state['chunks'])==1 and state['status']=='needs_review'
    assert c.sha256(master)==before
    assert 'checkpoint' in client.jobs[-1]
    body.write_bytes(body.read_bytes()+b'changed')
    jobs=len(client.jobs)
    with pytest.raises(c.ChainError,match='Body reference changed'):d.execute(run,client)
    assert len(client.jobs)==jobs and c.sha256(master)==before


def test_dual_node_executes_two_reference_analysis_and_invalidates_each_image(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    torch=pytest.importorskip('torch')
    mm=ModuleType('comfy.model_management')
    mm.throw_exception_if_processing_interrupted=lambda:None
    mm.unload_all_models=Mock();mm.soft_empty_cache=Mock()
    comfy=ModuleType('comfy');comfy.__path__=[];comfy.model_management=mm
    monkeypatch.setitem(sys.modules,'comfy',comfy)
    monkeypatch.setitem(sys.modules,'comfy.model_management',mm)
    monkeypatch.setattr(an,'data_root',lambda:tmp_path)
    source=tmp_path/'source.mp4';source.write_bytes(b'fixture')
    video=SimpleNamespace(get_stream_source=lambda:str(source),get_active_trim_window=lambda:(0.,0.))
    monkeypatch.setattr(a,'current_model_digest',lambda name:'digest')
    calls=[]
    def query(images,times,target,settings,*args):
        calls.append([im.size for im in images])
        assert settings.reference_count==2 and settings.reference_layout==opt.DUAL_LAYOUT
        return evidence(),{'requested':settings.record(),'eval_count':6000,'done_reason':'stop'}
    monkeypatch.setattr(a,'query',query)
    monkeypatch.setattr(an.media,'sample_frames',lambda *args,sample_count=16,**kwargs:
        ([Image.new('RGB',(40,50))]*sample_count,list(range(sample_count)),{'audio_analyzed':False}))
    face=torch.zeros(1,64,32,3);body=torch.ones(1,96,32,3)*.5
    def analyze(fa=face,bo=body,**kwargs):
        return an.H3SwapAnalyzeFaceBody().analyze(video,fa,bo,'person',0.,124,**kwargs)
    first=analyze();cached=analyze()
    assert len(calls)==1 and calls[0][:2]==[(32,64),(32,96)]
    assert len(calls[0])==18
    report=json.loads(cached[2]);assert report['cached_draft'] is True
    assert report['reference_roles']['Picture 2']=='physique/proportions/outfit'
    assert 'prompt_drafts_face_body' in report['files']['chain_prompt']
    assert Path(report['files']['chain_prompt']).read_text().strip()==first[1]
    assert first[1]==f.compose_prompt('person',first[1])
    analyze(bo=body+.1);assert len(calls)==2
    analyze(fa=face+.1);assert len(calls)==3
    analyze(reuse_cached_draft=False);assert len(calls)==4
    analyze(bo=body.transpose(1,2));assert len(calls)==5
    bad=body.clone();bad[0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='invalid pixels'):analyze(bo=bad)


def test_new_canvas_widgets_follow_registered_node_order(monkeypatch):
    monkeypatch.setattr(dn,'choices',lambda ext:['placeholder'])
    for fn,kind,cls,sockets in [(graphs.face_body_studio,'H3SwapAnalyzeFaceBody',an.H3SwapAnalyzeFaceBody,{'video','face','body'}),
                               (graphs.face_body_chain,'H3SwapDiskChainFaceBody',dn.H3SwapDiskChainFaceBody,set())]:
        canvas,api=fn();node=next(n for n in canvas['nodes'] if n['type']==kind)
        names=[k for k in cls.INPUT_TYPES()['required'] if k not in sockets]
        assert list(node['widgets_values_named'])==names
        assert node['widgets_values']==[api[str(node['id'])]['inputs'][k] for k in names]


def test_staged_body_tampering_stops_next_queue(tmp_path):
    inp,out,src,face=fixture_media(tmp_path,6)
    body=inp/'body.png';Image.new('RGB',(40,80),'brown').save(body)
    run=d.create_run(source=src,reference=face,body_reference=body,input_root=inp,output_root=out,
        run_name='dual_tamper',duration=6,review='Auto check + retry')
    class Tamper(FakeClient):
        def queue(self,graph):
            ident=super().queue(graph)
            path=inp/graph['body_reference']['inputs']['image']
            path.write_bytes(path.read_bytes()+b'changed during execution')
            return ident
    client=Tamper(run)
    with pytest.raises(c.ChainError,match='Staged body reference changed'):d.execute(run,client)
    assert len(client.jobs)==1
    state=c.read_json(run/'state.json')
    assert len(state['chunks'])==1 and state['status']=='error'


def test_full_body_prompt_is_not_silently_truncated():
    details='Visible body and garment construction. '*800
    text=f.compose_prompt('person',details)
    assert details.strip() in text and len(text)>20000
    with pytest.raises(ValueError,match='never silently truncated'):
        f.compose_prompt('person','x'*opt.MAX_PROMPT_CHARS)


def test_cancel_two_ref_query_still_unloads_model(monkeypatch):
    fake=FakeOllama();monkeypatch.setattr(a.prompts,'session',lambda:fake)
    def cancel():raise InterruptedError('cancel fixture')
    with pytest.raises(InterruptedError):
        a.query([Image.new('RGB',(2,2))]*18,list(range(16)),'person',options(),cancel)
    assert fake.calls[-1][0].endswith('/api/generate')
