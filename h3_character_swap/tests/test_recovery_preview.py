"""Offline/CPU contract tests. Not visual acceptance or live frontend tests."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from swapkit import recovery as r, recovery_web as rw, preview_graph as pg, preview_assets as pa
from swapkit.graphs import validate_graph
from swapkit.disk_common import ComfyClient

ROOT=Path(__file__).resolve().parents[1]


def config(tmp_path, dual=False):
    inputs=tmp_path/'input';inputs.mkdir(exist_ok=True)
    outputs=tmp_path/'output';outputs.mkdir(exist_ok=True)
    return dict(run_name='old_run',input_root=str(inputs),output_root=str(outputs),
        source=str(inputs/'original video.mp4'),reference=str(inputs/'portrait.png'),
        body_reference=str(inputs/'body.png') if dual else None,
        reference_mode='face_body' if dual else 'single',
        prompt='Exact prompt: accents caf\u00e9; symbols = # \\ and\nnewlines.',
        duration_seconds=48.,start_seconds=2.,resolution='768',window_frames=124,
        target_person='person in blue',selection='leftmost',selection_index=0,
        mask_expansion=32,seed=99999,review_mode='Review every chunk',max_retries=0,
        source_sha256='a'*64,reference_sha256='b'*64,workflow_digest='c'*64)


@pytest.mark.parametrize('dual,resume',[(False,False),(False,True),(True,False),(True,True)])
def test_controller_recovers_exact_settings(tmp_path, dual, resume):
    c=config(tmp_path,dual);before=copy.deepcopy(c)
    canvas,api=r.control_workflow(c,resume=resume)
    assert validate_graph(canvas)
    node=next(n for n in canvas['nodes'] if n['type'].startswith('H3SwapDiskChain'))
    v=node['widgets_values_named']
    assert list(v.values())==node['widgets_values']
    assert v['action']==('Resume' if resume else 'Start new run')
    assert (v['run_name']==c['run_name'])==resume
    assert v['seed']==99999 and v['mask_expansion']==32 and v['duration_seconds']==48
    assert v['source_video']=='original video.mp4'
    assert v['additional_instructions' if dual else 'prompt']==c['prompt']
    assert c==before and api[str(node['id'])]['inputs']==v
    assert ('body_image' in v)==dual
    assert canvas['extra']['h3_recovery']['assets_embedded'] is False


def test_export_is_sidecar_only_and_idempotent(tmp_path):
    c=config(tmp_path);run=tmp_path/'output/h3_character_swap/disk_chain_v2/old_run';run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(c));(run/'state.json').write_text('{"status":"complete"}')
    (run/'final.mp4').write_bytes(b'keep video bytes')
    before={p.name:p.read_bytes() for p in run.iterdir()}
    first=r.export_run(run);second=r.export_run(run)
    assert first==second
    for name,data in before.items():assert (run/name).read_bytes()==data
    assert (run/'repeat.workflow.json').is_file() and (run/'resume.api.json').is_file()


def test_outside_input_not_silently_renamed(tmp_path):
    c=config(tmp_path);c['reference']=str(tmp_path/'elsewhere/portrait.png')
    with pytest.raises(ValueError,match='outside the input'):r.control_workflow(c)


def test_optout(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA','0')
    assert r.media_metadata({'a':1},{'workflow':{'nodes':[]}})=={}
    assert r.export_run(tmp_path)=={}
    data={'prompt':{'1':{'class_type':'H3SwapDiskChain','inputs':{}}}}
    assert r.journal_request(data) is data


def test_metadata_only_workflow_and_prompt_not_arbitrary_extra():
    meta=r.media_metadata({'1':{'class_type':'LoadImage'}},{'workflow':{'nodes':[]},'secret':'do not copy'})
    assert set(meta)=={'workflow','prompt'} and 'secret' not in json.dumps(meta)


def test_journal_observes_without_changing_request_or_dumping_env(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_DATA',str(tmp_path));monkeypatch.setenv('WB_PASSWORD','DO-NOT-EXPORT-ME')
    data={'prompt':{'1':{'class_type':'MiniMaxH3ReferenceToVideo','inputs':{'prompt':'example'}}},
          'extra_data':{'extra_pnginfo':{'workflow':{'nodes':[],'links':[]}},'private_flag':'no'}}
    original=copy.deepcopy(data)
    assert r.journal_request(data) is data and data==original
    files=list((tmp_path/'output/h3_character_swap/workflow_history').rglob('*.json'))
    assert len(files)==3
    assert 'DO-NOT-EXPORT-ME' not in ''.join(p.read_text() for p in files)
    assert next(p for p in files if p.name=='submitted.workflow.json')


def test_non_h3_cleanup_does_not_fill_journal(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_DATA',str(tmp_path))
    r.journal_request({'prompt':{'x':{'class_type':'PreviewAny','inputs':{'source':'cleanup'}}}})
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('path',['../secret.workflow.json','/etc/secret.workflow.json',
 'h3_character_swap/disk_chain_v2/test/config.json','h3_character_swap/other/a.workflow.json'])
def test_recovery_download_rejects_unrelated_files(tmp_path,path):
    with pytest.raises(ValueError):rw.allowed_file(tmp_path,path)


def test_recovery_download_checks_auth_and_traversal(tmp_path,monkeypatch):
    import asyncio,base64
    from aiohttp.test_utils import TestClient,TestServer
    from swapkit.gateway import make_app
    monkeypatch.setenv('H3_SWAP_DATA',str(tmp_path))
    out=tmp_path/'output';f=out/'h3_character_swap/disk_chain_v2/run/repeat.workflow.json'
    f.parent.mkdir(parents=True);f.write_text('{"nodes":[],"links":[]}')
    url=rw.download_url(f,out)
    password='a-very-long-test-password'
    auth='Basic '+base64.b64encode(('workbench:'+password).encode()).decode()
    async def scenario():
        async with TestClient(TestServer(make_app(password,{'ready':True}))) as client:
            resp=await client.get(url);assert resp.status==401
            resp=await client.get(url,headers={'Authorization':auth});assert resp.status==200
            assert await resp.json()=={'nodes':[],'links':[]}
            assert 'attachment' in resp.headers['Content-Disposition']
            resp=await client.get('/_swap/recovery/file?path=../secret.workflow.json',headers={'Authorization':auth})
            assert resp.status==404
    asyncio.run(scenario())


def test_chain_worker_history_has_controller_but_cleanup_does_not():
    client=ComfyClient();client.request=Mock(return_value={'prompt_id':'abc'})
    workflow={'nodes':[]};client.recovery_workflow=workflow
    client.queue({'disk_save':{'class_type':'H3SwapDiskSave','inputs':{}}})
    assert client.request.call_args.args[1]['extra_data']['extra_pnginfo']['workflow']==workflow
    client.queue({'cleanup':{'class_type':'PreviewAny','inputs':{}}})
    assert 'extra_data' not in client.request.call_args.args[1]


def native_fixture():
    return json.loads((ROOT/'tests/fixtures/native_canvas_for_preview.json').read_text()), json.loads((ROOT/'swapkit/data/akatz_disk_base.api.json').read_text())


def test_native_preview_does_not_change_model_noise_or_sampling():
    graph,api=native_fixture();original=copy.deepcopy(api)
    canvas,modified=pg.native_preview(graph,api)
    assert validate_graph(canvas)
    for key in original:
        if key not in {'reference_conditioning','append','save'}:assert modified[key]==original[key]
    a,b=copy.deepcopy(original['reference_conditioning']),copy.deepcopy(modified['reference_conditioning'])
    b['inputs']['prompt']=a['inputs']['prompt'];assert a==b
    assert modified['reference_conditioning']['inputs']['prompt']==r.SIMPLE_PROMPT
    assert modified['append']['inputs']['frames']==['900001',0]
    assert modified['900001']['inputs']['images']==['decode',0]
    assert modified['900001']['inputs']['iteration_index']==['start',0]
    assert modified['save']['class_type']=='H3SwapSaveVideo'
    assert api==original


def test_simple_chain_has_one_image_no_wrapper_no_automatic_rerolls():
    graph,api=pg.simple_chain();assert validate_graph(graph)
    v=next(iter(api.values()))['inputs']
    assert v['prompt']==r.SIMPLE_PROMPT and 'body_image' not in v
    assert v['duration_seconds']==10 and v['window_frames']=='124'
    assert v['review_mode']=='Review every chunk' and v['max_retries']==0


def test_decoded_preview_returns_identical_tensor_and_can_cancel(monkeypatch):
    torch = pytest.importorskip('torch', reason='Optional CPU tensor check; built-image smoke also exercises Torch')
    from swapkit.preview_nodes import H3SwapWindowPreview
    preview=Mock(return_value={'ui':{'images':[{'filename':'test.png','type':'temp','subfolder':''}]}})
    nodes=ModuleType('nodes');nodes.PreviewImage=lambda:SimpleNamespace(save_images=preview)
    comfy=ModuleType('comfy');mm=ModuleType('comfy.model_management');utils=ModuleType('comfy.utils')
    mm.throw_exception_if_processing_interrupted=Mock()
    updates=[]
    class Progress:
        def __init__(self,total):pass
        def update_absolute(self,*args):updates.append(args)
    utils.ProgressBar=Progress;comfy.model_management=mm;comfy.utils=utils
    for name,mod in [('nodes',nodes),('comfy',comfy),('comfy.model_management',mm),('comfy.utils',utils)]:monkeypatch.setitem(sys.modules,name,mod)
    images=torch.rand(7,32,32,3)
    original=images.clone()
    out=H3SwapWindowPreview().inspect(images,hold_seconds=0)
    assert out['result'][0] is images and torch.equal(images,original)
    assert preview.call_args.args[0].shape[0]==1
    assert updates[0][2][0]=='JPEG'
    mm.throw_exception_if_processing_interrupted=Mock(side_effect=[None,InterruptedError('cancel')])
    with pytest.raises(InterruptedError):H3SwapWindowPreview().inspect(images,hold_seconds=10)
    assert torch.equal(images,original)


def test_preview_asset_checks_git_blob_and_size(tmp_path,monkeypatch):
    content=b'four';p=tmp_path/'test.safetensors';p.write_bytes(content)
    monkeypatch.setattr(pa,'SIZE',4);monkeypatch.setattr(pa,'BLOB',hashlib.sha1(b'blob 4\0'+content).hexdigest())
    assert pa.valid(p)
    p.write_bytes(b'bad!');assert not pa.valid(p)


@pytest.mark.parametrize('method',['taesd','latent2rgb','none'])
def test_preview_method_explicit_optout(monkeypatch,method):
    monkeypatch.setenv('H3_SWAP_PREVIEW_METHOD',method);assert pa.method()==method


def test_preview_default_and_bad_override(monkeypatch):
    monkeypatch.delenv('H3_SWAP_PREVIEW_METHOD',raising=False);assert pa.method()=='taesd'
    monkeypatch.setenv('H3_SWAP_PREVIEW_METHOD','execute');
    with pytest.raises(ValueError):pa.method()


def test_build_retains_actual_cpu_checks():
    text=(ROOT.parent/'.github/workflows/build-h3-character-swap.yml').read_text()
    assert '-m swapkit.check --smoke' in text and '-m swapkit.preview_assets --smoke' in text
    docker=(ROOT/'Dockerfile').read_text()
    assert 'swapkit.filebrowser --smoke' in docker and 'preview_assets --download' in docker
    runtime=(ROOT/'swapkit/runtime.py').read_text()
    assert 'if not boolean("H3_SWAP_SAVE_METADATA", True)' in runtime
    assert 'args += ["--preview-method", env["H3_SWAP_PREVIEW_METHOD"]]' in runtime


def test_ffmpeg_embeds_exact_controller_json_real_media(tmp_path):
    """Real FFmpeg encode and ffprobe extraction, not a mocked metadata check."""
    from swapkit.disk_chain import assemble
    from swapkit.disk_storage import write_master
    c=config(tmp_path);workflow,api=r.control_workflow(c)
    # Include every FFmetadata metacharacter plus a long prompt and Unicode.
    node=next(n for n in workflow['nodes'] if n['type']=='H3SwapDiskChain')
    node['widgets_values_named']['prompt'] += 'x' * 50000
    metadata={'workflow':workflow,'prompt':api}
    part=tmp_path/'part';part.mkdir();logs=tmp_path/'logs';logs.mkdir()
    frames=np.random.default_rng(4).random((6,32,32,3),dtype=np.float32)
    write_master(frames,part/'master.mkv')
    target=tmp_path/'final.mp4'
    assemble([{'folder':str(part),'new_frames':6}],part/'master.mkv',target,6,logs,None,metadata=metadata)
    result=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format_tags','-of','json',str(target)]))
    tags=result['format']['tags']
    assert json.loads(tags['workflow'])==workflow
    assert json.loads(tags['prompt'])==api
    assert json.loads(Path(str(target)+'.workflow.json').read_text())==workflow
    before=hashlib.sha256(target.read_bytes()).hexdigest()
    assert before==hashlib.sha256(target.read_bytes()).hexdigest()


def test_actual_reasoning_counter_is_not_just_answer_count(monkeypatch):
    from swapkit import akatz_analysis as a, analysis_options as opt, prompts
    from PIL import Image
    class Reply:
        def __init__(self,obj=None,rows=None):self.obj=obj;self.rows=rows or []
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def raise_for_status(self):pass
        def json(self):return self.obj
        def iter_lines(self,**kwargs):
            for row in self.rows:yield json.dumps(row).encode()
    evidence={key:'visible supported detail' for key in a.FIELDS}
    class Session:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def post(self,url,**kwargs):
            if url.endswith('/api/show'):return Reply({'capabilities':['vision']})
            if url.endswith('/api/chat'):return Reply(rows=[{'message':{'thinking':'not stored reasoning'}},
                {'message':{'content':json.dumps(evidence)}}, {'done':True,'done_reason':'stop','eval_count':40}])
            return Reply({})
        def get(self,url,**kwargs):return Reply({'models':[]})
    monkeypatch.setattr(prompts,'session',Session)
    _,stats=a.query([Image.new('RGB',(32,32))]*17,list(range(16)),'person',opt.AnalysisOptions())
    assert stats['reasoning_characters']==len('not stored reasoning') and stats['stream_messages']==3
    assert 'not stored reasoning' not in json.dumps(stats)


def test_export_refuses_changed_config(tmp_path):
    c=config(tmp_path);run=tmp_path/'run';run.mkdir();(run/'config.json').write_text(json.dumps(c))
    r.export_run(run);c['seed']=123;(run/'config.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='configuration changed'):r.export_run(run)


def test_video_saver_passes_metadata_and_writes_sidecars(tmp_path,monkeypatch):
    from swapkit.nodes import H3SwapSaveVideo
    folder_paths=ModuleType('folder_paths')
    folder_paths.get_output_directory=lambda:str(tmp_path)
    folder_paths.get_save_image_path=lambda *a:(str(tmp_path),'native',1,'','native')
    latest=ModuleType('comfy_api.latest');latest.Types=SimpleNamespace(VideoContainer=lambda x:x,VideoCodec=lambda x:x)
    latest.io=SimpleNamespace(FolderType=SimpleNamespace(output='output'))
    class Preview:
        def __init__(self,items):self.items=items
        def as_dict(self):return {'images':self.items}
    latest.ui=SimpleNamespace(PreviewVideo=Preview,SavedResult=lambda *a:a)
    pkg=ModuleType('comfy_api');pkg.latest=latest
    comfy=ModuleType('comfy');cli=ModuleType('comfy.cli_args');cli.args=SimpleNamespace(disable_metadata=False)
    for name,mod in [('folder_paths',folder_paths),('comfy_api',pkg),('comfy_api.latest',latest),
                     ('comfy',comfy),('comfy.cli_args',cli)]:monkeypatch.setitem(sys.modules,name,mod)
    class Video:
        def get_dimensions(self):return 32,32
        def save_to(self,path,**kwargs):self.path=Path(path);self.options=kwargs;self.path.write_bytes(b'media fixture')
    video=Video();prompt={'s':{'class_type':'SaveVideo','inputs':{}}};workflow={'nodes':[],'links':[]}
    H3SwapSaveVideo().save(video,'test/native',16,prompt,{'workflow':workflow})
    assert video.options['metadata']=={'prompt':prompt,'workflow':workflow}
    assert json.loads(Path(str(video.path)+'.workflow.json').read_text())==workflow
    assert video.options['codec']=='h264' and video.options['crf']==16
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA','0')
    H3SwapSaveVideo().save(video,'test/native',16,prompt,{'workflow':workflow})
    assert video.options['metadata'] is None and not Path(str(video.path)+'.workflow.json').exists()


def test_journal_metadata_failure_does_not_abort_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv('H3_SWAP_DATA', str(tmp_path))
    data={'prompt':{'h3':{'class_type':'H3SwapDiskChain','inputs':{}}},
          'extra_data':{'extra_pnginfo':{'workflow':{'nodes':[]}}}}
    monkeypatch.setattr(r,'MAX_METADATA_BYTES',5)
    assert r.journal_request(data) is data
    assert not list(tmp_path.rglob('submitted.*'))
