"""Economy contracts and real local file access. No GPU/render-quality claims."""
import asyncio
import copy
import json
import os
from pathlib import Path
import shutil
import threading
import uuid
import pytest
from aiohttp import BasicAuth
from aiohttp.test_utils import TestClient, TestServer
from swapkit import precision as p, economy_batch as b, economy_graphs as eg
from swapkit import files_builtin as fb, job_history as jh
from swapkit import recovery
from swapkit.graphs import validate_graph, adapt_akatz
from swapkit.independent_graph import independent_native, join_graph
from swapkit.disk_common import validate_api_graph

ROOT=Path(__file__).resolve().parents[1]


def seed(root):
    (root/'workflows').mkdir(parents=True,exist_ok=True)
    (root/'swapkit/data').mkdir(parents=True,exist_ok=True)
    for name in ('assets.spec.json',):shutil.copy2(ROOT/name,root/name)
    src=json.loads((ROOT/'tests/fixtures/native_canvas_for_preview.json').read_text())
    api=json.loads((ROOT/'swapkit/data/akatz_disk_base.api.json').read_text())
    shutil.copy2(ROOT/'swapkit/data/akatz_disk_base.api.json',root/'swapkit/data/akatz_disk_base.api.json')
    ca,ga,_=adapt_akatz(src,api)
    from swapkit.preview_graph import native_preview
    for name,pair in [('02B_Akatz_Native_Simple_Preview',native_preview(ca,ga)),
                      ('12_Akatz_Independent_Chunk_Preview',independent_native(ca,ga)),
                      ('13_H3_Join_Rendered_Chunks',join_graph())]:
        for suffix,value in [('.json',pair[0]),('.api.json',pair[1])]:
            (root/'workflows'/(name+suffix)).write_text(json.dumps(value))
    return root


def test_economy_original_bf16_source_hash_and_no_bfs(tmp_path):
    r=seed(tmp_path/'image');p.prepare_build(p.ECONOMY,r)
    rows=json.loads((r/'assets.spec.json').read_text())['assets']
    assert not any(x['group']=='bfs' for x in rows)
    diffusion=next(x for x in rows if x['id']=='h3-ref2va-bf16-pruned')
    assert diffusion['file']=='diffusion_models/'+p.PRUNED
    assert diffusion['revision']=='3792b245ac4685dda00ebf43a33c646f7ce98064'
    assert diffusion['expected_sha256']=='37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe'
    assert next(x for x in rows if x['id']=='h3-encoder-bf16')['file'].endswith(p.ENCODER)
    before=(r/'assets.spec.json').read_bytes();p.prepare_build(p.ECONOMY,r)
    assert before==(r/'assets.spec.json').read_bytes()
    assert p.folder_suffix(r)=='_ECONOMY' and '--bf16-unet' in p.cli_flags(r)


def test_economy_transform_does_not_change_sampler_or_mask(tmp_path):
    r=seed(tmp_path/'image');p.prepare_build(p.ECONOMY,r)
    original=json.loads((r/'swapkit/data/akatz_disk_base.api.json').read_text())
    p.finalize_build(r);eg.build_economy(r)
    worker=json.loads((r/'swapkit/data/akatz_disk_base.api.json').read_text())
    p.validate_recipe(worker,diffusion=p.PRUNED)
    for key in original:
        if key not in ('model','text','attention'):assert worker[key]==original[key]
    assert worker['model']['inputs']['unet_name']==p.PRUNED
    assert worker['text']['inputs']['clip_name']==p.ENCODER
    for f in (r/'workflows').glob('*.json'):
        data=json.loads(f.read_text())
        if f.name.endswith('.api.json'):validate_api_graph(data)
        else:validate_graph(data)
    assert not (r/'workflows/02C_Akatz_BF16_HighRef_Preview.json').exists()


def test_economy_image_no_ollama_or_filebrowser_daemon():
    text=(ROOT/'Dockerfile.economy').read_text()
    assert 'github.com/ollama' not in text and 'ollama-linux' not in text
    assert 'linux-amd64-filebrowser' not in text
    assert 'H3_BUILD_PROFILE=bf16_economy' in text
    assert 'swapkit.preview_assets --download' in text
    assert 'H3_SWAP_PROMPT_ASSIST=0' in text


def test_gateway_is_bound_before_configure_and_gpu_preflight():
    text=(ROOT/'swapkit/runtime.py').read_text().split('async def main_async():',1)[1]
    assert text.index('web.TCPSite')<text.index('env=await asyncio.to_thread(configure)')<text.index('--gpu-smoke')
    assert "os.environ['H3_SWAP_SAVE_METADATA']='1'" in text
    assert "os.environ['H3_SWAP_PROMPT_ASSIST']='0'" in text


def test_each_chunk_preserves_exact_canvas_and_only_inputs_change(tmp_path):
    r=seed(tmp_path/'image')
    c=json.loads((r/'workflows/12_Akatz_Independent_Chunk_Preview.json').read_text())
    a=json.loads((r/'workflows/12_Akatz_Independent_Chunk_Preview.api.json').read_text())
    conf={'run_name':'t','resolution':'768','target_person':'person in blue','selection':'leftmost',
          'selection_index':0,'mask_expansion':48,'seed':19,'window_frames':124,'prompt':'simple',
          'reference_sha256':'a'*64,'model_profile':'baseline'}
    ca,api=b.render_job(c,a,conf,'input/h3_independent/t/clip_0002.mkv','same.png',2,0)
    validate_graph(ca);validate_api_graph(api)
    for n in ca['nodes']:
        if 'widgets_values_named' in n:assert n['widgets_values']==list(n['widgets_values_named'].values())
    assert api['source']['inputs']['source_clip'].endswith('clip_0002.mkv')
    assert api['plan']['inputs']['seed']==19
    assert api['reference']['inputs']['image']=='same.png'
    assert api['reference_conditioning']['inputs']['prompt']=='simple'
    assert not any(n['class_type']=='StartLoop' for n in api.values())
    assert 'previous' not in api['window']['inputs'] and 'initial_history' not in api['window']['inputs']
    for key in ('model','text','sampler','schedule','adapter','turbo','guide','history','edit_mask'):
        assert api[key]==a[key]
    _,retry=b.render_job(c,a,conf,'input/h3_independent/t/clip_0002.mkv','same.png',2,1)
    assert retry['plan']['inputs']['seed']!=19 and retry['reference_conditioning']==api['reference_conditioning']


def test_auto_submission_supplies_actual_frontend_workflow_container():
    c=b.Client();calls=[]
    c.request=lambda path,payload: calls.append((path,payload)) or {'prompt_id':'123','node_errors':{}}
    workflow={'nodes':[{'id':1}],'version':.4};graph={'1':{'class_type':'Whatever','inputs':{}}}
    assert c.submit(graph,workflow)=='123'
    assert calls[0][1]['extra_data']['extra_pnginfo']['workflow'] is workflow
    assert calls[0][1]['prompt'] is graph


@pytest.mark.parametrize('path',['../etc/passwd','/etc/passwd','a/../../etc/passwd','.env','a/.secret','bad\\path'])
def test_builtin_paths_reject_escape_and_hidden(tmp_path,path):
    with pytest.raises(Exception):fb.local(tmp_path,path,must_exist=False)


def test_builtin_no_symlink(tmp_path):
    (tmp_path/'link').symlink_to('/tmp',target_is_directory=True)
    with pytest.raises(Exception):fb.local(tmp_path,'link/foo',must_exist=False)


def test_builtin_ready_before_models_real_http(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_DATA',str(tmp_path));monkeypatch.setenv('H3_SWAP_FILES_BACKEND','builtin')
    monkeypatch.setenv('H3_SWAP_FILE_BROWSER','1')
    (tmp_path/'input').mkdir();(tmp_path/'secret.html').write_text('<script>bad()</script>')
    async def run():
        from swapkit.gateway import make_app
        status={'ready':False,'stage':'models still downloading'}
        app=make_app('0123456789abcdef0123456789',status)
        async with TestClient(TestServer(app)) as client:
            assert (await client.get('/files/')).status==401
            h={'Authorization':BasicAuth('workbench','0123456789abcdef0123456789').encode()}
            landing=await client.get('/',headers=h)
            assert landing.status==200 and 'File access is available now' in await landing.text()
            assert (await client.get('/files/',headers=h)).status==200
            assert status['filebrowser']['ready'] is True and status['ready'] is False
            assert (await client.get('/files/api/list',headers=h)).status==200
            response=await client.get('/files/api/download?path=secret.html',headers=h)
            assert response.status==200 and 'attachment;' in response.headers['Content-Disposition']
            assert response.headers['Content-Type']=='application/octet-stream'
            uh={**h,'X-H3-Files':'1'}
            response=await client.post('/files/api/upload?path=input&name=hello.txt',headers=uh,data=b'hello')
            assert response.status==200
            assert (tmp_path/'input/hello.txt').read_bytes()==b'hello'
            response=await client.post('/files/api/upload?path=input&name=hello.txt',headers=uh,data=b'overwrite')
            assert response.status==409 and (tmp_path/'input/hello.txt').read_bytes()==b'hello'
            assert not list((tmp_path/'input').glob('.upload-*'))
            assert (await client.post('/files/api/upload?path=input&name=x',headers=h,data=b'x')).status==403
            bad={**uh,'Origin':'https://attacker.example','Sec-Fetch-Site':'cross-site'}
            assert (await client.post('/files/api/upload?path=input&name=x',headers=bad,data=b'x')).status==403
    asyncio.run(run())


def record():
    ident=str(uuid.uuid4());canvas={'nodes':[{'id':1,'type':'H3SwapWindowPreview'}],'links':[],'version':.4}
    prompt={'1':{'class_type':'H3SwapWindowPreview','inputs':{'hold_seconds':0},'is_changed':[float('nan')]}}
    return ident,{'prompt':[0,ident,prompt,{'extra_pnginfo':{'workflow':canvas,'secret':'NO'},'create_time':123,'api_key':'NO'},['1']],
        'status':{'completed':True,'status_str':'success','messages':[['execution_success',{'timestamp':234,'prompt_id':ident}]]},
        'outputs':{'1':{'images':[{'filename':'test.png','subfolder':'','type':'output'}]}},'meta':{}}


class Queue:
    def __init__(self):self.mutex=threading.RLock();self.history={};self.currently_running={};self.calls=0
    def task_done(self,item_id,history_result,status,process_item=None):
        self.calls+=1;p=self.currently_running.pop(item_id)
        self.history[p[1]]={'prompt':p,'outputs':history_result['outputs'],'status':status}
    def get_history(self,prompt_id=None):return {prompt_id:self.history[prompt_id]} if prompt_id in self.history else {}


def test_persist_restore_is_actual_job_record_and_does_not_mutate(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA','1')
    ident,r=record();orig=r['prompt'][2]['1']['is_changed'][0]
    assert jh.persist(ident,r,tmp_path)
    saved=json.loads((tmp_path/'h3-swap/job-history'/(ident+'.json')).read_text())
    assert 'is_changed' not in saved['prompt'][2]['1']
    assert orig!=orig and r['prompt'][2]['1']['is_changed'][0]!=r['prompt'][2]['1']['is_changed'][0]
    assert 'NO' not in json.dumps(saved)
    q=Queue();assert jh.restore(q,tmp_path)==1 and not q.currently_running
    assert q.history[ident]['prompt'][3]['extra_pnginfo']['workflow']==r['prompt'][3]['extra_pnginfo']['workflow']
    assert q.history[ident]['outputs']==r['outputs']


def test_queue_completion_hook_preserves_execution_and_records_history(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA','1')
    ident,r=record();q=Queue();jh.attach(q,tmp_path);jh.attach(q,tmp_path)
    q.currently_running[8]=r['prompt'];q.task_done(8,{'outputs':r['outputs']},r['status'])
    assert q.calls==1 and not q.currently_running
    assert (tmp_path/'h3-swap/job-history'/(ident+'.json')).exists()
    assert 'is_changed' in q.history[ident]['prompt'][2]['1']


def test_restore_pre_restart_backup_never_invents_missing_canvas(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA','1');ident,r=record()
    backup=tmp_path/'h3-swap/history-backups';backup.mkdir(parents=True)
    bad_id,bad=record();bad['prompt'][3]={}
    (backup/'comfy-history-one.json').write_text(json.dumps({ident:r,bad_id:bad}))
    q=Queue();assert jh.restore(q,tmp_path)==1 and ident in q.history and bad_id not in q.history


def test_real_nonfinite_workflow_input_is_not_replaced(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA','1');ident,r=record()
    r['prompt'][2]['1']['inputs']['hold_seconds']=float('nan')
    with pytest.raises(ValueError):jh.persist(ident,r,tmp_path)


def test_current_patch_has_metadata_fix_and_both_existing_precision_paths():
    assert "node.pop('is_changed', None)" in (ROOT/'swapkit/recovery.py').read_text()
    assert "BF16 = 'bf16_full'" in (ROOT/'swapkit/precision.py').read_text()
    assert p.cli_flags(ROOT)==[]  # source checkout has not been compiled as an image


def test_nonterminal_and_corrupt_history_are_not_restored(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA','1');ident,r=record()
    r['status']={'completed':False,'status_str':'running'}
    with pytest.raises(ValueError):jh.clean_record(ident,r)
    with pytest.raises(ValueError):jh.clean_record(ident,None)
    backup=tmp_path/'h3-swap/history-backups';backup.mkdir(parents=True)
    (backup/'comfy-history-bad.json').write_text(json.dumps({ident:r,str(uuid.uuid4()):None}))
    assert jh.restore(Queue(),tmp_path)==0


def test_real_10s_economy_review_resume_join_and_job_workflows(tmp_path,monkeypatch):
    """Actual media I/O; generation is passthrough, NOT H3 inference."""
    from PIL import Image
    from test_independent_chunks import source,storage,rgb
    from swapkit import independent_media as m
    import numpy as np
    (tmp_path/'data').mkdir();root=storage(tmp_path/'data')
    image=seed(tmp_path/'image')
    monkeypatch.setattr(b,'ROOT',image)
    monkeypatch.setenv('H3_SWAP_DATA',str(root));monkeypatch.setenv('H3_SWAP_SAVE_METADATA','1')
    src,original_frames=source(root,n=240)
    Image.new('RGB',(32,32),(80,90,100)).save(root/'input/character.png')
    run=b.create('input/source.mkv','input/character.png','auto_test',duration_seconds=10)
    class FakeClient:
        def __init__(self):self.jobs={};self.submitted=[];self.frees=0;self.current=None
        def wait_idle(self):pass
        def clear_memory(self):self.frees+=1
        def cancel_owned(self):self.current=None
        def request(self,path,*args):
            return {path.split('/')[-1]:self.jobs[path.split('/')[-1]]}
        def wait(self,ident):self.current=None;return self.jobs[ident]
        def submit(self,graph,workflow):
            validate_graph(workflow);validate_api_graph(graph)
            self.submitted.append((copy.deepcopy(graph),copy.deepcopy(workflow)))
            ident=str(uuid.uuid4());metadata={'workflow':workflow,'prompt':graph}
            if 'save' in graph:
                clip=m.local_media(graph['source']['inputs']['source_clip'])
                target=root/'output'/('fake_'+ident);target.mkdir()
                master=target/'master.mkv';shutil.copy2(clip,master)
                m.make_mp4(master,target/'preview.mp4')
                rec={'format':'h3-independent-render-v1','model_profile':'baseline',
                     'master_sha256':b.file_sha(master),'frames':m.video_info(master)['frames'],
                     'reference_sha256':{'reference':b.file_sha(root/'input'/graph['reference']['inputs']['image'])},
                     'source_clip':json.loads(clip.with_suffix('.source.json').read_text())}
                (target/'receipt.json').write_text(json.dumps(rec))
                out={'save':{'text':[master.relative_to(root).as_posix()]}}
            else:
                v=graph['1']['inputs']
                report=m.join_videos(v['ordered_videos'],v['join_name'],v['overlap_frames'],v['cut_placement'],
                    v['audio_mode'],v['original_source'],v['source_start_seconds'],v['crf'],metadata,root)
                out={'1':{'text':[json.dumps(report)]}}
            self.current=ident
            self.jobs[ident]={'outputs':out,'status':{'completed':True,'status_str':'success'}}
            return ident
    client=FakeClient();first=b.execute(run,client)
    assert first['status']=='needs_review' and len(client.submitted)==1
    assert not first['accepted'];assert first['total_chunks']==3
    saved=json.loads((run/'state.json').read_text());saved['candidate']['approved']=True
    prefix=saved['candidate']['sha256'];b.atomic_json(run/'state.json',saved)
    done=b.execute(run,client)
    assert done['status']=='complete' and len(done['accepted'])==3 and len(client.submitted)==4
    assert done['accepted'][0]['sha256']==prefix and client.frees==0
    for graph,canvas in client.submitted[:-1]:
        assert graph['plan']['inputs']['history_frames']=='0'
        assert graph['plan']['inputs']['seed']==904239
        assert graph['reference']['inputs']['image'].endswith('/character.png')
        assert not any(n['class_type']=='StartLoop' for n in graph.values())
        assert 'previous' not in graph['window']['inputs']
        assert canvas['extra']['h3_batch']['mode']=='independent'
    final=Path(done['final']);info=m.video_info(final)
    assert info['frames']==240 and info['audio'] is not None
    assert json.loads(info['format']['tags']['workflow'])==client.submitted[-1][1]
    np.testing.assert_array_equal(rgb(final.with_name('final.mkv')),original_frames)
    assert b.execute(run,client)['status']=='complete' and len(client.submitted)==4


def test_old_archived_job_resolves_for_asset_right_click_without_requeue(tmp_path,monkeypatch):
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA','1')
    ident,r=record();jh.persist(ident,r,tmp_path)
    q=Queue();jh.attach(q,tmp_path)
    q.history.clear()  # Simulate an older job evicted from Comfy's recent list.
    result=q.get_history(prompt_id=ident)
    assert result[ident]['prompt'][3]['extra_pnginfo']['workflow']==r['prompt'][3]['extra_pnginfo']['workflow']
    assert not q.history and not q.currently_running and q.calls==0
    assert q.get_history(prompt_id='../../etc/passwd')=={}
