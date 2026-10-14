"""Offline release contracts. No test here certifies GPU image quality."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from workbench import identity_catalog as catalog
from workbench.private_compat import krea_metadata, normalize_entry
from workbench.graph_support import required_node_types
from workbench import runtime_recovery as recovery
import build_identity_workflows as builder


def graph_fixture():
    # Exact IDs, port ordering and connections from the pinned Krea developer JSON.
    specs = [
        (29,'SaveImage',[('images','IMAGE')],[],['output']),
        (53,'KSampler',[('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT')],['LATENT'],[1,'fixed',10,1,'euler','simple',1]),
        (54,'VAEDecode',[('samples','LATENT'),('vae','VAE')],['IMAGE'],[]),
        (55,'UNETLoader',[],['MODEL'],['krea2_turbo_fp8_scaled.safetensors','default']),
        (56,'CLIPLoader',[],['CLIP'],['qwen3vl_4b_fp8_scaled.safetensors','krea2','default']),
        (57,'VAELoader',[],['VAE'],['qwen_image_vae.safetensors']),
        (71,'LoraLoaderModelOnly',[('model','MODEL')],['MODEL'],['Krea2/krea2_identity_edit_v1_2.safetensors',1]),
        (72,'LoadImage',[],['IMAGE','MASK'],['example.png','image']),
        (73,'VAEEncode',[('pixels','IMAGE'),('vae','VAE')],['LATENT'],[]),
        (79,'Krea2EditModelPatch',[('model','MODEL'),('source_latent','LATENT'),('source_latent_b','LATENT'),('ref_boost_mask','MASK'),('vae','VAE'),('source_image','IMAGE'),('source_image_b','IMAGE'),('target_latent','LATENT')],['MODEL'],[4,1,'fit']),
        (82,'EmptySD3LatentImage',[('width','INT'),('height','INT')],['LATENT'],[1024,1024,1]),
        (83,'ResolutionSelector',[],['INT','INT'],['1:1 (Square)',1,8]),
        (84,'Krea2EditGroundedEncode',[('clip','CLIP'),('image','IMAGE'),('image_b','IMAGE')],['CONDITIONING'],['Edit',768,'']),
        (85,'Krea2EditGroundedEncode',[('clip','CLIP'),('image','IMAGE'),('image_b','IMAGE')],['CONDITIONING'],['',768,'']),
        (90,'LoadImage',[],['IMAGE','MASK'],['example_person.png','image']),
        (92,'VAEEncode',[('pixels','IMAGE'),('vae','VAE')],['LATENT'],[]),
    ]
    graph = {'version':0.4,'nodes':[],'links':[],'last_node_id':110,'last_link_id':27}
    for ident,kind,inputs,outputs,widgets in specs:
        graph['nodes'].append({'id':ident,'type':kind,'pos':[ident*10,0],'size':[300,200],
            'inputs':[{'name':n,'type':t,'link':None} for n,t in inputs],
            'outputs':[{'name':t,'type':t,'links':[]} for t in outputs],
            'widgets_values':widgets,'mode':4 if ident in (90,92) else 0, 'flags':{},'properties':{}})
    nodes = {n['id']:n for n in graph['nodes']}
    connections = [(55,0,71,0),(71,0,79,0),(79,0,53,0),(56,0,84,0),(56,0,85,0),
        (57,0,54,1),(57,0,73,1),(57,0,92,1),(83,0,82,0),(83,1,82,1),(72,0,73,0),
        (72,0,84,1),(72,0,85,1),(72,0,79,5),(73,0,79,1),(57,0,79,4),(90,0,92,0),
        (90,0,84,2),(90,0,85,2),(90,0,79,6),(92,0,79,2),(84,0,53,1),(85,0,53,2),
        (82,0,53,3),(53,0,54,0),(54,0,29,0),(82,0,79,7)]
    for lid,(src,out,dst,inp) in enumerate(connections,1):
        graph['links'].append([lid,src,out,dst,inp,nodes[src]['outputs'][out]['type']])
        nodes[src]['outputs'][out]['links'].append(lid)
        nodes[dst]['inputs'][inp]['link']=lid
    return graph


@pytest.mark.parametrize('raw,two',[(False,False),(False,True),(True,False),(True,True)])
def test_krea_full_topology_and_optional_lora_bypass(raw,two):
    original = graph_fixture()
    graph = builder.adapter.adapt_krea(original,raw,two)
    assert original == graph_fixture()
    graph = builder.add_optional_krea_loras(graph)
    builder.validate_topology(graph)
    slots = [n for n in graph['nodes'] if n.get('title','').startswith('Optional LoRA')]
    assert len(slots)==2
    assert all(n['mode']==4 and n['widgets_values'][1]==0 for n in slots)
    by_id={n['id']:n for n in graph['nodes']}
    assert by_id[71]['widgets_values'][1]==1
    assert by_id[53]['widgets_values'][2:4]==([20,3] if raw else [10,1])
    assert by_id[90]['mode']==(0 if two else 4)
    assert by_id[85]['widgets_values'][0]==''
    assert 'fp8_scaled' not in json.dumps(graph)
    assert {'Krea2EditModelPatch','Krea2EditGroundedEncode'} <= required_node_types(graph)


def test_topology_rejects_bad_ports():
    graph=graph_fixture();graph['nodes'][0]['inputs'][0]['link']=999
    with pytest.raises(ValueError):builder.validate_topology(graph)


def test_catalog_preserves_existing_tasks_and_defaults():
    tasks={'tasks':[{'id':'K01','variants':[{'file':'existing.json'}]}]}
    result=catalog.apply('tasks.json',copy.deepcopy(tasks))
    assert result['tasks'][0]==tasks['tasks'][0]
    assert [t['id'] for t in result['tasks']]==['K01','KDEV','QDEV']
    assert all(len(t['variants'])==4 for t in result['tasks'][1:])
    assert all('ollama' not in t['requires'] for t in result['tasks'][1:])
    models={'models':[{'id':'krea','default_tasks':['K01'],'default_groups':['krea','ollama']}]}
    assert catalog.apply('models.json',copy.deepcopy(models))==models


def test_asset_pins_and_no_duplicate_identity_adapter():
    value={'assets':[{'id':'krea-krea2_identity_edit_v1_2','destination':'loras/krea2_identity_edit_v1_2.safetensors','revision':'main'}]}
    result=catalog.apply('assets.json',value)
    assert len(result['assets'])==4
    assert all(len(a['revision'])==40 for a in result['assets'])
    assert all(len(a['sha256'])==64 for a in result['assets'][1:])
    with pytest.raises(ValueError):catalog.apply('assets.json',result)


@pytest.mark.parametrize('name',['Krea2','Krea 2','KREA_2','Krea-2-Raw','Krea 2 Turbo','krea/Krea-2-Raw'])
def test_krea_metadata_names(name):
    assert krea_metadata(name)
    row=normalize_entry({'url':'https://civitai.red/models/1','kind':'diffusion_models'}, {'reported_base_model':name})
    assert row['family']=='krea' and row['architecture']=='krea2'


def test_checkpoint_kind_routed_to_diffusion_models():
    assert normalize_entry({'kind':'checkpoints','family':'krea'}, {'reported_base_model':'Krea2'})['kind']=='diffusion_models'


@pytest.mark.parametrize('metadata',['SDXL 1.0','Flux.1 D','Qwen Image 2.1','NotKrea2'])
def test_incompatible_declared_krea_rejected(metadata):
    assert not krea_metadata(metadata)
    with pytest.raises(ValueError):normalize_entry({'kind':'diffusion_models','family':'krea','architecture':'krea2'}, {'reported_base_model':metadata})


def test_unknown_krea_metadata_requires_declaration():
    with pytest.raises(ValueError):normalize_entry({'kind':'diffusion_models','family':'krea'}, {})
    row=normalize_entry({'kind':'diffusion_models','family':'krea','architecture':'krea2'}, {})
    assert row['architecture']=='krea2'


def test_other_family_rules_not_relaxed():
    entry={'family':'qwen','kind':'diffusion_models'}
    assert normalize_entry(entry,{'reported_base_model':'Qwen Image 2.1'})==entry
    with pytest.raises(ValueError):normalize_entry(entry,{'reported_base_model':'Krea2'})


def test_nested_subgraphs_not_reported_as_missing_plugins():
    graph={'nodes':[{'type':'sub-uuid'}], 'definitions':{'subgraphs':[{'id':'sub-uuid','nodes':[{'type':'UNETLoader'},{'type':'Note'}]}]}}
    assert required_node_types(graph)=={'UNETLoader'}


@pytest.mark.parametrize('family,expected',[('qwen',['--disable-comfy-compiler']),('krea',['--disable-comfy-compiler']),('flux',[]),('h3',[])])
def test_stability_flag_defaults(tmp_path,monkeypatch,family,expected):
    monkeypatch.delenv('WB_DISABLE_COMFY_COMPILER',raising=False)
    (tmp_path/'comfy').mkdir();(tmp_path/'comfy/cli_args.py').write_text('--disable-comfy-compiler')
    assert recovery.launch_options(tmp_path,family)==expected


def test_stability_flag_explicit_opt_out_and_invalid(tmp_path,monkeypatch):
    monkeypatch.setenv('WB_DISABLE_COMFY_COMPILER','0')
    assert recovery.launch_options(tmp_path,'krea')==[]
    monkeypatch.setenv('WB_DISABLE_COMFY_COMPILER','yes')
    with pytest.raises(ValueError):recovery.launch_options(tmp_path,'krea')


def test_guard_refuses_second_process(monkeypatch):
    runtime=Mock();runtime.comfy.poll.return_value=None
    with pytest.raises(ValueError,match='already running'):recovery.guard_single_instance(runtime)
    runtime.comfy=None;conn=Mock()
    monkeypatch.setattr(recovery.socket,'create_connection',lambda *a,**k:conn)
    with pytest.raises(ValueError,match='occupied'):recovery.guard_single_instance(runtime)
    conn.close.assert_called_once()


def test_guard_allows_crashed_child_on_free_port(monkeypatch):
    runtime=Mock();runtime.comfy.poll.return_value=-11
    def refuse(*a,**k):raise ConnectionRefusedError()
    monkeypatch.setattr(recovery.socket,'create_connection',refuse)
    recovery.guard_single_instance(runtime)


class RuntimeFixture:
    def __init__(self,running=True,queue=None):
        self.job=None;self.comfy=Mock();self.comfy.poll.return_value=None if running else -11
        self.started=0
        data=queue if queue is not None else {'queue_running':[],'queue_pending':[]}
        class Response:
            status=200
            async def __aenter__(self):return self
            async def __aexit__(self,*a):pass
            async def json(self):return data
        self.session=Mock();self.session.get.return_value=Response()
    def start_comfy(self):self.started+=1


def test_restart_crashed_process():
    runtime=RuntimeFixture(running=False)
    response=asyncio.run(recovery.restart(runtime,None))
    assert response.status==202 and runtime.started==1


def test_restart_refuses_busy_queue():
    runtime=RuntimeFixture(queue={'queue_running':[1],'queue_pending':[]})
    response=asyncio.run(recovery.restart(runtime,None))
    assert response.status==409 and runtime.started==0
    runtime.comfy.terminate.assert_not_called()


def test_restart_idle_process():
    runtime=RuntimeFixture();proc=runtime.comfy
    response=asyncio.run(recovery.restart(runtime,None))
    assert response.status==202 and runtime.started==1
    proc.terminate.assert_called_once();proc.wait.assert_called_once()


def test_restart_during_model_download_rejected():
    runtime=RuntimeFixture();runtime.job=Mock();runtime.job.done.return_value=False
    assert asyncio.run(recovery.restart(runtime,None)).status==409
    assert runtime.started==0


def test_red_checkpoint_download_and_registration_contract(tmp_path,monkeypatch):
    # Full existing-repository integration runs in CI; unavailable in isolated bundle tests.
    assets=pytest.importorskip('workbench.assets')
    monkeypatch.setenv('WB_WORKSPACE','krea');monkeypatch.setenv('WB_DATA_ROOT',str(tmp_path))
    monkeypatch.setenv('WB_CHECKPOINTS','https://civitai.red/api/download/models/123?fileId=456')
    monkeypatch.setenv('WB_LORA_URLS','https://civitai.red/api/download/models/234?fileId=567')
    monkeypatch.delenv('WB_CHECKPOINT_FILE',raising=False);monkeypatch.delenv('WB_LORA_FILE',raising=False)
    calls=[]
    def resolve(url,transport,allowed_types=None):
        calls.append((url,allowed_types))
        return {'reported_base_model':'Krea 2','sha256':'a'*64,'download_url':url}
    monkeypatch.setattr(assets.pd,'resolve_civitai',resolve)
    def transfer(row,target,transport,**kw):
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'fixture')
        return {'tensors':1,'dtypes':['BF16'],'bytes':7}
    monkeypatch.setattr(assets,'transfer',transfer)
    monkeypatch.setattr(assets.pd,'check_adapter_file',lambda p:None)
    report=assets.prepare_private()
    assert not report['failures'] and len(report['assets'])==2
    assert {r['kind'] for r in report['assets']}=={'loras','diffusion_models'}
    assert all(r['family']=='krea' for r in report['assets'])
    assert calls[0][1] is None and 'checkpoint' in calls[1][1]


def test_generated_developer_files_in_full_build():
    if not (ROOT/'catalog/models.json').exists():
        pytest.skip('Full repository build is performed in GitHub Actions, not in isolated bundle tests.')
    for task,rows in catalog.FILES.items():
        for _,name in rows:
            path=ROOT/catalog.workflow_path(task,name)
            assert path.is_file(), 'Run scripts/build_identity_workflows.py first.'
            graph=json.loads(path.read_text());builder.validate_topology(graph)
            assert 'WBQwenPromptControl' not in required_node_types(graph)
