"""Offline MOSS integration contracts; no model weights or GPU inference.

Unit template fixtures are explicit. The final test uses the real official
snapshot in full repository CI, after scripts/build_site.py downloads it.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from workbench import moss_catalog as c, moss_workflows as b, moss_support as s

ROOT = Path(__file__).resolve().parents[1]


def addon(tmp_path):
    (tmp_path/'catalog').mkdir(exist_ok=True)
    (tmp_path/'catalog/tasks.moss.json').write_bytes((ROOT/'catalog/tasks.moss.json').read_bytes())
    return tmp_path


def fixture_raw():
    # Minimal runtime contract fixture, not an official upstream snapshot.
    return (ROOT/'tests/fixtures/moss_stock_runtime_contract.json').read_bytes()


def prepare_fixture(tmp_path, monkeypatch):
    addon(tmp_path)
    raw = fixture_raw()
    # Test only: the production blob is immutable and has no runtime override.
    monkeypatch.setattr(b, 'TEMPLATE_BLOB', b.git_blob(raw))
    path = tmp_path/'vendor/moss-stock'/(b.TEMPLATE_REVISION+'.json')
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    return raw, path


@pytest.mark.parametrize('name,value', [('models.json',{'models':[{'id':'qwen'}]}),
    ('tasks.json',{'tasks':[{'id':'existing'}]}), ('assets.json',{'assets':[]})])
def test_absent_addon_leaves_fixture_catalog_untouched(tmp_path,name,value):
    before=copy.deepcopy(value)
    assert c.apply(name,value,tmp_path)==before


def test_moss_shares_workspace_but_not_model_family(tmp_path):
    tasks={'tasks':[{'id':'old','workspaces':['qwen']} ]}
    c.apply('tasks.json',tasks,addon(tmp_path))
    assert tasks['tasks'][0]=={'id':'old','workspaces':['qwen']}
    assert [t['id'] for t in tasks['tasks'][1:]]==['QMOSS','QMOSS_I2I']
    task=tasks['tasks'][1]
    assert task['id']=='QMOSS' and task['family']=='moss' and task['workspaces']==['qwen']
    assert task['requires']==['moss_image'] and len(task['variants'])==1
    with pytest.raises(ValueError,match='Duplicate'):c.apply('tasks.json',tasks,tmp_path)


def test_only_qwen_core_pin_changes(tmp_path):
    q={'id':'qwen','comfy_commit':c.PREVIOUS_COMFY_COMMIT,'default_tasks':['SQ'],
       'default_groups':['qwen','ollama'],'local_nodes':['qwen_people'],'toolbox':['sam']}
    k={'id':'krea','comfy_commit':'a'*40,'default_tasks':['K01']}
    original=copy.deepcopy([q,k]);data={'models':[q,k]}
    c.apply('models.json',data,addon(tmp_path))
    assert q['comfy_commit']==c.COMFY_COMMIT
    for key in ('default_tasks','default_groups','local_nodes'):assert q[key]==original[0][key]
    assert q['toolbox']==['sam','moss_image'] and k==original[1]
    c.apply('models.json',data,tmp_path)
    assert q['toolbox'].count('moss_image')==1


def test_unknown_core_version_is_not_silently_downgraded(tmp_path):
    with pytest.raises(ValueError,match='ComfyUI pin'):
        c.apply('models.json',{'models':[{'id':'qwen','comfy_commit':'9'*40}]},addon(tmp_path))


@pytest.mark.parametrize('mutation', ['workspace','dependency','path','second_task'])
def test_invalid_addon_rejected(tmp_path,mutation):
    addon(tmp_path);path=tmp_path/'catalog/tasks.moss.json';data=json.loads(path.read_text());t=data['tasks'][0]
    if mutation=='workspace':t['workspaces']=['krea']
    elif mutation=='dependency':t['requires']=['qwen']
    elif mutation=='path':t['variants'][0]['file']='../../evil'
    else:data['tasks'].append(copy.deepcopy(t))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):c.apply('tasks.json',{'tasks':[]},tmp_path)


def test_three_exact_models_no_qwen_replacements(tmp_path):
    old={'id':'qwen-original','destination':'diffusion_models/qwen.safetensors','family':'qwen'}
    data={'assets':[copy.deepcopy(old)]}
    c.apply('assets.json',data,addon(tmp_path))
    assert data['assets'][0]==old and len(data['assets'])==4
    assert all(a['family']=='moss' and a['group']=='moss_image' for a in data['assets'][1:])
    assert sum(a['estimated_bytes'] for a in c.asset_rows())==13006811386
    assert all(len(a['sha256'])==64 and len(a['revision'])==40 for a in c.asset_rows())


@pytest.mark.parametrize('field',['id','destination'])
def test_collision_stops_catalog(tmp_path,field):
    row={'id':'other','destination':'other'};row[field]=c.asset_rows()[0][field]
    with pytest.raises(ValueError):c.apply('assets.json',{'assets':[row]},addon(tmp_path))


@pytest.mark.parametrize('idx',[0,1])
def test_exact_default_pair_are_only_quantization_exceptions(idx):
    row=c.asset_rows()[idx]
    assert c.is_stock_quantized_asset(row)
    for key in ('id','family','group','repo_id','revision','filename','destination','sha256'):
        assert not c.is_stock_quantized_asset({**row,key:'different'})
    assert not c.is_stock_quantized_asset(c.asset_rows()[2])
    assert not c.is_stock_quantized_asset({'family':'qwen','kind':'diffusion_models','precision':'int8'})


@pytest.mark.parametrize('task,value,result', [('QMOSS','moss_image_sft_int8_convrot.safetensors',True),
    ('QMOSS','qwen3.5_9b_w4a8_gptq.safetensors',True),('QMOSS','untrusted_int8.safetensors',False),
    ('QNATIVE','moss_image_sft_int8_convrot.safetensors',False),('QKEEP1',None,False)])
def test_static_precision_exception_is_task_and_filename_scoped(task,value,result):
    assert c.is_stock_model_selection(task,value) is result


def test_corrupt_upstream_bytes_rejected():
    with pytest.raises(ValueError,match='checksum'):b.verify_template(fixture_raw())


def test_offline_missing_source_stops(tmp_path):
    addon(tmp_path)
    with pytest.raises(FileNotFoundError,match='snapshot'):b.build(tmp_path,offline=True)
    assert not (tmp_path/c.WORKFLOW).exists()


def test_source_snapshot_is_copied_without_edits(tmp_path,monkeypatch):
    raw,path=prepare_fixture(tmp_path,monkeypatch)
    report=b.build(tmp_path,offline=True)
    assert path.read_bytes()==raw==(tmp_path/c.WORKFLOW).read_bytes()
    assert report['copied_byte_for_byte'] and report['gpu_tested'] is False
    b.build(tmp_path,offline=True)
    assert (tmp_path/c.WORKFLOW).read_bytes()==raw


def test_corrupt_cached_source_never_overwrites_output(tmp_path,monkeypatch):
    raw,path=prepare_fixture(tmp_path,monkeypatch)
    b.build(tmp_path,offline=True);path.write_bytes(raw+b'bad')
    with pytest.raises(ValueError,match='checksum'):b.build(tmp_path)
    assert (tmp_path/c.WORKFLOW).read_bytes()==raw


@pytest.mark.parametrize('key',list(b.DEFAULTS))
def test_default_model_sampler_upsampler_values_checked(tmp_path,monkeypatch,key):
    data=json.loads(fixture_raw());data['nodes'][0]['widgets_values_named'][key]='changed'
    raw=json.dumps(data).encode();monkeypatch.setattr(b,'TEMPLATE_BLOB',b.git_blob(raw))
    with pytest.raises(ValueError,match='default'):b.verify_template(raw)


def test_no_addon_has_no_network_or_side_effects(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'_download',lambda:pytest.fail('unexpected network'))
    assert b.build(tmp_path)['installed'] is False
    assert not (tmp_path/'vendor').exists()


def test_http_path_calls_verification_before_writing(tmp_path,monkeypatch):
    addon(tmp_path);raw=fixture_raw();monkeypatch.setattr(b,'TEMPLATE_BLOB',b.git_blob(raw))
    class Response:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def geturl(self):return b.TEMPLATE_URL
        def read(self,size):return raw
    monkeypatch.setattr(b,'urlopen',lambda *a,**k:Response())
    b.build(tmp_path)
    assert (tmp_path/c.WORKFLOW).read_bytes()==raw


def test_unexpected_download_redirect_rejected(monkeypatch):
    class Response:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def geturl(self):return 'https://untrusted.invalid/workflow.json'
    monkeypatch.setattr(b,'urlopen',lambda *a,**k:Response())
    with pytest.raises(ValueError,match='redirect'):b._download()


def core_fixture(tmp_path):
    (tmp_path/'requirements.txt').write_text('\n'.join(s.REQUIRED_REQUIREMENTS)+'\n')
    paths=('comfy/ldm/moss_image/model.py','comfy/text_encoders/moss_image.py',
           'comfy_extras/nodes_moss_image.py','comfy_extras/nodes_qwen.py','comfy_extras/nodes_sam3.py')
    for name in paths:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
    (tmp_path/'nodes.py').write_text('moss_image\n')
    (tmp_path/'comfy/cli_args.py').write_text('--disable-comfy-compiler\n')


def test_new_qwen_core_dependency_checks_are_nonmutating(tmp_path):
    core_fixture(tmp_path);before=(tmp_path/'requirements.txt').read_bytes()
    s.verify_core(tmp_path,c.COMFY_COMMIT)
    assert (tmp_path/'requirements.txt').read_bytes()==before


@pytest.mark.parametrize('case',['pin','requirements','node','flag'])
def test_incompatible_core_stops_build(tmp_path,case):
    core_fixture(tmp_path);commit=c.COMFY_COMMIT
    if case=='pin':commit='0'*40
    if case=='requirements':(tmp_path/'requirements.txt').write_text('comfy-aimdo==0.5.5\n')
    if case=='node':(tmp_path/'comfy_extras/nodes_qwen.py').unlink()
    if case=='flag':(tmp_path/'comfy/cli_args.py').write_text('other\n')
    with pytest.raises(ValueError):s.verify_core(tmp_path,commit)


def test_real_official_template_and_runtime_catalog_in_full_repo(monkeypatch):
    if not (ROOT/'catalog/models.json').is_file():
        pytest.skip('Full repository CI downloads the actual pinned MOSS source; isolated bundle tests use explicit fixtures.')
    from workbench import config, assets
    raw=(ROOT/c.WORKFLOW).read_bytes()
    source=ROOT/'vendor/moss-stock'/(b.TEMPLATE_REVISION+'.json')
    assert raw==source.read_bytes()
    assert b.git_blob(raw)==b.TEMPLATE_BLOB
    b.verify_template(raw)
    from workbench.moss_i2i_workflows import make_img2img, encode_graph
    assert (ROOT/c.I2I_WORKFLOW).read_bytes()==encode_graph(make_img2img(json.loads(raw)))
    monkeypatch.setenv('WB_WORKSPACE','qwen');monkeypatch.setenv('WB_TASKS','QKEEP1,QMOSS,QMOSS_I2I')
    monkeypatch.delenv('WB_TOOLBOX',raising=False)
    assert 'moss_image' in config.requested_groups()
    assert config.model_config()['comfy_commit']==c.COMFY_COMMIT
    for row in c.asset_rows():
        assert assets._standard_asset_high_precision(row) is (not c.is_stock_quantized_asset(row))
    for family in ('flux','krea','h3','restoration'):
        monkeypatch.setenv('WB_WORKSPACE',family);monkeypatch.setenv('WB_TASKS','QMOSS')
        with pytest.raises(ValueError):config.selected_tasks()
