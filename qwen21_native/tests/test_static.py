"""GitHub host tests. This file and its imports never import PyTorch."""
import json,sys,hashlib
from pathlib import Path
import pytest,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from validate_workflows import check_graph,selected_catalog
from build_workflows import build_all
from install_into_comfy import install

CAT=json.loads((ROOT/'config/workflow_catalog.json').read_text())['workflows']

@pytest.mark.parametrize('row',CAT,ids=[r['file'] for r in CAT])
def test_each_graph_pair(row):
    p=ROOT/'workflows'/row['file'];api=json.loads((ROOT/'api_workflows'/(p.stem+'.api.json')).read_text())
    check_graph(json.loads(p.read_text()),api)
    if row['requires']!='bfs_optional':assert not any(n['class_type']=='LoraLoaderModelOnly' for n in api.values())

@pytest.mark.parametrize('profile,bfs,count',[('identity',False,34),('upscale',False,36),('identity',True,36),('upscale',True,38)])
def test_profile_workflow_counts(profile,bfs,count):
    assert len(selected_catalog(profile,bfs))==count

def test_generation_deterministic():
    paths=list((ROOT/'workflows').glob('*.json'))+list((ROOT/'api_workflows').glob('*.json'))+[ROOT/'config/workflow_catalog.json']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};build_all()
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

@pytest.mark.parametrize('name',['90_AB_Head_Native_vs_BFS','91_AB_Body_Native_vs_BFS'])
def test_controlled_lora_comparison(name):
    api=json.loads((ROOT/'api_workflows'/(name+'.api.json')).read_text())
    by=lambda t:[n['inputs'] for n in api.values() if n['class_type']==t]
    sams=by('KSampler');assert len(sams)==5
    for k in ('seed','steps','cfg','sampler_name','scheduler','denoise'):assert len({s[k] for s in sams})==1
    assert {n['strength_model'] for n in by('LoraLoaderModelOnly')}=={.25,.5,.75,1.0}
    assert len({n['prompt'] for n in by('Q21NEncodeReferences')})==1
    assert len({n['megapixels'] for n in by('Q21NResizeBudget')})==1
    for k in ('target_megapixels','context_factor','blend_pixels'):assert len({n[k] for n in by('AUSBOSS_NODES_CropForInpaint')})==1
    assert not by('Q21NWriteHandoff')

def test_expression_roles():
    api=json.loads((ROOT/'api_workflows/13_Expression_Image2_Only.api.json').read_text())
    prompt=next(n['inputs']['prompt'] for n in api.values() if n['class_type']=='Q21NEncodeReferences')
    assert 'sole identity and appearance anchor' in prompt
    assert '<image2> solely for expression' in prompt

def test_build_separates_torch_tests():
    path=ROOT/'config/ci_workflow.yml';data=yaml.safe_load(path.read_text())
    step=next(s for s in data['jobs']['validate']['steps'] if 'run' in s and 'pytest' in s['run'])
    assert 'test_helpers_and_graphs.py' not in step['run']
    assert 'test_static.py' in step['run']
    docker=(ROOT/'Dockerfile').read_text()
    assert '--system-site-packages' in docker and '/tmp/q21-tests/bin/python -m pytest' in docker
    assert 'build_smoke.sh' not in docker
    assert 'ENABLE_BFS_COMPARISONS=0' in docker
    external=ROOT.parent/'.github/workflows/build-qwen21-native.yml'
    if external.exists():assert external.read_bytes()==path.read_bytes()

def test_installer_preserves_existing_workflows_and_models(tmp_path):
    comfy=tmp_path/'comfy';user=tmp_path/'user'
    qwen=comfy/'comfy_extras/nodes_qwen.py';qwen.parent.mkdir(parents=True);qwen.write_text('class TextEncodeQwenImage21: pass')
    aus=comfy/'custom_nodes/ComfyUI-AusBoss/nodes/node_inpaint_crop_stitch.py';aus.parent.mkdir(parents=True);aus.write_text('AUSBOSS_NODES_StitchInpaint')
    keep=comfy/'models/my-existing-model.safetensors';keep.parent.mkdir();keep.write_bytes(b'not changed')
    old=user/'default/workflows/old_workflow.json';old.parent.mkdir(parents=True);old.write_text('old')
    out=install(comfy,user)
    assert len(list(out.glob('*.json')))==34
    assert not list(out.glob('90_*'))
    first=out/'00_START_HERE_Scene_Face_Body.json';first.write_text('personal edit')
    install(comfy,user)
    assert first.read_text()=='personal edit' and old.read_text()=='old'
    assert keep.read_bytes()==b'not changed'
    assert (comfy/'custom_nodes/qwen21_native_tools/__init__.py').exists()

def test_installer_rejects_missing_core(tmp_path):
    with pytest.raises(RuntimeError,match='lacks native Qwen'):install(tmp_path/'comfy',tmp_path/'user')
