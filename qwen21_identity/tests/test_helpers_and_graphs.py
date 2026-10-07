import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest
import torch
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from validate_workflows import check_graph
spec=importlib.util.spec_from_file_location('q21helpers',ROOT/'custom_nodes/qwen21_identity_tools/__init__.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

@pytest.fixture
def scene():return torch.zeros(1,64,96,3)
@pytest.fixture
def mask():
    m=torch.zeros(1,64,96);m[:,16:48,24:72]=1;return m

def test_all_ui_api_graphs():
    paths=list((ROOT/'workflows').glob('*.json'));assert len(paths)==7
    for p in paths:check_graph(json.loads(p.read_text()),json.loads((ROOT/'api_workflows'/(p.stem+'.api.json')).read_text()))

def test_mask_correct(scene,mask):
    a,b=q.Q21MaskGuard().check(scene,mask,'head');assert a is scene and b is mask

def test_empty_mask_fails(scene):
    with pytest.raises(ValueError,match='no mask painted'):q.Q21MaskGuard().check(scene,torch.zeros(1,64,96),'body')

def test_mask_dimensions_fail(scene):
    with pytest.raises(ValueError,match='match the full scene'):q.Q21MaskGuard().check(scene,torch.ones(1,64,64),'head')

def test_bad_mask_values_fail(scene,mask):
    mask[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='invalid mask'):q.Q21MaskGuard().check(scene,mask,'head')

def test_combined_rgb_check(scene,mask):
    q.Q21MaskGuard().check(scene,mask,'head',scene.clone(),scene)
    with pytest.raises(ValueError,match='matching RGB'):q.Q21MaskGuard().check(scene,mask,'head',scene+0.1,scene)

def test_alpha_compositing(scene):
    image=torch.ones(1,64,96,4);image[...,3]=.25
    assert torch.allclose(q.Q21OpaqueImage().flatten(image,scene)[0],torch.full_like(scene,.25))

def test_alpha_size_failure(scene):
    with pytest.raises(ValueError,match='does not match'):q.Q21OpaqueImage().flatten(torch.ones(1,32,32,4),scene)

def test_protected_audit_pass(scene,mask):
    edit=scene.clone();edit[mask>0]=.5
    result,report=q.Q21AuditPreservation().audit(scene,edit,mask)
    assert result is edit and report.startswith('PASS:')

def test_protected_audit_detects_change(scene,mask):
    edit=scene.clone();edit[0,0,0,0]=.01
    with pytest.raises(ValueError,match='outside the actual blend'):q.Q21AuditPreservation().audit(scene,edit,mask)

def test_union(mask):
    other=mask.roll(10,1)
    assert torch.equal(q.Q21MaskUnion().combine(mask,other)[0],torch.maximum(mask,other))

@pytest.mark.parametrize('w,h,mp',[(4000,6000,.59),(1024,1024,.59),(6000,4000,1.0),(640,480,1.0)])
def test_size_alignment(w,h,mp):
    nw,nh=q.budget_dimensions(w,h,mp,False)
    assert nw%32==nh%32==0
    assert abs(nw/nh-w/h)<.07
    assert nw*nh<=max(mp*1e6*1.08,w*h)

def test_native_encoder_delegation(monkeypatch,scene):
    capture={};p=types.ModuleType('comfy_extras');m=types.ModuleType('comfy_extras.nodes_qwen')
    class Native:
        @staticmethod
        def execute(**kwargs):capture.update(kwargs);return types.SimpleNamespace(result=('positive','negative',{'samples':'native'}))
    m.TextEncodeQwenImage21=Native
    monkeypatch.setitem(sys.modules,'comfy_extras',p);monkeypatch.setitem(sys.modules,'comfy_extras.nodes_qwen',m)
    result=q.Q21EncodeReferences().encode('clip','vae',scene,'replace subject',scene)
    assert result[2]['samples']=='native' and capture['resolution']==0
    assert list(capture['images'])==['image_1','image_2'] and capture['negative_prompt']==''
    with pytest.raises(ValueError,match='image_2'):q.Q21EncodeReferences().encode('clip','vae',scene,'x',image_3=scene)

def test_atomic_handoff(tmp_path,monkeypatch,scene):
    f=types.ModuleType('folder_paths');f.get_input_directory=lambda:str(tmp_path)
    monkeypatch.setitem(sys.modules,'folder_paths',f)
    result=q.Q21WriteHandoff().save(scene+.5,'body')
    p=tmp_path/'qwen21_handoff/body_latest.png'
    assert p.exists() and Image.open(p).size==(96,64) and result[0].shape==scene.shape
    assert not list(p.parent.glob('.handoff-*'))
    with pytest.raises(ValueError):q.Q21WriteHandoff().save(scene,'../../bad')

def test_reject_batch():
    with pytest.raises(ValueError,match='exactly one'):q.require_image(torch.zeros(2,64,64,3))
