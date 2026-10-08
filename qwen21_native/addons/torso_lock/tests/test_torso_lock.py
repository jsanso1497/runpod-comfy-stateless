import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

q=load_module('torso_lock_test_module',ROOT/'custom_nodes/qwen21_torso_tools/__init__.py')
installer=load_module('torso_lock_installer_test',ROOT/'INSTALL.py')
generator=load_module('torso_lock_generator_test',ROOT/'scripts/build_workflows.py')

@pytest.fixture
def data():
    torch.manual_seed(17)
    scene=torch.rand(1,128,96,3)
    edit=torch.zeros(1,128,96);edit[:,16:94,15:82]=1
    protect=torch.zeros_like(edit);protect[:,80:117,23:73]=1
    return scene,edit,protect

def prepared(data,feather=8,padding=0):
    scene,edit,protect=data
    return q.Q21TPrepareMasks().prepare(scene,edit,scene,protect,'first_pass_same_scene',feather,padding)

def finalize(data,p=None,candidate=None):
    scene,_,_=data
    p=p or prepared(data)
    return q.Q21TFinalize().finish(p[0],scene,candidate if candidate is not None else torch.ones_like(scene),p[2],p[3],p[5])

def test_context_keeps_original_navel_region_visible(data):
    _,edit,protect=data;p=prepared(data)
    assert torch.equal(p[1],((edit>0)|(protect>0)).float())
    assert torch.equal(p[0],data[0])

def test_never_feathers_outside_painted_edit(data):
    _,edit,_=data;p=prepared(data)
    assert (p[2][edit==0]==0).all()
    assert p[2][0,16,30]==0 and p[2][0,40,40]==1
    assert 0<p[2][0,19,30]<1

def test_original_navel_protection_overrides_fullframe_corruption(data):
    scene,edit,protect=data;out=finalize(data)[0]
    assert torch.equal(out[protect>0],scene[protect>0])
    assert torch.equal(out[edit==0],scene[edit==0])
    assert out[0,40,40,0]==1

def test_overlap_preserves_original_not_candidate(data):
    scene,edit,protect=data;p=prepared(data);out=finalize(data,p)[0]
    assert torch.equal(out[(edit>0)&(protect>0)],scene[(edit>0)&(protect>0)])
    assert (p[2][(edit>0)&(protect>0)]==0).all()
    assert json.loads(p[5])['edit_protection_overlap_pixels']>0

def test_any_positive_protection_value_locks(data):
    scene,edit,protect=data;protect=protect*.001
    p=prepared((scene,edit,protect));out=finalize((scene,edit,protect),p)[0]
    assert torch.equal(out[protect>0],scene[protect>0])

def test_soft_edit_mask_never_strengthened(data):
    scene,edit,protect=data;edit=edit*.25;p=prepared((scene,edit,protect))
    assert p[2].max()<=.25

def test_zero_feather_is_raw_mask_except_protection(data):
    _,edit,protect=data;p=prepared(data,0)
    assert torch.equal(p[2],torch.where(protect>0,torch.zeros_like(edit),edit))

def test_protection_padding_locks_more_pixels(data):
    p=prepared(data,padding=3)
    assert p[3].sum()>data[2].sum()
    assert (p[2][p[3]>0]==0).all()

def test_already_locked_base_checked(data):
    scene,_,_=data;p=list(prepared(data));p[0]=p[0].clone();p[0][p[3]>0]=.5
    with pytest.raises(ValueError,match='Locked base'):
        finalize(data,p)

def test_refinement_restores_original_before_and_after(data):
    scene,edit,protect=data;accepted=scene*.75
    p=q.Q21TPrepareMasks().prepare(accepted,edit,scene,protect,'refine_accepted_candidate',8,0)
    result=q.Q21TFinalize().finish(p[0],scene,torch.zeros_like(scene),p[2],p[3],p[5])
    out=result[0]
    assert torch.equal(out[protect>0],scene[protect>0])
    assert torch.equal(out[(edit==0)&(protect==0)],accepted[(edit==0)&(protect==0)])

def test_first_pass_rejects_different_rgb(data):
    scene,edit,protect=data
    with pytest.raises(ValueError,match='SAME original'):
        q.Q21TPrepareMasks().prepare(scene*.5,edit,scene,protect,'first_pass_same_scene')

@pytest.mark.parametrize('which',['edit','protect'])
def test_empty_masks_rejected(data,which):
    scene,edit,protect=data
    if which=='edit':edit=torch.zeros_like(edit)
    else:protect=torch.zeros_like(protect)
    with pytest.raises(ValueError,match='empty'):
        q.Q21TPrepareMasks().prepare(scene,edit,scene,protect,'first_pass_same_scene')

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.0,1.1])
def test_invalid_masks_rejected(data,value):
    scene,edit,protect=data;edit[0,1,1]=value
    with pytest.raises(ValueError,match='values'):
        prepared(data)

def test_wrong_dimensions_rejected(data):
    scene,edit,protect=data
    with pytest.raises(ValueError,match='dimensions'):
        q.Q21TPrepareMasks().prepare(scene,edit,scene[:,:96],protect,'first_pass_same_scene')

def test_wrong_mask_dimensions_rejected(data):
    scene,edit,protect=data
    with pytest.raises(ValueError,match='full-scene'):
        q.Q21TPrepareMasks().prepare(scene,edit[:,:64],scene,protect,'first_pass_same_scene')

def test_full_protection_rejected(data):
    scene,edit,protect=data
    with pytest.raises(ValueError,match='whole image'):
        q.Q21TPrepareMasks().prepare(scene,edit,scene,torch.ones_like(protect),'first_pass_same_scene')

def test_all_edit_pixels_locked_rejected(data):
    scene,edit,_=data
    with pytest.raises(ValueError,match='No editable'):
        q.Q21TPrepareMasks().prepare(scene,edit,scene,edit,'first_pass_same_scene')

def test_unaligned_candidate_rejected(data):
    with pytest.raises(ValueError,match='aligned images'):
        finalize(data,candidate=torch.zeros(1,64,64,3))

def test_inpaint_alpha_flattening(data):
    scene,_,_=data;candidate=torch.ones(1,128,96,4);candidate[...,3]=.5
    result=q.Q21TOpaqueCrop().flatten(candidate,scene)[0]
    assert torch.allclose(result,.5+.5*scene)

def test_opaque_crop_must_match_canvas(data):
    with pytest.raises(ValueError,match='crop size'):
        q.Q21TOpaqueCrop().flatten(torch.zeros(1,64,64,3),data[0])

def test_final_reports_original_dimensions_and_pass(data):
    out=finalize(data);report=json.loads(out[3])
    assert report['status']=='PASS' and report['width']==96 and report['height']==128
    assert report['protected_original_max_error']==report['outside_edit_max_error']==0
    assert out[2].shape==data[0].shape and out[1].shape==(1,830,1280,3)

@pytest.mark.parametrize('secondary,third,garment',list(itertools.product([False,True],repeat=3)))
def test_optional_reference_combinations_are_numbered_correctly(secondary,third,garment):
    image=torch.zeros(1,64,64,3)
    refs,prompt,roles=q.Q21TTorsoReferences().plan(image,image,'replace_crop_top','blue sports bra','',1.0,
        image if secondary else None,image if third else None,image if garment else None)
    count=2+secondary+third+garment
    assert list(refs)==[f'image_{i}' for i in range(1,count+1)]
    assert {int(n) for n in re.findall(r'<image(\d+)>',prompt)}<=set(range(1,count+1))
    if garment:assert f'<image{count}> = GARMENT DESIGN ONLY' in roles
    else:assert 'blue sports bra' in prompt
    assert 'Do not move, reshape or duplicate the navel' in prompt
    assert 'SUPPORTING TORSO ANGLE' in roles if secondary or third else 'SUPPORTING TORSO ANGLE' not in roles

def test_refinement_never_redesigns_bra_or_uses_garment():
    image=torch.zeros(1,64,64,3)
    refs,prompt,roles=q.Q21TTorsoReferences().plan(image,image,'repair_skin_seam','','',1,garment=image)
    assert len(refs)==2 and 'GARMENT DESIGN ONLY' not in roles
    assert 'Keep the accepted sports bra' in prompt

def test_unknown_manual_image_reference_rejected():
    with pytest.raises(ValueError,match='unconnected'):
        q.make_prompt('replace_crop_top','bra','Use <image8>',[(1,'BASE'),(2,'TORSO')])

def test_missing_garment_description_rejected():
    image=torch.zeros(1,64,64,3)
    with pytest.raises(ValueError,match='garment photo'):
        q.Q21TTorsoReferences().plan(image,image,'replace_crop_top','','',1)

def test_native_encoder_delegates_exact_inputs(monkeypatch):
    seen={};parent=types.ModuleType('comfy_extras');module=types.ModuleType('comfy_extras.nodes_qwen')
    class Native:
        @staticmethod
        def execute(**kw):seen.update(kw);return types.SimpleNamespace(result=('positive','negative',{'samples':'matching'}))
    module.TextEncodeQwenImage21=Native
    monkeypatch.setitem(sys.modules,'comfy_extras',parent);monkeypatch.setitem(sys.modules,'comfy_extras.nodes_qwen',module)
    result=q.Q21TNativeEncode().encode('clip','vae',{'image_1':'crop','image_2':'torso'},'edit prompt')
    assert result[2]['samples']=='matching' and seen['resolution']==0 and seen['negative_prompt']==''
    assert seen['images']=={'image_1':'crop','image_2':'torso'}

def core_stub(monkeypatch):
    module=types.ModuleType('nodes')
    class Loader:
        @classmethod
        def INPUT_TYPES(cls):return {'required':{'image':(['test.png'],{'image_upload':True})}}
        @classmethod
        def VALIDATE_INPUTS(cls,image):return image=='test.png'
        @classmethod
        def IS_CHANGED(cls,image):return 'hash:'+image
        def load_image(self,image):return (torch.ones(1,64,64,3),None)
    module.LoadImage=Loader;monkeypatch.setitem(sys.modules,'nodes',module);return module

def test_optional_loader_is_genuinely_empty(monkeypatch):
    core_stub(monkeypatch)
    assert q.Q21TOptionalImage().load('__none__')==(None,)
    assert q.Q21TOptionalImage.VALIDATE_INPUTS('__none__') is True
    assert q.Q21TOptionalImage.IS_CHANGED('__none__')=='__none__'
    assert q.Q21TOptionalImage.INPUT_TYPES()['required']['image'][0][0]=='__none__'

def test_optional_loader_delegates_to_core(monkeypatch):
    core_stub(monkeypatch)
    assert q.Q21TOptionalImage().load('test.png')[0].shape==(1,64,64,3)
    assert q.Q21TOptionalImage.VALIDATE_INPUTS('test.png') is True
    assert q.Q21TOptionalImage.IS_CHANGED('test.png')=='hash:test.png'

def test_all_own_node_schema_orders_and_types(monkeypatch):
    core_stub(monkeypatch)
    for name,cls in q.NODE_CLASS_MAPPINGS.items():
        schema=cls.INPUT_TYPES();inputs={**schema.get('required',{}),**schema.get('optional',{})}
        widgets=[n for n,d in inputs.items() if not (len(d)>1 and d[1].get('forceInput')) and (isinstance(d[0],list) or d[0] in ('INT','FLOAT','STRING','BOOLEAN'))]
        expected=[n for n in generator.S[name][2] if n!='upload']
        assert widgets==expected,(name,widgets,expected)
        assert tuple(cls.RETURN_TYPES)==tuple(generator.S[name][1])
        for input_name,input_type in generator.S[name][0]:assert inputs[input_name][0]==input_type

def test_workflows_validate_in_fresh_python_process():
    p=subprocess.run([sys.executable,str(ROOT/'scripts/validate_workflows.py')],capture_output=True,text=True)
    assert p.returncode==0,p.stderr+p.stdout
    assert '4 UI/API workflow pairs' in p.stdout

def test_generation_is_deterministic():
    files=list((ROOT/'workflows').glob('*.json'))+list((ROOT/'api_workflows').glob('*.json'))+[ROOT/'config/catalog.json']
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    generator.main()
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

@pytest.mark.parametrize('stem',['36_Sports_Bra_Torso_Landmark_Lock','37_Torso_Skin_Seam_Repair_Locked'])
def test_native_graph_uses_matching_latent_and_strict_final(stem):
    api=json.loads((ROOT/'api_workflows'/(stem+'.api.json')).read_text())
    by=lambda t:[(k,v['inputs']) for k,v in api.items() if v['class_type']==t]
    assert not any('Lora' in n['class_type'] or 'SeedVR' in n['class_type'] for n in api.values())
    crop=by('AUSBOSS_NODES_CropForInpaint')[0][1]
    assert crop['output_multiple']==32 and crop['target_megapixels']==2
    assert crop['blend_pixels']==crop['mask_grow']==crop['mask_blur']==0
    prepare_id=by('Q21TPrepareMasks')[0][0]
    assert crop['mask']==[prepare_id,1]
    sampler=by('KSampler')[0][1]
    assert sampler['cfg']==sampler['denoise']==1
    assert sampler['steps']==40 and api[sampler['latent_image'][0]]['class_type']=='Q21TNativeEncode'
    fin=by('Q21TFinalize')[0][1];assert fin['paste_alpha']==[prepare_id,2]
    save=by('Q21TSaveVerified')[0][1];assert api[save['image'][0]]['class_type']=='Q21TFinalize'

@pytest.mark.parametrize('stem',['36A_Check_Masks_Only_NO_MODELS','38_Aligned_Photographic_Patch_NO_AI'])
def test_non_generative_graphs_load_no_models(stem):
    api=json.loads((ROOT/'api_workflows'/(stem+'.api.json')).read_text())
    assert not any(n['class_type'] in ('UNETLoader','CLIPLoader','VAELoader','KSampler','Q21TNativeEncode') for n in api.values())

def test_installer_additive_and_preserves_user_edits(tmp_path):
    comfy=tmp_path/'comfy';user=tmp_path/'user'
    p=comfy/'comfy_extras/nodes_qwen.py';p.parent.mkdir(parents=True);p.write_text('class TextEncodeQwenImage21: pass')
    p=comfy/'custom_nodes/ComfyUI-AusBoss/nodes/node_inpaint_crop_stitch.py';p.parent.mkdir(parents=True);p.write_text('AUSBOSS_NODES_StitchInpaint')
    existing=comfy/'custom_nodes/qwen21_native_tools/__init__.py';existing.parent.mkdir();existing.write_text('UNCHANGED')
    weights=comfy/'models/test.safetensors';weights.parent.mkdir();weights.write_bytes(b'keep')
    out=installer.install(comfy,user)
    assert len(list(out.glob('*.json')))==4
    target=out/'36_Sports_Bra_Torso_Landmark_Lock.json';target.write_text('user edits')
    installer.install(comfy,user)
    assert target.read_text()=='user edits' and existing.read_text()=='UNCHANGED' and weights.read_bytes()==b'keep'
    assert not (user/'qwen21_torso_backups').exists()

def test_installer_requires_compatible_core(tmp_path):
    with pytest.raises(RuntimeError,match='absent'):installer.install(tmp_path,tmp_path/'user')

def test_save_bundle_uses_audited_final_and_sidecars(data,tmp_path,monkeypatch):
    out=finalize(data);module=core_stub(monkeypatch)
    f=types.ModuleType('folder_paths');f.get_output_directory=lambda:str(tmp_path)
    monkeypatch.setitem(sys.modules,'folder_paths',f)
    class Save:
        def save_images(self,images,prefix,prompt,extra):
            q.image_pil(images).save(tmp_path/'final_00001_.png')
            return {'ui':{'images':[{'filename':'final_00001_.png','subfolder':'','type':'output'}]}}
    module.SaveImage=Save
    result=q.Q21TSaveVerified().save(out[0],out[2],out[1],out[3],'test')
    assert 'images' in result['ui']
    assert json.loads((tmp_path/'final_00001_.audit.json').read_text())['status']=='PASS'
    assert Image.open(tmp_path/'final_00001_.png').size==(96,128)
    assert (tmp_path/'final_00001_.edit_mask.png').is_file()
    assert (tmp_path/'final_00001_.comparison.jpg').is_file()
    # Image export retains exactly the same 8-bit values in the protected region.
    actual=np.array(Image.open(tmp_path/'final_00001_.png'))
    expected=np.array(q.image_pil(data[0]));mask=data[2][0].numpy()>0
    assert np.array_equal(actual[mask],expected[mask])
