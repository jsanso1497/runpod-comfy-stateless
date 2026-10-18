"""CPU image assertions and actual PNG writes; model inference is not run."""
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from workbench import qwen_people as p
from workbench import qkeep_repair as r
from workbench.sam_selection import detection_prompt


def data():
    h,w=96,256
    source=torch.full((1,h,w,3),.3)
    masks=torch.zeros((3,h,w))
    for i,x in enumerate((20,110,195)):
        masks[i,20:80,x:x+25]=1
        source[0,20:80,x:x+25]=torch.tensor([1.,0.,0.])
    manual=torch.zeros((1,h,w));manual[:,50:55,210:214]=.01
    result=p.prepare_people(p.people_from_masks(source,masks),keep_person=2,
        resolution=256,repair_grow_pixels=6,manual_protection=manual)
    return result[0],result[1],source


@pytest.mark.parametrize('alpha',[0.,.1,.5,.9,.96])
def test_hidden_purple_never_exposed(alpha):
    package,work,source=data()
    raw=torch.zeros((*work.shape[:-1],4));raw[...,:3]=torch.tensor([1.,0.,1.]);raw[...,3]=alpha
    final,rgba,mask,audit=p.compose_people(package,raw,0)
    allowed=package['effective_mask']
    # Coarse prefill uses only the gray background, never red-person donors.
    assert np.max(final[allowed])-np.min(final[allowed])<=1
    original=np.rint(source[0].numpy()*255).astype(np.uint8)
    assert np.array_equal(final[~allowed],original[~allowed])
    assert not rgba[...,3][package['keep_protection']].any()
    assert not rgba[...,3][package['manual_protection']].any()
    assert audit['completion_status']=='DRAFT_FALLBACK_REVIEW_REQUIRED'
    assert audit['semantic_repair_verified'] is False
    assert audit['generated_low_alpha_edit_fraction']==1.


def test_original_source_untouched_and_prefill_not_cutout():
    package,work,source=data()
    assert torch.equal(package['source'],source)
    assert work.shape[-1]==3 and torch.isfinite(work).all()
    assert package['report']['prefill_background_donor_pixels']>0
    assert p.WBPeopleExport().check_lazy_status(package,False)==[]


def test_confident_model_rgb_not_replaced_by_prefill():
    package,work,_=data();raw=torch.ones((*work.shape[:-1],4))
    raw[...,:3]=torch.tensor([.2,.4,.7])
    final,rgba,mask,audit=p.compose_people(package,raw,0)
    assert not audit['fallback_review_required']
    assert np.array_equal(final[package['effective_mask']][0],np.rint(np.array([.2,.4,.7])*255).astype(np.uint8))


def test_strict_mode_rejects_transparency():
    package,work,_=data();raw=torch.zeros((*work.shape[:-1],4))
    with pytest.raises(ValueError,match='No final photo'):p.compose_people(package,raw,transparency_policy='stop on transparency')


def test_no_background_donors_is_actionable():
    with pytest.raises(ValueError,match='visible background'):
        r.prefill(torch.zeros(1,32,32,3),np.ones((32,32),bool),np.zeros((32,32),bool),32,32)


def test_migrates_old_generated_cutout_prompt():
    package,_,source=data();detections=package['detections']
    result=p.prepare_people(detections,keep_person=2,resolution=256,
        instructions='Remove all people from <image1> except the single person shown in <image2>.')
    assert result[3].startswith(r.DEFAULT_INSTRUCTIONS)
    assert '<image2>' not in result[3]


def test_custom_prompt_is_preserved():
    package,_,_=data()
    result=p.prepare_people(package['detections'],keep_person=2,resolution=256,instructions='Paint a quiet beach in the filled regions.')
    assert result[3].startswith('Paint a quiet beach')


def test_draft_filename_and_diagnostics(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir()
    def paths(prefix,root,w,h):
        pp=Path(prefix);return str(Path(root)/pp.parent),pp.name,1,str(pp.parent),prefix
    monkeypatch.setitem(__import__('sys').modules,'folder_paths',SimpleNamespace(
        get_output_directory=lambda:str(out),get_save_image_path=paths))
    package,work,source=data();raw=torch.zeros((*work.shape[:-1],4));raw[...,0]=1;raw[...,2]=1
    result=p.WBPeopleExport().save(package,True,generated_crop=raw)
    assert '_DRAFT_FALLBACK_' in result['ui']['images'][0]['filename']
    assert 'NOT VERIFIED REPAIR' in result['ui']['text'][0]
    files=list(out.rglob('*.png'));assert len(files)==5
    final=np.array(Image.open(next(f for f in files if f.name.endswith('_composite.png'))))
    rgba=np.array(Image.open(next(f for f in files if f.name.endswith('_overlay.png'))))
    original=np.rint(source[0].numpy()*255).astype(np.uint8)
    recomposed=np.rint(original.astype(float)+(rgba[...,:3].astype(float)-original)*(rgba[...,3:]/255)).astype(np.uint8)
    assert np.array_equal(recomposed,final)


@pytest.mark.parametrize('text,index,expected',[
    ('watermark',-1,'watermark:64'),('person:6',-1,'person:6'),
    ('shirt sleeves:2',-1,'shirt sleeves:2'),('face, hat',-1,'face:64, hat:64'),
    ('watermark',0,'watermark'),('person: 5',-1,'person: 5')])
def test_sam_all_instances_is_actually_plural(text,index,expected):
    assert detection_prompt(text,index)==expected
