"""CPU contracts and real PNG writes; SAM/Qwen inference is mocked."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch

from workbench import qwen_people as p
from workbench.qwen_people_catalog import apply, WORKFLOW
from workbench.qwen_people_workflows import make_people, build, SUBGRAPH_ID
from workbench.qwen_stock_workflows import verify_links

ROOT=Path(__file__).resolve().parents[1]


def scene(count=3,h=96,w=240):
    rng=np.random.default_rng(53)
    array=rng.integers(20,230,(h,w,3),dtype=np.uint8)
    source=torch.from_numpy(array.astype(np.float32)/255)[None]
    masks=torch.zeros((count,h,w))
    width=max(3,w//(count*3))
    for i in range(count):
        x=8+i*(w-16)//count
        masks[i,15:h-12,x:min(w-3,x+width)]=1
    return source,masks,array


def package(**kwargs):
    source,masks,array=scene()
    manual=torch.zeros((1,96,240))
    manual[:,30:38,10:20]=0.05  # Any nonzero painted pixel must be protected.
    manual[:,5:15,140:160]=1
    opts=dict(keep_person=2,resolution=256,context_pixels=16,manual_protection=manual)
    opts.update(kwargs)
    data=p.prepare_people(p.people_from_masks(source,masks),**opts)[0]
    return data,array


def generated(data,rgba=False):
    plan=data['plan']
    image=torch.full((1,plan.work_height,plan.work_width,3),0.8)
    return torch.cat([image,torch.ones_like(image[...,:1])],-1) if rgba else image


@pytest.mark.parametrize('count',range(1,6))
def test_detects_one_through_five_and_sorted_ids(count):
    source,masks,_=scene(count=count)
    data=p.people_from_masks(source,masks.flip(0))
    assert data['count']==count
    assert np.array_equal(data['masks'],masks.numpy().astype(bool))
    assert data['source'].device.type=='cpu'


def test_count_does_not_depend_on_confidence_order():
    source,masks,_=scene(5)
    a=p.people_from_masks(source,masks)
    b=p.people_from_masks(source,masks[torch.tensor([3,1,4,0,2])])
    assert a['boxes']==b['boxes']
    assert np.array_equal(a['masks'],b['masks'])


@pytest.mark.parametrize('count',[0,6,7])
def test_zero_or_too_many_stops(count):
    source,masks,_=scene(count=max(1,count))
    if count==0:masks=masks[:0]
    with pytest.raises(ValueError):p.people_from_masks(source,masks)


@pytest.mark.parametrize('case',['size','empty','nan','negative','over1','expected'])
def test_bad_detection_is_not_silently_resized_or_ignored(case):
    source,masks,_=scene()
    expected=0
    if case=='size':masks=masks[:,:,:-1]
    elif case=='empty':masks[1]=0
    elif case=='nan':masks[0,2,2]=float('nan')
    elif case=='negative':masks[0,2,2]=-1
    elif case=='over1':masks[0,2,2]=2
    elif case=='expected':expected=5
    with pytest.raises(ValueError):p.people_from_masks(source,masks,expected)


@pytest.mark.parametrize('index',[0,4,5,True])
def test_keeper_must_exist(index):
    source,masks,_=scene()
    with pytest.raises(ValueError):p.prepare_people(p.people_from_masks(source,masks),keep_person=index)


def test_manual_paint_and_keeper_override_all_growth():
    data,original=package(repair_grow_pixels=40,protection_grow_pixels=4)
    final,overlay,alpha,report=p.compose_people(data,generated(data))
    assert final.dtype==np.uint8 and final.shape==original.shape
    assert overlay.shape==(*original.shape[:2],4)
    for key in ['manual_protection','keep_protection']:
        protect=data[key]
        assert protect.any()
        assert not alpha[protect].any()
        assert np.array_equal(final[protect],original[protect])
    assert np.array_equal(final[alpha==0],original[alpha==0])
    assert np.any(final[alpha>0]!=original[alpha>0])
    assert report['manual_changed_channels']==0 and report['kept_person_changed_channels']==0
    assert report['manual_overlap_with_removed_people_pixels']>0


def test_straight_rgba_recomposition_is_exact():
    data,original=package()
    final,rgba,alpha,_=p.compose_people(data,generated(data),feather_pixels=12)
    recomposed=np.rint(original.astype(np.float64)+(rgba[...,:3].astype(np.float64)-original)*(rgba[...,3:4]/255.)).astype(np.uint8)
    assert np.array_equal(recomposed,final)
    assert np.any((alpha>0)&(alpha<255))


def test_no_full_frame_processing_of_original():
    source,masks,original=scene(h=128,w=360)
    source_copy=source.clone()
    data=p.prepare_people(p.people_from_masks(source,masks),keep_person=2,
                          resolution=256,context_pixels=0,repair_grow_pixels=0)[0]
    assert (data['plan'].work_width*data['plan'].work_height)<=256**2
    final,_,alpha,_=p.compose_people(data,generated(data))
    assert final.shape==original.shape
    assert np.array_equal(final[alpha==0],original[alpha==0])
    assert torch.equal(source,source_copy)


def test_one_person_skips_qwen_and_saves_original():
    source,masks,original=scene(1)
    data=p.prepare_people(p.people_from_masks(source,masks),resolution=256)[0]
    assert data['report']['no_op']
    assert p.WBPeopleExport().check_lazy_status(data,run_edit=True)==[]
    final,rgba,mask,_=p.compose_people(data,None)
    assert np.array_equal(final,original)
    assert not np.any(rgba) and not np.any(mask)


def test_preview_does_not_request_qwen():
    data,_=package()
    node=p.WBPeopleExport()
    assert node.check_lazy_status(data,False)==[]
    assert node.check_lazy_status(data,True)==['generated_crop']
    assert node.check_lazy_status(data,True,generated(data))==[]


def test_blank_placeholder_manual_mask_is_supported():
    data,_=package(manual_protection=torch.zeros((1,64,64)))
    assert not data['manual_protection'].any()


@pytest.mark.parametrize('mask',[torch.ones(1,64,64),torch.full((1,96,240),float('nan')),torch.full((1,96,240),-0.1)])
def test_invalid_manual_mask_fails(mask):
    with pytest.raises(ValueError):package(manual_protection=mask)


def test_all_removal_protected_fails():
    with pytest.raises(ValueError,match='Protection covers'):
        package(manual_protection=torch.ones((1,96,240)))


def test_extra_repair_can_expand_but_never_erase_manual_protection():
    extra=torch.zeros((1,96,240));extra[:,0:30,140:190]=1
    data,_=package(extra_repair_mask=extra)
    assert data['effective_mask'][1,170]
    assert not data['effective_mask'][8,150]  # Painted protection is inside extra repair.


def test_reference_is_isolated_keeper_and_prompt_is_grounded():
    source,masks,_=scene()
    result=p.prepare_people(p.people_from_masks(source,masks),keep_person=2,resolution=256,context_pixels=0)
    data,work,reference,prompt=result
    assert reference.shape[1]%32==reference.shape[2]%32==0
    assert reference.shape[1]*reference.shape[2]<=768**2
    assert 'rough background' in prompt and 'cut out' not in prompt
    assert '2 unwanted person(s)' in prompt and 'gray background' not in prompt
    assert data['report']['remove_people']==[1,3]
    plan=data['plan']
    for x,y,x2,y2 in data['detections']['boxes']:
        assert plan.x<=x<x2<=plan.x+plan.crop_width
        assert plan.y<=y<y2<=plan.y+plan.crop_height


@pytest.mark.parametrize('dims',[(701,1049),(2048,4096),(79,65),(64,14000)])
def test_crop_budget_alignment(dims):
    sw,sh,ww,wh=p._budget(*dims,1536)
    assert ww%32==wh%32==0
    assert ww*wh<=1536**2 and max(ww,wh)<=3072
    assert sw<=dims[0] and sh<=dims[1]
    assert 0<=ww-sw<32 and 0<=wh-sh<32


def test_padding_removed_before_stitch():
    data,_=package()
    a=generated(data);b=a.clone();plan=data['plan']
    b[:,plan.scaled_height:,:,:]=0
    b[:,:,plan.scaled_width:,:]=0
    assert np.array_equal(p.compose_people(data,a)[0],p.compose_people(data,b)[0])


@pytest.mark.parametrize('case',['dimensions','nan','transparent'])
def test_bad_generation_never_saved(case):
    data,_=package()
    image=generated(data,rgba=case=='transparent')
    if case=='dimensions':image=image[:,:-1]
    elif case=='nan':image[0,0,0,0]=float('nan')
    else:image[...,3]=0
    with pytest.raises(ValueError):p.compose_people(data,image,transparency_policy='stop on transparency')


def test_opaque_rgba_generation_supported():
    data,_=package()
    assert p.compose_people(data,generated(data,True))[0].shape[-1]==3


@pytest.fixture
def folders(tmp_path,monkeypatch):
    out=tmp_path/'output';temp=tmp_path/'temp';out.mkdir();temp.mkdir()
    def get_save(prefix,root,w,h):
        path=Path(prefix)
        return str(Path(root)/path.parent),path.name,1,str(path.parent),prefix
    module=SimpleNamespace(get_output_directory=lambda:str(out),get_temp_directory=lambda:str(temp),
                           get_save_image_path=get_save)
    monkeypatch.setitem(sys.modules,'folder_paths',module)
    monkeypatch.setenv('WB_SAVE_METADATA','0')
    return out,temp


def test_real_png_outputs_and_metadata(folders):
    out,temp=folders
    data,original=package()
    result=p.WBPeopleExport().save(data,True,generated_crop=generated(data),
        prompt={'secret_fixture':'do-not-export'},extra_pnginfo={'workflow':{'private':'omit'}})
    shown=result['ui']['images']
    assert len(shown)==1 and shown[0]['filename'].endswith('_composite.png')
    files=list(out.rglob('*.png'));assert len(files)==3
    composite=next(f for f in files if f.name.endswith('_composite.png'))
    overlay=next(f for f in files if f.name.endswith('_overlay.png'))
    with Image.open(composite) as im:
        assert im.mode=='RGB' and im.size==(240,96)
        assert 'icc_profile' in im.info
        assert 'prompt' not in im.info and 'workflow' not in im.info
        final=np.asarray(im).copy()
    with Image.open(overlay) as im:
        assert im.mode=='RGBA' and im.size==(240,96)
        rgba=np.asarray(im).copy()
    assert not rgba[...,3][data['manual_protection']].any()
    assert not rgba[...,3][data['keep_protection']].any()
    check=np.rint(original.astype(np.float64)+(rgba[...,:3].astype(np.float64)-original)*(rgba[...,3:4]/255.)).astype(np.uint8)
    assert np.array_equal(check,final)
    assert len(list(out.rglob('*_audit.json')))==1
    assert not list(out.rglob('.qkeep-*'))


def test_show_overlay_adds_second_preview_not_first(folders):
    data,_=package()
    result=p.WBPeopleExport().save(data,True,show_overlay_preview=True,generated_crop=generated(data))
    images=result['ui']['images']
    assert len(images)==2
    assert images[0]['filename'].endswith('_composite.png')
    assert images[1]['filename'].endswith('_overlay.png')


def test_review_writes_only_temp_numbered_preview(folders):
    out,temp=folders
    data,_=package()
    result=p.WBPeopleExport().save(data,False)
    assert result['ui']['images'][0]['type']=='temp'
    assert '3 people' in result['ui']['text'][0]
    assert list(temp.glob('*.png')) and not list(out.rglob('*.png'))


def test_no_output_on_invalid_generated_crop(folders):
    out,_=folders;data,_=package()
    with pytest.raises(ValueError):p.WBPeopleExport().save(data,True,generated_crop=generated(data)[:,:-1])
    assert not list(out.rglob('*.png'))


def test_output_prefix_cannot_escape(folders):
    data,_=package()
    with pytest.raises(ValueError,match='escapes'):
        p.WBPeopleExport().save(data,True,filename_prefix='../outside/name',generated_crop=generated(data))


def mock_sam(monkeypatch,masks,fail=False):
    mm=ModuleType('comfy.model_management');mm.unload_all_models=Mock();mm.soft_empty_cache=Mock()
    comfy=ModuleType('comfy');comfy.model_management=mm
    native=ModuleType('comfy_extras.nodes_sam3')
    executor=Mock(side_effect=ValueError('SAM failure')) if fail else Mock(return_value=SimpleNamespace(result=(masks,None)))
    native.SAM3_Detect=SimpleNamespace(execute=executor)
    node=ModuleType('nodes');loader=Mock(return_value=('sam_model','sam_clip','vae'))
    encode=Mock(return_value=('cond',))
    node.CheckpointLoaderSimple=lambda:SimpleNamespace(load_checkpoint=loader)
    node.CLIPTextEncode=lambda:SimpleNamespace(encode=encode)
    for name,module in [('nodes',node),('comfy',comfy),('comfy.model_management',mm),
                        ('comfy_extras',ModuleType('comfy_extras')),('comfy_extras.nodes_sam3',native),
                        ('folder_paths',SimpleNamespace(get_full_path=lambda *a:'fixture.safetensors'))]:
        monkeypatch.setitem(sys.modules,name,module)
    return executor,encode,mm


def test_native_sam_multi_instance_contract_and_unload(monkeypatch):
    source,masks,_=scene(5)
    call,encode,mm=mock_sam(monkeypatch,masks)
    data,count=p.WBPeopleDetect().detect(source)
    assert count==5
    encode.assert_called_once_with('sam_clip','person:6')
    assert call.call_args.kwargs['individual_masks'] is True
    assert call.call_args.kwargs['refine_iterations']==2
    mm.unload_all_models.assert_called_once();mm.soft_empty_cache.assert_called_once()


@pytest.mark.parametrize('failure',['inference','overflow'])
def test_sam_failures_still_release_models(monkeypatch,failure):
    source,masks,_=scene(6)
    _,_,mm=mock_sam(monkeypatch,masks,fail=failure=='inference')
    with pytest.raises(ValueError):p.WBPeopleDetect().detect(source)
    mm.unload_all_models.assert_called_once();mm.soft_empty_cache.assert_called_once()


def stock():
    path=ROOT/'workflows/qwen/QNATIVE/Qwen21_Comfy_Template_BF16.json'
    assert path.exists(),'Build native workflows with scripts/build_site.py before testing.'
    return json.loads(path.read_text())


def test_graph_native_settings_and_wiring():
    original=stock();before=copy.deepcopy(original)
    graph=make_people(original)
    assert original==before
    verify_links(graph)
    nodes={n['type']:n for n in graph['nodes']}
    assert {'WBPeopleDetect','WBPeoplePrepare','WBPeopleExport','LoadImage','Note',SUBGRAPH_ID}==set(nodes)
    assert nodes['WBPeopleExport']['widgets_values'][0] is False
    assert nodes['WBPeoplePrepare']['widgets_values'][-1]==p.DEFAULT_INSTRUCTIONS
    assert nodes['WBPeoplePrepare']['inputs'][1]['name']=='manual_protection'
    assert nodes['WBPeoplePrepare']['inputs'][1]['link'] is not None
    assert nodes['WBPeoplePrepare']['inputs'][2]['link'] is None
    new=nodes[SUBGRAPH_ID]
    old=next(n for n in original['nodes'] if 'unet_name' in n.get('widgets_values_named',{}))
    for key,value in old['widgets_values_named'].items():
        if key!='prompt':assert new['widgets_values_named'][key]==value
    assert next(i for i in new['inputs'] if i.get('label')=='image_1')['link'] is not None
    assert next(i for i in new['inputs'] if i.get('label')=='image_2')['link'] is None
    assert next(i for i in new['inputs'] if i['name']=='prompt')['link'] is not None
    assert graph['extra']['qkeep1']['text_preview_sinks_removed']==1


def test_subgraph_preview_removed_but_generation_nodes_unchanged():
    original=stock();graph=make_people(original)
    old=original['definitions']['subgraphs'][0];new=graph['definitions']['subgraphs'][0]
    before={n['type']:n for n in old['nodes']};after={n['type']:n for n in new['nodes']}
    assert 'PreviewAny' not in after and 'TextGenerate' in after
    for typ in ['UNETLoader','VAELoader','CLIPLoader','KSampler','TextGenerate']:
        assert before[typ].get('widgets_values')==after[typ].get('widgets_values')
    index={n['id']:n for n in new['nodes']}
    pseudo={new['inputNode']['id'],new['outputNode']['id']}
    for row in new['links']:
        a,b=row['origin_id'],row['target_id']
        assert a in index or a in pseudo
        assert b in index or b in pseudo
        if a in index:assert row['id'] in index[a]['outputs'][row['origin_slot']]['links']
        if b in index:assert row['id']==index[b]['inputs'][row['target_slot']]['link']
    assert len([n for n in graph['nodes'] if n['type']=='WBPeopleExport'])==1


def test_graph_reproducible_and_matches_committed_json(tmp_path):
    from workbench.qwen_stock_catalog import STOCK_PATH
    path=tmp_path/STOCK_PATH;path.parent.mkdir(parents=True);path.write_text(json.dumps(stock()))
    build(tmp_path);first=(tmp_path/WORKFLOW).read_bytes()
    build(tmp_path);assert first==(tmp_path/WORKFLOW).read_bytes()
    committed=json.loads((ROOT/WORKFLOW).read_text())
    assert json.loads(first)==committed


def test_catalog_requires_only_existing_weights_and_is_additive(tmp_path):
    (tmp_path/'catalog').mkdir()
    (tmp_path/'catalog/tasks.qwen-people.json').write_bytes((ROOT/'catalog/tasks.qwen-people.json').read_bytes())
    models={'models':[{'id':'qwen','local_nodes':['old'],'default_tasks':['SQ']},
                      {'id':'krea','local_nodes':['keep'],'default_tasks':['K01']}]}
    result=apply('models.json',copy.deepcopy(models),tmp_path)
    assert result['models'][0]['local_nodes']==['old','qwen_people']
    assert result['models'][0]['default_tasks']==['SQ']
    assert result['models'][1]==models['models'][1]
    tasks={'tasks':[{'id':'existing'}]}
    result=apply('tasks.json',copy.deepcopy(tasks),tmp_path)
    assert result['tasks'][0]==tasks['tasks'][0]
    assert result['tasks'][1]['requires']==['qwen','sam','qwen_stock_pe']
    with pytest.raises(ValueError):apply('tasks.json',result,tmp_path)
    assets={'assets':[{'id':'existing-weight'}]}
    assert apply('assets.json',assets,tmp_path) is assets


def test_no_addon_means_no_fixture_catalog_changes(tmp_path):
    value={'tasks':[{'id':'old'}]}
    assert apply('tasks.json',copy.deepcopy(value),tmp_path)==value
