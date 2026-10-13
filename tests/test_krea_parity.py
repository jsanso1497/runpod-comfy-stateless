"""Krea/Qwen parity regression tests. All inference and service calls are mocked."""
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import Mock
import numpy as np
from PIL import Image, ImageCms
import pytest
import torch
from test_krea_edit import krea, package, load_module, ROOT


@pytest.mark.parametrize('bad',['Use R1 only.','Use R1 and R9.','Use image 3.','<think>unfinished'])
def test_invalid_rewrite_retries_then_uses_original(krea,monkeypatch,bad):
    from workbench import krea_prompt as pe
    original = 'Use R1 and R2. Keep "ABC". exact_token'
    reply = Mock(return_value=bad)
    monkeypatch.setattr(pe,'_local_reply',reply)
    monkeypatch.setattr(pe.local,'failure_mode',lambda:'original')
    result,report = pe.enhance(package(krea,2),original,True,triggers='exact_token')
    assert reply.call_count == 2 and report['fallback_used']
    assert result.endswith(original+'\nexact_token')
    assert report['attempts'] == 2 and 'WARNING' in report['status']


def test_repair_can_succeed_without_dropping_reference_or_trigger(krea,monkeypatch):
    from workbench import krea_prompt as pe
    reply = Mock(side_effect=['Use R1.', 'Use R1 and R2. Keep "ABC". exact_token'])
    monkeypatch.setattr(pe,'_local_reply',reply)
    result,report = pe.enhance(package(krea,2),'Use R1 and R2. Keep "ABC".',True,triggers='exact_token')
    assert report['retry_count'] == 1 and not report['fallback_used']
    assert 'R2' in result and 'exact_token' in result


@pytest.mark.parametrize('reason',['missing weights','wrong precision','no vision','out of memory','still resident'])
def test_service_and_residency_errors_never_fallback(krea,monkeypatch,reason):
    from workbench import krea_prompt as pe
    reply = Mock(side_effect=pe.local.LocalServiceError(reason))
    monkeypatch.setattr(pe,'_local_reply',reply)
    with pytest.raises(pe.local.LocalServiceError,match=reason):
        pe.enhance(package(krea),'Red coat.',True)
    assert reply.call_count == 1


def test_unload_runs_when_local_chat_fails(krea,monkeypatch):
    from workbench import krea_prompt as pe
    monkeypatch.setattr(pe.local,'verify_model',lambda *args:None)
    monkeypatch.setattr(pe.local,'vision_png',lambda image:'png')
    release = Mock()
    monkeypatch.setattr(pe.local,'unload_and_verify',release)
    def request(path,*args,**kwargs):
        if path == '/api/show':return {'capabilities':['vision']}
        raise pe.local.LocalServiceError('chat failed')
    monkeypatch.setattr(pe.local,'request_json',request)
    with pytest.raises(pe.local.LocalServiceError,match='chat failed'):
        pe.enhance(package(krea),'Red coat.',True)
    release.assert_called_once()


def test_standard_rejects_image_dependent_rewrite_and_official_backend(krea,monkeypatch):
    from workbench import krea_prompt as pe
    reply = Mock(return_value='Use R1 from image 2.')
    monkeypatch.setattr(pe,'_local_reply',reply)
    _,report = pe.enhance(package(krea),'A red coat.',True,standard=True)
    assert report['fallback_used'] and reply.call_count == 2
    with pytest.raises(ValueError,match='edit enhancer'):
        pe.enhance(package(krea),'A red coat.',True,backend=pe.OFFICIAL,standard=True)


def test_official_backend_gets_exact_source_and_sheet(krea,monkeypatch):
    from workbench import krea_prompt as pe
    from workbench import qwen_prompt
    p = package(krea,4)
    call = Mock(return_value=('Final edit.',{'enabled':True,'backend':'official','attempts':1,'fallback_used':False}))
    monkeypatch.setattr(qwen_prompt,'enhance_edit_prompt',call)
    result,report = pe.enhance(p,'Use R1 and R4.',True,backend=pe.OFFICIAL)
    images = call.call_args.args[1]
    assert len(images) == 2 and images['image_1'] is p['source'] and images['image_2'] is p['board']
    assert result == 'Final edit.' and report['backend'] == 'official'


def test_legacy_helper_tag_uses_workspace_selection(monkeypatch):
    from workbench import krea_prompt as pe
    monkeypatch.setattr(pe.local,'selected_model',lambda:pe.local.ALTERNATE_MODEL)
    assert pe.selected_model('qwen3.8:27b-bf16') == pe.local.ALTERNATE_MODEL
    assert pe.selected_model('auto') == pe.local.ALTERNATE_MODEL
    with pytest.raises(pe.local.LocalServiceError):pe.selected_model('small-quantized-model')


@pytest.fixture
def photo(krea):
    return sys.modules[krea.__name__+'.photo_nodes']


@pytest.mark.parametrize('shape',[(97,151),(512,256),(1536,2048)])
def test_masked_crop_preserves_canvas_holes_and_outside_pixels(photo,shape):
    h,w = shape
    source = torch.rand(1,h,w,3)
    mask = torch.zeros(1,h,w)
    mask[:,h//4:3*h//4,w//4:3*w//4] = 1
    mask[:,h//2:h//2+2,w//2:w//2+2] = 0
    data,work = photo.WBKreaCrop().crop(source,mask,'{}',megapixels=1.9)
    assert work.shape[1]*work.shape[2] <= 2_000_000
    result = photo.WBKreaStitch().stitch(data,torch.zeros_like(work))[0]
    assert result['image'].shape == source.shape
    assert torch.equal(result['image'][mask==0],source[mask==0])
    assert result['report']['outside_mask_changed_channels'] == 0
    assert result['report']['model'] == 'Krea 2 Identity Edit'


def test_protection_applied_after_expanded_mask(photo):
    source = torch.rand(1,128,128,3)
    mask = torch.ones(1,128,128)
    protect = torch.zeros_like(mask);protect[:,50:80,50:80] = 1
    data,work = photo.WBKreaCrop().crop(source,mask,'{}',protect_mask=protect)
    result = photo.WBKreaStitch().stitch(data,torch.zeros_like(work))[0]['image']
    assert torch.equal(result[protect>0],source[protect>0])


def test_empty_shifted_and_oversized_inputs_rejected(photo):
    source = torch.rand(1,128,128,3)
    for mask in [torch.zeros(1,64,64),torch.ones(1,64,64)]:
        with pytest.raises(ValueError):photo.WBKreaCrop().crop(source,mask,'{}')
    with pytest.raises(ValueError):
        photo.WBKreaCrop().crop(source,torch.ones(1,128,128),'{}',megapixels=4.)
    data,work = photo.WBKreaCrop().crop(source,torch.ones(1,128,128),'{}')
    with pytest.raises(ValueError):photo.WBKreaStitch().stitch(data,work[:,:-1])


def test_preview_is_lazy_and_only_output_in_masked_graph(photo):
    node = photo.WBKreaReviewSave()
    assert node.check_lazy_status({},False) == []
    assert node.check_lazy_status({},True) == ['edited_result']
    generator = load_module('krea_lazy_generator',ROOT/'scripts/build_krea_workflow.py')
    graph,api = generator.build_masked()
    assert api['13']['inputs']['run_edit'] is False
    assert api['13']['class_type'] == 'WBKreaReviewSave'
    assert not any(n['type'] in ('PreviewImage','SaveImage','WBKreaSaveImage','WBQwenPromptPreview') for n in graph['nodes'])
    assert api['16']['inputs']['source'] == ['1',0]
    assert api['6']['inputs']['source'] == ['18',1]


def test_source_loader_keeps_native_upload_contract(krea):
    assert krea.NODE_CLASS_MAPPINGS['WBKreaLoadImage'].INPUT_TYPES()['required']['image'][1]['image_upload']


def test_srgb_export_and_metadata_opt_in(photo,tmp_path,monkeypatch):
    monkeypatch.setattr(photo.folder_paths,'get_output_directory',lambda:str(tmp_path),raising=False)
    monkeypatch.setattr(photo.folder_paths,'get_save_image_path',lambda *args:(str(tmp_path),'edit',1,'',None),raising=False)
    monkeypatch.setenv('WB_SAVE_METADATA','0')
    saved = photo._save(torch.rand(1,64,64,3),'edit',{'outside_mask_changed_channels':0},prompt={'private':'prompt'})
    image = Image.open(tmp_path/saved['filename'])
    assert image.info.get('icc_profile') and 'prompt' not in image.info
    assert 'krea_photo_verification' in image.info
    assert (tmp_path/saved['filename']).with_suffix('.verification.json').is_file()


def test_color_profile_conversion_preserves_mask_geometry(photo,tmp_path,monkeypatch):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    path = tmp_path/'profile.png'
    Image.new('RGB',(80,64),(30,80,120)).save(path,icc_profile=profile)
    native = torch.from_numpy(np.asarray(Image.open(path)).copy()).float().unsqueeze(0)/255
    mask = torch.zeros(1,64,80)
    monkeypatch.setattr(photo.folder_paths,'get_annotated_filepath',lambda image:str(path))
    monkeypatch.setattr(photo.nodes.LoadImage,'load_image',lambda self,image:(native,mask),raising=False)
    image,loaded_mask = photo.load_srgb('profile.png')
    assert image.shape == native.shape and loaded_mask is mask
    assert torch.allclose(image,native)


def test_krea_catalog_uses_shared_helper_and_optional_tools():
    from workbench import krea_catalog as catalog
    from workbench.qwen_prompt_local import selected_model
    value = {'models':[{'id':'krea','local_nodes':['krea_edit'],'external_nodes':['krea2edit'],'toolbox':['ollama']} ]}
    row = catalog.apply('models.json',value)['models'][0]
    assert 'qwen_prompt' in row['local_nodes'] and 'seedvr2' in row['external_nodes']
    assert {'sam','seedvr2','qwen_pe_i2i'} <= set(row['toolbox'])
    assert catalog.apply('assets.json',{})['ollama_by_workspace']['krea']['models'] == [selected_model()]
