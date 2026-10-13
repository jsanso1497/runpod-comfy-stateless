"""Offline prompt-routing and safety tests. Inference is mocked, not GPU-tested."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from workbench import qwen_prompt as pe, config, assets, public_companions as companions

ROOT=Path(__file__).resolve().parents[1]

def answer(prompt='A precise edit.', follow='<image1>', ratio=''):
    return json.dumps({'rewritten_prompt':prompt,'ratio_follow':follow,'wh_ratio':ratio})

@pytest.fixture(autouse=True)
def clean_cache():
    pe._CACHE.clear();pe._VERIFIED.clear()
    yield
    pe._CACHE.clear();pe._VERIFIED.clear()

def test_off_does_not_touch_images_or_model(monkeypatch):
    fail=Mock(side_effect=AssertionError('OFF must not load or inspect'))
    monkeypatch.setattr(pe,'_generate',fail);monkeypatch.setattr(pe,'_ordered_images',fail)
    original='  Edit <image1>.\nPreserve\twhitespace.  '
    result,report=pe.enhance_edit_prompt(original,None,{'enabled':False})
    assert result==original and report['original_prompt']==original and not report['enabled']
    fail.assert_not_called()

@pytest.mark.parametrize('settings',[None,{}, {'enabled':False,'seed':7}])
def test_old_or_off_graphs_are_exact(settings):
    assert pe.enhance_edit_prompt(' a\n ',{},settings)[0]==' a\n '

@pytest.mark.parametrize('bad',[1,True,'yes',{'enabled':'true'},{'seed':-1},{'seed':2**64},{'keep_exact':None}])
def test_settings_reject_bad_values(bad):
    with pytest.raises(pe.EnhancementError):pe.Options.from_value(bad)

@pytest.mark.parametrize('prefix',['','<think>hidden analysis</think>','hidden analysis</think>','```json\n'])
def test_final_answer_only(prefix):
    raw=prefix+answer()+ ('\n```' if prefix.startswith('```') else '')
    result=pe.parse_response(raw,'edit',1)
    assert result['rewritten_prompt']=='A precise edit.'
    assert 'hidden' not in str(result)

@pytest.mark.parametrize('raw',[
    '<think>unfinished','{}','[]','not json',answer('',follow='<image1>'),
    answer('x',follow='<image2>'),answer('x',follow='',ratio='0:1'),
    answer('x',follow='',ratio=''),answer('x',follow='<image1>',ratio='4:3'),
    answer('x <image11>'),answer('<think>bad</think>')])
def test_invalid_response_stops_generation(raw):
    with pytest.raises(pe.EnhancementError):pe.parse_response(raw,'edit',1)

def test_single_image_does_not_require_literal_tag():
    assert pe.parse_response(answer('Make the wall blue.'),'Edit <image1>',1)['rewritten_prompt']

def test_multi_image_role_tags_cannot_disappear():
    with pytest.raises(pe.EnhancementError):
        pe.parse_response(answer('Edit <image1>.'),'Edit <image1> with <image2>',2)
    assert pe.parse_response(answer('Edit <image1> using <image2>.'),'Edit <image1> with <image2>',2)

@pytest.mark.parametrize('original,lock', [('Add "EXACT TEXT".',''),('Use <lora:private:0.8>.',''),('Use special_trigger.','special_trigger')])
def test_exact_text_and_trigger_guards(original,lock):
    with pytest.raises(pe.EnhancementError):pe.parse_response(answer('Replace with something else.'),original,1,lock)
    assert pe.parse_response(answer(original),original,1,lock)

def test_keep_exact_terms_only_apply_to_matching_pass():
    assert pe.parse_response(answer('Preserve this person.'),'Preserve this person.',1,'other_person_trigger')

def test_images_order_shapes_identity_and_cached_result(monkeypatch):
    a=torch.rand(1,32,64,3);b=torch.rand(1,64,32,3)
    snapshot=(a.clone(),b.clone());calls=[]
    def generate(prompt,images,options):
        calls.append((prompt,images,options));return answer('Edit <image1> using <image2>.')
    monkeypatch.setattr(pe,'_generate',generate)
    kwargs={'prompt':'Edit <image1> with <image2>','images':{'image_2':b,'image_1':a},'settings':{'enabled':True,'seed':42}}
    first,report=pe.enhance_edit_prompt(**kwargs);second,again=pe.enhance_edit_prompt(**kwargs)
    assert first==second and not report['cache_hit'] and again['cache_hit']
    assert len(calls)==1 and calls[0][1][0].shape==a.shape and calls[0][1][1].shape==b.shape
    assert torch.equal(a,snapshot[0]) and torch.equal(b,snapshot[1])
    assert calls[0][1][0].data_ptr()==a.data_ptr() and calls[0][1][1].data_ptr()==b.data_ptr()
    assert report['ratio_applied'] is False
    kwargs['settings']['seed']=43;pe.enhance_edit_prompt(**kwargs);assert len(calls)==2
    a[0,0,0,0]=0;pe.enhance_edit_prompt(**kwargs);assert len(calls)==3
    kwargs['prompt']+=' carefully';pe.enhance_edit_prompt(**kwargs);assert len(calls)==4

@pytest.mark.parametrize('images',[{}, {'image_2':torch.zeros(1,32,32,3)},
    {'image_1':torch.zeros(2,32,32,3)}, {'image_1':torch.full((1,32,32,3),float('nan'))}])
def test_bad_images_rejected_before_inference(images,monkeypatch):
    mock=Mock();monkeypatch.setattr(pe,'_generate',mock)
    with pytest.raises(pe.EnhancementError):pe.enhance_edit_prompt('edit',images,{'enabled':True})
    mock.assert_not_called()

def test_missing_assets_is_actionable(tmp_path,monkeypatch):
    monkeypatch.setenv('WB_DATA_ROOT',str(tmp_path))
    with pytest.raises(pe.EnhancementError,match='U10'):pe._installed_assets()

def test_companion_allowlist_is_private_library_independent(tmp_path):
    row={'group':'qwen_pe_i2i','repo_id':'personal/private','filename':'system_prompt.txt','revision':pe.REVISION}
    with pytest.raises(assets.pd.LinkError,match='allowlist'):companions.prepare_companion(row,tmp_path/'target',Mock())

class Response:
    status_code=200
    def __init__(self,data):self.data=data
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_content(self,*args):yield self.data

@pytest.mark.parametrize('name,data',[('system_prompt.txt',b'Official system prompt fixture.'),('model.safetensors.index.json',b'{"weight_map":{"weight":"model-00001.safetensors"}}')])
def test_companion_verified_transfer_and_cache(name,data,tmp_path):
    blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
    transport=Mock();transport.json.return_value={'sha':pe.REVISION,'siblings':[{'rfilename':name,'size':len(data),'blobId':blob}]};transport.request.return_value=Response(data)
    row={'group':pe.GROUP,'repo_id':pe.MODEL_ID,'filename':name,'revision':pe.REVISION}
    target=tmp_path/name
    assert companions.prepare_companion(row,target,transport)['git_blob_sha1']==blob
    assert target.read_bytes()==data
    companions.prepare_companion(row,target,transport);assert transport.request.call_count==1

@pytest.mark.parametrize('bad',['revision','checksum','size'])
def test_companion_download_rejects_tampering(bad,tmp_path):
    data=b'abc';blob=hashlib.sha1(b'blob 3\0abc').hexdigest()
    meta={'sha':pe.REVISION,'siblings':[{'rfilename':'system_prompt.txt','size':3,'blobId':blob}]}
    if bad=='revision':meta['sha']='a'*40
    if bad=='checksum':meta['siblings'][0]['blobId']='a'*40
    if bad=='size':meta['siblings'][0]['size']=2
    transport=Mock();transport.json.return_value=meta;transport.request.return_value=Response(data)
    with pytest.raises(assets.pd.LinkError):companions.prepare_companion({'group':pe.GROUP,'repo_id':pe.MODEL_ID,'filename':'system_prompt.txt','revision':pe.REVISION},tmp_path/'out',transport)
    assert not (tmp_path/'out').exists()

def import_node(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path,submodule_search_locations=[str((ROOT/path).parent)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

@pytest.fixture
def fake_native(monkeypatch):
    module=ModuleType('comfy_extras.nodes_qwen');parent=ModuleType('comfy_extras');calls=[]
    def execute(**kwargs):
        calls.append(kwargs);img=kwargs['images']['image_1']
        return SimpleNamespace(result=('pos','neg',{'samples':torch.zeros(1,64,img.shape[1]//16,img.shape[2]//16)}))
    module.TextEncodeQwenImage21=SimpleNamespace(execute=execute);parent.nodes_qwen=module
    monkeypatch.setitem(sys.modules,'comfy_extras',parent);monkeypatch.setitem(sys.modules,'comfy_extras.nodes_qwen',module)
    return calls

@pytest.mark.parametrize('enabled',[False,True])
def test_native_encoder_routes_only_prompt(enabled,monkeypatch,fake_native):
    native=import_node('test_native_prompt','src/nodes/qwen_native/__init__.py')
    image=torch.zeros(1,32,64,3);reference=torch.zeros(1,64,32,3);prompt=' Edit <image1> with <image2>. '
    monkeypatch.setattr(pe,'_generate',lambda *a:answer('Revised <image1> with <image2>.'))
    output=native.Q21NEncodeReferences().encode('clip','vae',image,prompt,prompt_enhancer={'enabled':enabled},image_2=reference)
    args=fake_native[0]
    assert args['prompt']==('Revised <image1> with <image2>.' if enabled else prompt)
    assert args['resolution']==0 and args['negative_prompt']=='' and args['clip']=='clip' and args['vae']=='vae'
    assert args['images']['image_1'] is image and args['images']['image_2'] is reference
    assert output['result'][:2]==('pos','neg') and output['ui']['qwen_prompt'][0]['enabled']==enabled

def test_native_old_graph_without_socket_returns_tuple(fake_native):
    native=import_node('test_native_old','src/nodes/qwen_native/__init__.py')
    out=native.Q21NEncodeReferences().encode('clip','vae',torch.zeros(1,32,32,3),'edit')
    assert isinstance(out,tuple) and fake_native[0]['prompt']=='edit'

@pytest.mark.parametrize('enabled',[False,True])
def test_photo_roles_reach_enhancer_without_changing_crop(enabled,monkeypatch,fake_native):
    photo=import_node('test_photo_prompt','src/nodes/qwen_photo/__init__.py')
    image=torch.zeros(1,32,64,3);ref=torch.zeros(1,64,32,3);captured=[]
    def generate(prompt,images,options):captured.append(prompt);return answer('Edit <image1> using identity <image2>.')
    monkeypatch.setattr(pe,'_generate',generate)
    crop={'work_image':image,'plan':SimpleNamespace(work_height=32,work_width=64)}
    reference={'image':ref,'role':'identity only','filename':'face.png','enabled':True}
    out=photo.Q21PhotoEncode().encode('clip','vae',crop,'Edit <image1> using <image2>.',prompt_enhancer={'enabled':enabled},reference_1=reference)
    args=fake_native[0]
    assert args['images']['image_1'] is image and args['images']['image_2'] is ref
    assert args['resolution']==0 and args['negative_prompt']==''
    assert tuple(out['result'][2]['samples'].shape)==(1,64,2,4)
    if enabled:assert '<image2>: identity only' in captured[0]
    else:assert not captured and '<image2>: identity only' in args['prompt']


def test_complete_native_shard_merge_not_four_separate_encoders(tmp_path,monkeypatch):
    comfy=ModuleType('comfy');sd=ModuleType('comfy.sd');utils=ModuleType('comfy.utils')
    comfy.sd=sd;comfy.utils=utils
    sd.TEModel=SimpleNamespace(QWEN35_9B='qwen35');sd.CLIPType=SimpleNamespace(QWEN_IMAGE='qwen-image')
    sd.detect_te_model=Mock(return_value='qwen35');sd.load_text_encoder_state_dicts=Mock(return_value='model')
    utils.load_torch_file=lambda path,safe_load:{Path(path).stem:torch.zeros(1,dtype=torch.bfloat16)}
    for name,module in [('comfy',comfy),('comfy.sd',sd),('comfy.utils',utils)]:monkeypatch.setitem(sys.modules,name,module)
    files={f'model-{i:05d}.safetensors':tmp_path/f'model-{i:05d}.safetensors' for i in range(1,5)}
    index=tmp_path/'index.json';index.write_text(json.dumps({'weight_map':{Path(name).stem:name for name in files}}));files['model.safetensors.index.json']=index
    assert pe._load_model(files)=='model'
    args,kw=sd.load_text_encoder_state_dicts.call_args
    assert len(args[0])==1 and len(args[0][0])==4
    assert all(t.dtype==torch.bfloat16 for t in args[0][0].values())
    assert kw=={'clip_type':'qwen-image','model_options':{},'disable_dynamic':True}


def test_every_qwen_graph_has_shared_off_control():
    addon = ROOT / 'workflows/qwen/QGS1_Green_Suit_Overlay/Green_Suit_Qwen_to_GPT25_Overlay_v1_0.json'
    assert addon.is_file()

    workflows = [
        path for path in (ROOT / 'workflows/qwen').rglob('*.json')
        if path != addon
    ]
    assert len(workflows) == 35
    total=0
    for path in workflows:
        graph=json.loads(path.read_text());nodes={n['id']:n for n in graph['nodes']}
        control=[n for n in nodes.values() if n['type']=='WBQwenPromptControl'];assert len(control)==1
        assert control[0]['widgets_values']==[False,42,'']
        links={l[0]:l for l in graph['links']}
        for n in nodes.values():
            if n['type'] not in ('Q21NEncodeReferences','Q21PhotoEncode'):continue
            port=next(p for p in n['inputs'] if p['name']=='prompt_enhancer')
            assert links[port['link']][1]==control[0]['id'];total+=1
    assert total==40

def test_standard_hq_retains_off_baseline():
    sq=next(t for t in config.read_catalog('tasks.json')['tasks'] if t['id']=='SQ')
    graph=json.loads((ROOT/sq['variants'][0]['file']).read_text())
    assert next(n for n in graph['nodes'] if n['type']=='WBQwenPromptControl')['widgets_values'][0] is False

def test_only_qwen_downloads_prompt_enhancer_by_default(monkeypatch):
    for family in config.FAMILIES:
        monkeypatch.setenv('WB_WORKSPACE',family);monkeypatch.delenv('WB_TASKS',raising=False);monkeypatch.delenv('WB_TOOLBOX',raising=False)
        assert (pe.GROUP in config.requested_groups())==(family=='qwen')
        assert ('qwen_prompt' in config.model_config()['local_nodes'])==(family=='qwen')

def test_public_companions_do_not_relax_private_model_rules():
    with pytest.raises(assets.pd.LinkError):
        assets.pd.resolve_hf('https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/resolve/main/system_prompt.txt',Mock())

def test_asset_rows_pin_original_full_precision():
    rows=[a for a in config.read_catalog('assets.json')['assets'] if a['group']==pe.GROUP]
    assert len(rows)==6
    assert all(a['repo_id']==pe.MODEL_ID and a['revision']==pe.REVISION for a in rows)
    assert len([a for a in rows if a.get('sha256') and len(a['sha256'])==64])==4
    assert sum(a['estimated_bytes'] for a in rows)>18_000_000_000


def test_native_generation_uses_official_sampling_and_frees_residency(tmp_path,monkeypatch):
    comfy=ModuleType('comfy');mm=ModuleType('comfy.model_management');comfy.model_management=mm
    mm.unload_all_models=Mock();mm.soft_empty_cache=Mock()
    monkeypatch.setitem(sys.modules,'comfy',comfy);monkeypatch.setitem(sys.modules,'comfy.model_management',mm)
    system=tmp_path/'system.txt';system.write_text('Official fixture instructions.')
    monkeypatch.setattr(pe,'_installed_assets',lambda:{'system_prompt.txt':system})
    clip=Mock();clip.tokenize.return_value='tokens';clip.generate.return_value='ids';clip.decode.return_value=answer()
    monkeypatch.setattr(pe,'_load_model',lambda files:clip)
    images=[torch.zeros(1,32,64,3),torch.zeros(1,64,32,3)]
    assert pe._generate('edit',images,pe.Options(True,123,''))==answer()
    kwargs=clip.tokenize.call_args.kwargs
    assert kwargs['images'] is images and kwargs['thinking'] is True
    assert kwargs['system_prompt'].startswith('Official fixture instructions.')
    assert clip.generate.call_args.kwargs=={'do_sample':True,'max_length':24000,'temperature':1.0,
        'top_k':20,'top_p':.95,'min_p':0.0,'repetition_penalty':1.0,'presence_penalty':0.0,'seed':123,'mtp':False}
    assert mm.unload_all_models.call_count==2 and mm.soft_empty_cache.call_count==1


def test_failed_native_generation_still_releases_residency(tmp_path,monkeypatch):
    comfy=ModuleType('comfy');mm=ModuleType('comfy.model_management');comfy.model_management=mm
    mm.unload_all_models=Mock();mm.soft_empty_cache=Mock()
    monkeypatch.setitem(sys.modules,'comfy',comfy);monkeypatch.setitem(sys.modules,'comfy.model_management',mm)
    system=tmp_path/'system.txt';system.write_text('Official fixture instructions.')
    monkeypatch.setattr(pe,'_installed_assets',lambda:{'system_prompt.txt':system})
    monkeypatch.setattr(pe,'_load_model',Mock(side_effect=RuntimeError('load failure')))
    with pytest.raises(RuntimeError,match='load failure'):pe._generate('edit',[],pe.Options(True))
    assert mm.unload_all_models.call_count==2 and mm.soft_empty_cache.call_count==1


def test_asset_preparation_routes_only_allowed_companions(tmp_path,monkeypatch):
    row={'id':'test-companion','group':pe.GROUP,'format':'utf-8','destination':'prompt_enhancers/test/system_prompt.txt','revision':pe.REVISION}
    monkeypatch.setenv('WB_DATA_ROOT',str(tmp_path))
    monkeypatch.setattr(assets,'read_catalog',lambda _: {'assets':[row]})
    def prepare(asset,target,transport):target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture');return {'format':'utf-8'}
    monkeypatch.setattr(companions,'prepare_companion',prepare)
    mock=Mock(side_effect=AssertionError('Companion files do not use the model-only URL resolver'))
    monkeypatch.setattr(assets,'resolve_standard',mock)
    result=assets.prepare_standard([pe.GROUP])
    assert not result['failures'] and result['installed'][0]['resolved_revision']==pe.REVISION
    mock.assert_not_called()
