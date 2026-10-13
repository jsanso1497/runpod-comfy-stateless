"""CPU contract tests. These do not certify model inference, VRAM or image quality."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture
def krea(monkeypatch,tmp_path):
    from workbench import registry
    folder=types.ModuleType('folder_paths')
    folder.folder_names_and_paths={}
    folder.get_input_directory=lambda: str(tmp_path)
    folder.get_annotated_filepath=lambda name: str(tmp_path/name)
    folder.exists_annotated_filepath=lambda name: (tmp_path/name).is_file()
    n=types.ModuleType('nodes');n.NODE_CLASS_MAPPINGS={}
    mm=types.ModuleType('comfy.model_management')
    mm.unload_all_models=lambda: None;mm.soft_empty_cache=lambda: None
    comfy=types.ModuleType('comfy');comfy.model_management=mm
    for name,module in [('folder_paths',folder),('nodes',n),('comfy',comfy),('comfy.model_management',mm)]:
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(registry,'choices',lambda family,kind: ['krea2_identity_edit_v1_2.safetensors','custom.safetensors'] if kind=='loras' else ['test.safetensors'])
    monkeypatch.setattr(registry,'check_selection',lambda *args: args[-1])
    return load_module('krea_edit_test_module',ROOT/'src/nodes/krea_edit/__init__.py')

@pytest.mark.parametrize('size',[(2000,3000),(3000,2000),(1024,1024),(4000,6000),(64,64)])
def test_canvas_is_bounded(krea,size):
    w,h=krea.canvas_size(*size,1.9)
    assert w*h<=1_900_000 and w%16==h%16==0
    assert w<=size[0] and h<=size[1]

@pytest.mark.parametrize('size,mp',[((1,1),1.9),((2000,3000),3.0),((64,100000),.25)])
def test_invalid_canvas_rejected(krea,size,mp):
    with pytest.raises(ValueError):krea.canvas_size(*size,mp)

@pytest.mark.parametrize('count',range(5))
def test_zero_to_four_references(krea,count):
    source=torch.rand(1,96,64,3)
    refs=[{'image':torch.rand(1,80+i,64,3),'role':f'role {i}'} for i in range(count)]
    package,preview=krea.WBKreaPrepare().prepare(source,1.9,**{f'reference_{i+1}':r for i,r in enumerate(refs)})
    assert package['source'] is source and len(package['references'])==count
    if not count:assert package['board'] is None and preview is source
    elif count==1:assert package['board'] is refs[0]['image']
    else:
        assert package['board'].shape[2]==1408
        for i in range(count):assert f'R{i+1}' in package['roles']


def test_disabled_reference_needs_no_file(krea):
    assert krea.WBKreaReference.VALIDATE_INPUTS(False,'none') is True
    assert krea.WBKreaReference().load(False,'none','face')==(None,)
    assert krea.WBKreaReference.VALIDATE_INPUTS(True,'none') is not True
    with pytest.raises(ValueError):krea.WBKreaReference().load(True,'none','face')


def package(krea,count=1):
    image=torch.zeros(1,64,64,3)
    refs={f'reference_{i+1}':{'image':image,'role':'face '+str(i)} for i in range(count)}
    return krea.WBKreaPrepare().prepare(image,1.9,**refs)[0]


def test_enhancer_off_does_not_call_network(krea,monkeypatch):
    monkeypatch.setattr(krea.urllib.request,'urlopen',lambda *a,**k: pytest.fail('Unexpected network access'))
    result=krea.WBKreaPrompt().enhance_prompt(package(krea),'Change only the coat.',False,krea.MODEL,'EXACT_trigger',768)
    text=result['result'][0]
    assert 'Change only the coat.' in text and text.endswith('EXACT_trigger')


def test_enhancer_sees_individual_images_and_unloads(krea,monkeypatch):
    events=[];request_data={}
    monkeypatch.setattr(krea.mm,'unload_all_models',lambda: events.append('unload'))
    monkeypatch.setattr(krea.mm,'soft_empty_cache',lambda: events.append('cache'))
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,*args):return json.dumps({'message':{'content':'Change only the coat to red.'},'done_reason':'stop'}).encode()
    def request(req,**kwargs):
        events.append('request');request_data.update(json.loads(req.data));return Response()
    monkeypatch.setattr(krea.urllib.request,'urlopen',request)
    result=krea.WBKreaPrompt().enhance_prompt(package(krea,4),'Red coat',True,krea.MODEL,'Token_Case_123',768)
    assert events==['unload','cache','request']
    assert request_data['model']=='qwen3.8:27b-bf16'
    assert request_data['keep_alive']==0 and request_data['think'] is False
    assert len(request_data['messages'][1]['images'])==5
    assert result['result'][0].endswith('Token_Case_123')
    assert result['ui']['text'][0]==result['result'][0]

@pytest.mark.parametrize('response',[{'error':'not found'},{'done_reason':'length','message':{'content':'cut'}},{'message':{'content':''}},{'message':{'content':'<think>unfinished'}}])
def test_bad_enhancer_response_is_not_silently_used(krea,response):
    with pytest.raises(RuntimeError):krea.clean_response(response)


def test_thinking_is_not_forwarded(krea):
    assert krea.clean_response({'message':{'content':'<think>reasoning</think>\nEdit only the hat.'}})=='Edit only the hat.'


def test_dual_grounding_and_target_latent_wiring(krea):
    calls=[];captured={}
    class Empty:
        def generate(self,w,h,batch):return ({'samples':torch.zeros(batch,16,h//8,w//8)},)
    class Encode:
        def encode(self,clip,prompt,**kwargs):calls.append((prompt,kwargs));return ('conditioning:'+prompt,)
    class Patch:
        def patch(self,model,**kwargs):captured.update(kwargs);return ('patched',)
    krea.nodes.NODE_CLASS_MAPPINGS.update(Krea2EditModelPatch=Patch,Krea2EditGroundedEncode=Encode,EmptySD3LatentImage=Empty)
    p=package(krea,2)
    result=krea.WBKreaCondition().condition('model','clip','vae',p,'edit',1.0,4.0,768)
    assert [c[0] for c in calls]==['edit','']
    assert calls[0][1]['image'] is calls[1][1]['image'] is p['source']
    assert calls[0][1]['image_b'] is calls[1][1]['image_b'] is p['board']
    assert captured['target_latent'] is result[-1]
    assert captured['source_image'] is p['source'] and captured['source_image_b'] is p['board']
    assert captured['fit_mode']=='fit'


def test_required_edit_adapter_and_optional_loras(krea):
    calls=[]
    class Lora:
        def load_lora_model_only(self,model,name,strength):calls.append((name,strength));return (model,)
    krea.nodes.LoraLoaderModelOnly=Lora
    krea.WBKreaLoRAs().apply('model',krea.EDIT_ADAPTER,1,'custom.safetensors',.7,'none',1)
    assert calls==[(krea.EDIT_ADAPTER,1),('custom.safetensors',.7)]
    with pytest.raises(ValueError):krea.WBKreaLoRAs().apply('m',krea.EDIT_ADAPTER,0,'none',1,'none',1)
    with pytest.raises(ValueError):krea.WBKreaLoRAs().apply('m','custom.safetensors',1,'none',1,'none',1)
    with pytest.raises(ValueError):krea.WBKreaLoRAs().apply('m',krea.EDIT_ADAPTER,1,krea.EDIT_ADAPTER,1,'none',1)

@pytest.mark.parametrize('checkpoint,preset,expected',[
 ('krea2_raw_bf16.safetensors','Auto',(20,3.0)),
 ('krea2_turbo_bf16.safetensors','Auto',(8,1.0)),
 ('private.safetensors','Raw',(20,3.0)),('private.safetensors','Custom',(24,2.5))])
def test_checkpoint_preset_and_correct_encoder(krea,checkpoint,preset,expected):
    calls=[]
    model=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace(_unpack_context=True)))
    class UNET:
        def load_unet(self,*args):return (model,)
    class CLIP:
        def load_clip(self,name,type):calls.append(type);return ('clip',)
    class VAE:
        def load_vae(self,name):return ('vae',)
    krea.nodes.UNETLoader=UNET;krea.nodes.CLIPLoader=CLIP;krea.nodes.VAELoader=VAE
    result=krea.WBKreaModels().load('ready',checkpoint,'encoder','vae',preset,24,2.5)
    assert result[-2:]==expected and calls==['krea2']


def test_custom_checkpoint_requires_sampling_choice(krea):
    with pytest.raises(ValueError,match='explicitly'):
        krea.WBKreaModels().load('ready','private.safetensors','encoder','vae','Auto',20,3)


def test_graph_matches_generator_and_has_valid_links():
    generator=load_module('krea_generator',ROOT/'scripts/build_krea_workflow.py')
    graph,api=generator.build()
    assert graph==json.loads((ROOT/generator.UI).read_text())
    assert api==json.loads((ROOT/generator.API).read_text())
    validator=load_module('krea_validator',ROOT/'scripts/validate.py')
    assert validator.validate_graph(graph)==[]
    byid={str(n['id']):n for n in graph['nodes']}
    for id,node in api.items():
        assert node['class_type']==byid[id]['type']
        for value in node['inputs'].values():
            if isinstance(value,list):assert value[0] in byid
    assert api['8']['inputs']['ready_prompt']==['7',0]
    assert api['11']['inputs']['steps']==['8',3]
    assert api['11']['inputs']['cfg']==['8',4]


def test_catalog_merges_without_replacing_other_families(tmp_path,monkeypatch):
    from workbench import config
    catalog=tmp_path/'catalog';catalog.mkdir()
    for name,value in {
        'models.json':{'models':[{'id':'qwen'}],'external_nodes':{}},
        'assets.json':{'assets':[],'ollama':{'models':['existing-f16-a','existing-f16-b'],'estimated_bytes':1}},
        'tasks.json':{'tasks':[{'id':'Q01','workspaces':['qwen']}]}}.items():
        (catalog/name).write_text(json.dumps(value))
    (catalog/'krea.json').write_text((ROOT/'catalog/krea.json').read_text())
    monkeypatch.setattr(config,'ROOT',tmp_path)
    monkeypatch.setenv('WB_WORKSPACE','qwen')
    assets=config.read_catalog('assets.json')
    assert assets['ollama']['models']==['huihui_ai/Qwen3.8-abliterated:27b-bf16']
    assert assets['ollama_by_workspace']['h3']['models']==['existing-f16-a','existing-f16-b']
    assert [m['id'] for m in config.read_catalog('models.json')['models']]==['qwen','krea']
    assert [t['id'] for t in config.read_catalog('tasks.json')['tasks']]==['Q01','K01']
    monkeypatch.setenv('WB_WORKSPACE','krea')
    assert config.read_catalog('assets.json')['ollama']['models']==['qwen3.8:27b-bf16']
    assert config.selected_tasks()==['K01'] and config.requested_groups()==['krea','ollama']


def test_model_hashes_and_pins_are_present():
    import re
    addon=json.loads((ROOT/'catalog/krea.json').read_text())
    assert len(addon['assets'])==5
    assert all(re.fullmatch('[0-9a-f]{64}',a['sha256']) for a in addon['assets'])
    assert re.fullmatch('[0-9a-f]{40}',addon['model']['comfy_commit'])
    assert addon['ollama']['models']==['qwen3.8:27b-bf16']


def test_standard_condition_has_no_unused_required_vae(krea):
    inputs=krea.WBKreaStandardCondition.INPUT_TYPES()['required']
    assert set(inputs)=={'model','clip','editing_inputs','prompt'}
    class Empty:
        def generate(self,w,h,b):return ({'samples':torch.zeros(b,16,h//8,w//8)},)
    class Encode:
        def encode(self,clip,prompt):return ('conditioning:'+prompt,)
    krea.nodes.NODE_CLASS_MAPPINGS.update(EmptySD3LatentImage=Empty,CLIPTextEncode=Encode)
    result=krea.WBKreaStandardCondition().condition('model','clip',package(krea),'A photograph.')
    assert result[:3]==('model','conditioning:A photograph.','conditioning:')


def test_standard_optional_loras_do_not_require_or_allow_edit_adapter(krea):
    assert krea.EDIT_ADAPTER not in krea.WBKreaOptionalLoRAs.INPUT_TYPES()['required']['lora_1'][0]
    calls=[]
    class Lora:
        def load_lora_model_only(self,model,name,strength):calls.append((name,strength));return (model,)
    krea.nodes.LoraLoaderModelOnly=Lora
    assert krea.WBKreaOptionalLoRAs().apply('model','none',1,'none',1)==('model',)
    assert calls==[]
    krea.WBKreaOptionalLoRAs().apply('model','custom.safetensors',.6,'none',1)
    assert calls==[('custom.safetensors',.6)]
    with pytest.raises(ValueError):krea.WBKreaOptionalLoRAs().apply('m',krea.EDIT_ADAPTER,1,'none',1)


def test_standard_prompt_off_is_standalone(krea,monkeypatch):
    monkeypatch.setattr(krea.urllib.request,'urlopen',lambda *a,**k:pytest.fail('network'))
    out=krea.WBKreaStandardPrompt().enhance_prompt(package(krea),'A photograph of a red coat.',False,krea.MODEL,'Token',768)
    assert out['result'][0]=='A photograph of a red coat.\nToken'
    assert 'first image' not in out['result'][0]


def test_standard_generator_and_api_custom_input_contracts(krea):
    generator=load_module('standard_generator',ROOT/'scripts/build_krea_workflow.py')
    graph,api=generator.build_standard()
    assert graph==json.loads((ROOT/generator.UI_STANDARD).read_text())
    assert api==json.loads((ROOT/generator.API_STANDARD).read_text())
    for row in api.values():
        if row['class_type'] not in krea.NODE_CLASS_MAPPINGS:continue
        required=krea.NODE_CLASS_MAPPINGS[row['class_type']].INPUT_TYPES().get('required',{})
        assert not (set(required)-set(row['inputs'])),row['class_type']
    assert api['10']['inputs']['model']==['9',0]
    assert api['9']['class_type']=='WBKreaOptionalLoRAs'
