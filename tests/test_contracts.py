import importlib.util,json,os,struct,hashlib
from pathlib import Path
import pytest,torch
from workbench import config,assets,registry
from workbench.runtime import authorized,make_app
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
tools=module('workbench_test_tools',ROOT/'src/nodes/workbench_tools/__init__.py')
def test_contract():
 validate=module('validator',ROOT/'scripts/validate.py');assert not validate.run()['errors']
@pytest.mark.parametrize('family',['qwen','flux','h3','restoration'])
def test_defaults(family,monkeypatch):
 monkeypatch.setenv('WB_WORKSPACE',family);monkeypatch.delenv('WB_TASKS',raising=False);monkeypatch.delenv('WB_TOOLBOX',raising=False)
 assert config.selected_tasks();assert config.requested_groups()
@pytest.mark.parametrize('value',['../a','/tmp/a','foo/../../a','a\\b','\x00',''])
def test_safe_paths(tmp_path,value):
 with pytest.raises(ValueError):config.safe_path(tmp_path,value)
def test_symlink_rejected(tmp_path):
 (tmp_path/'link').symlink_to('/tmp');
 with pytest.raises(ValueError):config.safe_path(tmp_path,'link/private')
def test_secret_mapping(monkeypatch):
 monkeypatch.delenv('HF_TOKEN',raising=False);monkeypatch.delenv('CIVITAI_TOKEN',raising=False);monkeypatch.setenv('hf_token','unit-test-secret');monkeypatch.setenv('civit_token','other-unit-test-secret')
 config.resolve_secret_aliases();assert os.environ['HF_TOKEN']=='unit-test-secret';assert os.environ['CIVITAI_TOKEN']=='other-unit-test-secret';assert 'HF_TOKEN' not in config.child_environment();assert 'civit_token' not in config.child_environment()
def test_secret_placeholder_rejected(monkeypatch):
 monkeypatch.setenv('HF_TOKEN','{{ RUNPOD_SECRET_hf_token }}')
 with pytest.raises(ValueError):config.resolve_secret_aliases()
def test_library_file_not_opened_when_env_present(monkeypatch):
 monkeypatch.setenv('WB_LORA_URLS','qwen | https://huggingface.co/example/repo/resolve/main/adapter.safetensors');monkeypatch.setenv('WB_LORA_FILE','/this/file/must/not/be/read.txt')
 assert 'https://' in assets.library_value('WB_LORA_URLS','WB_LORA_FILE')
@pytest.mark.parametrize('value',['https://example.com/a.safetensors','http://huggingface.co/a/b','https://huggingface.co/a/b?token=secret','https://user:pass@huggingface.co/a/b','qwen | https://huggingface.co/a/b | extra'])
def test_private_invalid_urls(value):
 with pytest.raises(ValueError):assets.parse_library(value,'loras')
def test_lora_family_prefix():
 a=assets.parse_library('qwen | https://huggingface.co/example/adapter/resolve/main/a.safetensors','loras');assert a[0]['family']=='qwen';assert a[0]['kind']=='loras'
def test_unknown_family():
 with pytest.raises(ValueError):assets.parse_library('other | https://huggingface.co/example/repo','loras')
def test_checkpoint_declaration():
 rows=assets.parse_library(json.dumps([{'url':'https://huggingface.co/example/repo/resolve/main/model.safetensors','family':'qwen','architecture':'qwen-image-2.1','kind':'diffusion_models','precision':'bf16'}]),'diffusion_models');assert rows[0]['architecture']=='qwen-image-2.1'
@pytest.mark.parametrize('precision',['int8','fp8','gguf','q4'])
def test_quantized_manifest_rejected(precision):
 with pytest.raises(ValueError):assets.parse_library(json.dumps([{'url':'https://huggingface.co/example/repo','precision':precision}]),'diffusion_models')
def safetensors(path,dtype='F32',payload=None):
 if payload is None:payload=b'\0'*({'F32':4,'F16':2,'BF16':2,'I8':1}[dtype])
 h=json.dumps({'tensor':{'dtype':dtype,'shape':[1],'data_offsets':[0,len(payload)]}}).encode();path.write_bytes(struct.pack('<Q',len(h))+h+payload)
@pytest.mark.parametrize('dtype',['F32','F16','BF16'])
def test_full_precision_header(tmp_path,dtype):
 p=tmp_path/'test.safetensors';safetensors(p,dtype);assert assets.inspect_safetensors(p,high_precision=True)['dtypes']==[dtype]
def test_quantized_header(tmp_path):
 p=tmp_path/'test.safetensors';safetensors(p,'I8')
 with pytest.raises(assets.AssetError):assets.inspect_safetensors(p,high_precision=True)
def test_truncated_header(tmp_path):
 p=tmp_path/'test.safetensors';safetensors(p);p.write_bytes(p.read_bytes()[:-1])
 with pytest.raises(assets.AssetError):assets.inspect_safetensors(p)
def test_registry_requires_declaration(tmp_path,monkeypatch):
 monkeypatch.setenv('WB_DATA_ROOT',str(tmp_path))
 with pytest.raises(ValueError):registry.check_selection('qwen','diffusion_models','unknown.safetensors')
def test_registry_choices(tmp_path,monkeypatch):
 monkeypatch.setenv('WB_DATA_ROOT',str(tmp_path));config.atomic_json(config.state_root()/'private/assets.json',{'assets':[{'family':'qwen','kind':'loras','filename':'Private/qwen/test.safetensors','compatibility':'declared','status':'ready'}]},private=True)
 assert registry.lora_choices('qwen')==['Private/qwen/test.safetensors'];assert not registry.lora_choices('flux')
def test_empty_mask_stays_empty():assert torch.equal(tools.feather_mask(torch.zeros(1,64,64),20,30),torch.zeros(1,64,64))
def test_feather_radius():
 m=torch.zeros(1,101,101);m[:,50,50]=1;f=tools.feather_mask(m,10,20)
 assert f[0,50,60]==1;assert 0<f[0,50,70]<1;assert f[0,50,80]==0;assert f[0,50,90]==0
@pytest.mark.parametrize('expand,feather',[(0,0),(2,0),(0,8),(4,16),(10,30)])
def test_exact_untouched_pixels(expand,feather):
 a=torch.rand(1,80,80,3);b=torch.rand_like(a);m=torch.zeros(1,80,80);m[:,30:40,30:40]=1;f=tools.feather_mask(m,expand,feather)
 out,report=tools.WBComposite().composite(a,b,f)
 assert torch.equal(out[f==0],a[f==0]);assert torch.equal(out[f==1],b[f==1]);assert json.loads(report)['outside_selection_max_error']==0
@pytest.mark.parametrize('invert',[True,False])
def test_inversion(invert):
 m=torch.rand(1,10,10);result=tools.feather_mask(m,0,0,invert);assert torch.equal(result,1-m if invert else m)
def test_protection_after_feather():
 a=torch.rand(1,40,40,3);b=torch.rand_like(a);m=torch.ones(1,40,40);p=torch.zeros_like(m);p[:,:20]=.01
 out,_=tools.WBComposite().composite(a,b,m,p);assert torch.equal(out[:,:20],a[:,:20]);assert torch.equal(out[:,20:],b[:,20:])
def test_mismatched_images_rejected():
 with pytest.raises(ValueError):tools.WBComposite().composite(torch.zeros(1,4,4,3),torch.zeros(1,5,5,3),torch.ones(1,4,4))
def test_nonfinite_mask_rejected():
 with pytest.raises(ValueError):tools.feather_mask(torch.full((1,3,3),float('nan')),0,0)
def test_password_auth():
 import base64
 header='Basic '+base64.b64encode(b'workbench:test-password-value').decode();assert authorized(header,'test-password-value');assert not authorized(header,'wrong');assert not authorized('Bearer test','test-password-value')
@pytest.mark.parametrize('header',['','Basic ???','Basic YQ==','Digest xyz'])
def test_bad_auth(header):assert not authorized(header,'test-password-value')
def test_password_required():
 with pytest.raises(ValueError):make_app('short')
def test_h3_still_no_lite_gate():
 s=(ROOT/'src/nodes/h3_portrait/__init__.py').read_text();assert "installed only in the H3 Portrait Lite" not in s;assert "cfg.get('profile')!='full'" in s

def test_adapter_rejects_base_checkpoint(tmp_path):
 from workbench.provider_downloads import check_adapter_file,LinkError
 p=tmp_path/'base.safetensors';safetensors(p)
 with pytest.raises(LinkError):check_adapter_file(p)
def test_shape_bytes_validated(tmp_path):
 p=tmp_path/'bad-shape.safetensors';safetensors(p,'F16',payload=b'\0'*4)
 with pytest.raises(assets.AssetError):assets.inspect_safetensors(p)
def test_linked_loader_is_validated_at_execution():
 assert tools.SafeH3UNETLoader.VALIDATE_INPUTS(weight_dtype='default') is True

def test_source_api_graphs_match_ui():
 index=json.loads((ROOT/'automation/index.json').read_text())['exports'];assert len(index)==42
 for row in index:
  ui={str(n['id']):n for n in json.loads((ROOT/row['ui']).read_text())['nodes']};api=json.loads((ROOT/row['api']).read_text())
  for id,node in api.items():
   assert id in ui;assert node['class_type']==ui[id]['type']
   for value in node['inputs'].values():
    if isinstance(value,list) and len(value)==2 and isinstance(value[1],int):assert str(value[0]) in api

def test_gateway_auth_and_origin():
 import asyncio,base64
 from aiohttp import web
 from aiohttp.test_utils import TestClient,TestServer
 from workbench.runtime import security
 async def check():
  app=web.Application(middlewares=[security]);app['password']='test-password-value'
  app.router.add_route('*','/{path:.*}',lambda request:web.json_response({'ok':True}))
  async with TestClient(TestServer(app)) as client:
   response=await client.get('/healthz');assert response.status==200
   response=await client.get('/_workbench/catalog');assert response.status==401
   auth='Basic '+base64.b64encode(b'workbench:test-password-value').decode()
   response=await client.get('/_workbench/catalog',headers={'Authorization':auth});assert response.status==200
   response=await client.post('/_workbench/prepare',headers={'Authorization':auth,'Origin':'https://untrusted.example'});assert response.status==403
   response=await client.get('/ws',headers={'Authorization':auth,'Origin':'https://untrusted.example','Upgrade':'websocket'});assert response.status==403
 asyncio.run(check())

def test_private_library_skips_other_families(tmp_path,monkeypatch):
 monkeypatch.setenv('WB_WORKSPACE','qwen');monkeypatch.setenv('WB_DATA_ROOT',str(tmp_path));monkeypatch.setenv('WB_LORA_URLS','h3 | https://huggingface.co/example/repo/resolve/main/adapter.safetensors');monkeypatch.delenv('WB_CHECKPOINTS',raising=False)
 def fail(*args,**kwargs):raise AssertionError('An unrelated family must not be requested.')
 monkeypatch.setattr(assets.pd,'resolve_hf',fail)
 result=assets.prepare_private();assert not result['assets'];assert not result['failures']
