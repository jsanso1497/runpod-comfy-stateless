import hashlib
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from download_models import download_one,load_assets,safe_destination

DATA=b'validated test model content'*100

def asset():
    return {'id':'test','profile':'identity','repo':'owner/repo','revision':'main','file':'test.safetensors',
        'destination':'loras/test.safetensors','sha256':hashlib.sha256(DATA).hexdigest(),'approx_bytes':len(DATA)}

class Response:
    def __init__(self,data,status=200,headers=None):self.data=data;self.status_code=status;self.headers=headers or {}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass
    def iter_content(self,chunk_size):
        for i in range(0,len(self.data),17):yield self.data[i:i+17]
class Session:
    def __init__(self,response):self.response=response;self.headers={};self.calls=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def get(self,url,**kwargs):self.calls.append((url,kwargs));return self.response

def test_only_explicit_profile_assets():
    a=load_assets(ROOT/'config/models.json','identity')
    b=load_assets(ROOT/'config/models.json','upscale')
    assert len(a)==5 and len(b)==7
    assert not any('seedvr' in x['destination'].lower() for x in a)
    assert {x['id'] for x in a} <= {x['id'] for x in b}

def test_fresh_download_atomic(tmp_path):
    s=Session(Response(DATA));a=asset()
    p=download_one(a,tmp_path,session_factory=lambda:s)
    assert p.read_bytes()==DATA
    assert not p.with_name(p.name+'.part').exists()

def test_resume_correct_range(tmp_path):
    a=asset();p=tmp_path/a['destination'];p.parent.mkdir();part=p.with_name(p.name+'.part');part.write_bytes(DATA[:20])
    s=Session(Response(DATA[20:],206,{'Content-Range':f'bytes 20-{len(DATA)-1}/{len(DATA)}'}))
    assert download_one(a,tmp_path,session_factory=lambda:s).read_bytes()==DATA
    assert s.calls[0][1]['headers']['Range']=='bytes=20-'

def test_range_ignored_restarts_not_appends(tmp_path):
    a=asset();p=tmp_path/a['destination'];p.parent.mkdir();p.with_name(p.name+'.part').write_bytes(DATA[:20])
    s=Session(Response(DATA))
    assert download_one(a,tmp_path,session_factory=lambda:s).read_bytes()==DATA

def test_bad_hash_not_promoted(tmp_path):
    s=Session(Response(b'bad'))
    with pytest.raises(RuntimeError,match='SHA-256 mismatch'):download_one(asset(),tmp_path,session_factory=lambda:s)
    assert not (tmp_path/asset()['destination']).exists()

def test_existing_corrupt_file_not_overwritten(tmp_path):
    a=asset();p=tmp_path/a['destination'];p.parent.mkdir();p.write_bytes(b'old and wrong')
    with pytest.raises(RuntimeError,match='Existing model failed'):download_one(a,tmp_path)
    assert p.read_bytes()==b'old and wrong'

def test_existing_verified_file_uses_no_network(tmp_path):
    a=asset();p=tmp_path/a['destination'];p.parent.mkdir();p.write_bytes(DATA)
    assert download_one(a,tmp_path,session_factory=lambda:pytest.fail('network used'))==p.resolve()

def test_complete_partial_promoted(tmp_path):
    a=asset();p=tmp_path/a['destination'];p.parent.mkdir();p.with_name(p.name+'.part').write_bytes(DATA)
    s=Session(Response(b''))
    assert download_one(a,tmp_path,session_factory=lambda:s).read_bytes()==DATA
    assert not s.calls

def test_invalid_range_rejected(tmp_path):
    s=Session(Response(DATA,206,{'Content-Range':f'bytes 100-{len(DATA)+99}/{len(DATA)+100}'}))
    with pytest.raises(RuntimeError,match='Invalid Content-Range'):download_one(asset(),tmp_path,session_factory=lambda:s)

def test_auth_error_clear(tmp_path):
    s=Session(Response(b'',403))
    with pytest.raises(PermissionError,match='HF_TOKEN'):download_one(asset(),tmp_path,session_factory=lambda:s)

@pytest.mark.parametrize('value',['../oops','/root/oops','loras/../../oops'])
def test_path_traversal_rejected(tmp_path,value):
    with pytest.raises(ValueError):safe_destination(tmp_path,value)
