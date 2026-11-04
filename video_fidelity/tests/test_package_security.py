import asyncio
import base64
import importlib.util
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gateway import authorized,validate_password,same_origin,create_app
from validate_package import validate


def test_package():validate()


def test_auth():
    pw='a-long-unique-password'
    header='Basic '+base64.b64encode(('workbench:'+pw).encode()).decode()
    assert authorized(header,pw)
    assert not authorized(header,'different-password')
    for bad in ('','Bearer abc','Basic !!','Basic '+base64.b64encode(b'admin:wrong').decode()):
        assert not authorized(bad,pw)


@pytest.mark.parametrize('password',['','short','{{ RUNPOD_SECRET_comfy_password }}'])
def test_weak_or_unresolved_password_rejected(password):
    with pytest.raises(ValueError):validate_password(password)


def test_origin_guard():
    assert same_origin('https://pod.proxy.runpod.net','pod.proxy.runpod.net')
    assert not same_origin('https://attacker.example','pod.proxy.runpod.net')
    assert not same_origin('null','pod.proxy.runpod.net')


def test_live_gateway_auth_and_health():
    from aiohttp.test_utils import TestClient,TestServer
    async def check():
        app=create_app('a-long-unique-password')
        async with TestClient(TestServer(app)) as client:
            assert (await client.get('/_health')).status==200
            response=await client.get('/object_info')
            assert response.status==401 and 'WWW-Authenticate' in response.headers
            token=base64.b64encode(b'workbench:a-long-unique-password').decode()
            response=await client.get('/object_info',headers={'Authorization':'Basic '+token,'Origin':'https://attacker.example'})
            assert response.status==403
    asyncio.run(check())


def test_node_schema_and_api_match(monkeypatch):
    monkeypatch.setenv('FAITHFUL_APP_ROOT',str(ROOT))
    spec=importlib.util.spec_from_file_location('faithful_nodes_test',ROOT/'nodes/__init__.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    schema=mod.FaithfulVideoRestore.INPUT_TYPES()['required']
    api=json.loads((ROOT/'api/01_Faithful_Video.api.json').read_text())['2']['inputs']
    assert set(api)==set(schema)
    assert schema['temporal_batch'][1]['min']==5
    assert mod.FaithfulVideoRestore.OUTPUT_NODE
