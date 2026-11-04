"""Regression coverage: RunPod Host rewriting must not block WebSocket auth."""
from __future__ import annotations
import asyncio
import base64
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gateway import same_origin, create_app

POD = 'nzws0nv09ey6q7'
PUBLIC = f'https://{POD}-8188.proxy.runpod.net'


def test_own_pod_origin_matches_when_runpod_rewrites_host():
    assert same_origin(PUBLIC, '127.0.0.1:8188', pod_id=POD)
    assert same_origin(PUBLIC, 'internal-runpod-host:8188', pod_id=POD)
    assert same_origin(PUBLIC, '127.0.0.1:8188', public_origin=PUBLIC)
    assert same_origin('https://pod.example', 'pod.example')
    assert same_origin('', '127.0.0.1:8188')

@pytest.mark.parametrize('origin', [
    'https://attacker.example',
    'https://otherpod-8188.proxy.runpod.net',
    'http://nzws0nv09ey6q7-8188.proxy.runpod.net',
    'https://nzws0nv09ey6q7-8188.proxy.runpod.net.evil.example',
    'https://nzws0nv09ey6q7-8188.proxy.runpod.net:443',
    'https://nzws0nv09ey6q7-8188.proxy.runpod.net/path',
    'null',
    'https://attacker.example@nzws0nv09ey6q7-8188.proxy.runpod.net',
    'https://nzws0nv09ey6q7-8188.proxy.runpod.net?bad=x',
])
def test_other_origins_rejected(origin):
    assert not same_origin(origin, '127.0.0.1:8188', pod_id=POD)


def test_actual_gateway_websocket_proxy_with_rewritten_host(monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    monkeypatch.setenv('RUNPOD_POD_ID', POD)
    password = 'test-password-not-for-production'
    token = base64.b64encode(f'workbench:{password}'.encode()).decode()
    async def echo(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await ws.send_str(msg.data)
        return ws
    async def check():
        upstream = web.Application()
        upstream.router.add_get('/ws', echo)
        # Gateway has a pinned loopback upstream port of 8189.
        async with TestServer(upstream, host='127.0.0.1', port=8189):
            async with TestClient(TestServer(create_app(password))) as client:
                headers = {'Authorization': 'Basic ' + token, 'Origin': PUBLIC}
                async with client.ws_connect('/ws?clientId=regression', headers=headers) as ws:
                    await ws.send_str('hello')
                    response = await ws.receive(timeout=5)
                    assert response.data == 'hello'
                bad = dict(headers, Origin='https://attacker.example')
                response = await client.get('/ws?clientId=bad', headers=bad)
                assert response.status == 403
                response = await client.get('/ws?clientId=unauthorized', headers={'Origin':PUBLIC})
                assert response.status == 401
    asyncio.run(check())
