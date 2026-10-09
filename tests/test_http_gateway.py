"""CPU HTTP/WebSocket regression tests. No GPU, downloads, or provider requests.

Uses real local aiohttp servers and the shipped make_app factory. Runtime model
preparation and configuration are isolated fixtures, not production services.
"""
from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import uuid

import aiohttp
from aiohttp import web, WSMsgType
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request
from multidict import CIMultiDict

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = 'browser-regression-password'
HTML = b'<!doctype html><html><head><title>Gateway test</title></head><body><h1>ComfyUI transport fixture</h1><script src="/assets/test.js"></script></body></html>'
JAVASCRIPT = b'window.gatewayTestLoaded = true;'
PAYLOAD = bytes(range(256)) * 4096


def load_runtime(root: Path):
    """Isolate only config/model preparation; import the real transport/runtime."""
    name = '_wb_gateway_test_' + uuid.uuid4().hex
    package = ModuleType(name)
    package.__path__ = [str(ROOT / 'src/workbench')]
    sys.modules[name] = package
    config = ModuleType(name + '.config')
    config.ROOT = root
    config.workspace = lambda: 'qwen'
    config.data_root = lambda: root
    config.state_root = lambda: root / 'workbench'
    config.model_root = lambda: root / 'models'
    config.model_config = lambda: {'toolbox': ['sam', 'seedvr2']}
    config.child_environment = lambda: {}
    config.selected_tasks = lambda: ['QGS1']
    config.requested_groups = lambda: ['qwen', 'sam', 'seedvr2']
    config.resolve_secret_aliases = lambda: None
    config.safe_path = lambda base, relative: base / relative
    config.atomic_json = lambda *args, **kwargs: None
    catalogs = {
        'tasks.json': {'tasks': []}, 'models.json': {'models': []},
        'assets.json': {'assets': []}, 'release.json': {'version': 'test'},
    }
    config.read_catalog = lambda key: catalogs[key]
    sys.modules[config.__name__] = config
    assets = ModuleType(name + '.assets')
    assets.prepare_private = lambda: {'failures': []}
    assets.prepare_standard = lambda groups: {'failures': [], 'installed': []}
    sys.modules[assets.__name__] = assets
    import importlib
    runtime = importlib.import_module(name + '.runtime')
    return name, runtime, runtime.http_gateway


class Harness:
    async def start(self, mode='none', *, max_size=None):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'site').mkdir()
        (self.root/'site/index.html').write_bytes(HTML)
        self.namespace, self.runtime, self.gateway = load_runtime(self.root)
        self.captured = []
        self.up_app = web.Application(handler_args={'auto_decompress': False})
        self.up_app.router.add_route('*', '/{path:.*}', self.upstream)
        self.up = TestServer(self.up_app)
        await self.up.start_server()
        base = str(self.up.make_url('/')).rstrip('/')
        outer = self

        class FixtureRuntime(self.runtime.Runtime):
            async def startup(self, app):
                self.session = outer.gateway.session()
            async def cleanup(self, app):
                await self.session.close()
            async def proxy(self, request):
                return await outer.gateway.proxy(request, self.session, base)
            async def ready(self, request):
                return web.json_response({'comfy_ready': True})

        self.fixture = FixtureRuntime()
        self.app = self.runtime.make_app(PASSWORD if mode=='basic' else '',
                                         self.fixture, auth_mode=mode)
        if max_size is not None:
            self.app._client_max_size = max_size
        self.client = TestClient(TestServer(self.app), auto_decompress=False)
        await self.client.start_server()
        return self

    async def close(self):
        await self.client.close()
        await self.up.close()
        self.temp.cleanup()
        for name in list(sys.modules):
            if name == self.namespace or name.startswith(self.namespace + '.'):
                del sys.modules[name]

    async def upstream(self, request):
        self.captured.append({'path': request.raw_path,
                              'method': request.method,
                              'headers': dict(request.headers)})
        p = request.path
        if p == '/':
            # Exercise browser compression negotiation with a precompressed asset.
            if 'gzip' in request.headers.get('Accept-Encoding', ''):
                return web.Response(body=gzip.compress(HTML), content_type='text/html',
                                    headers={'Content-Encoding': 'gzip'})
            return web.Response(body=HTML, content_type='text/html')
        if p == '/assets/test.js':
            return web.Response(body=JAVASCRIPT, content_type='application/javascript')
        if p == '/json':
            return web.json_response({'ok': True})
        if p == '/gzip':
            return web.Response(body=gzip.compress(HTML), content_type='text/html',
                                headers={'Content-Encoding': 'gzip'})
        if p == '/empty':
            return web.Response(status=204)
        if p == '/not-modified':
            return web.Response(status=304, headers={'ETag': '"fixture"'})
        if p == '/redirect':
            raise web.HTTPFound('/json')
        if p == '/cookies':
            response = web.Response(text='two cookies')
            response.set_cookie('first', 'one', httponly=True)
            response.set_cookie('second', 'two', httponly=True)
            return response
        if p == '/stream':
            response = web.StreamResponse(headers={'Content-Type': 'application/octet-stream'})
            await response.prepare(request)
            for i in range(0, len(PAYLOAD), 65536):
                await response.write(PAYLOAD[i:i+65536])
            await response.write_eof()
            return response
        if p == '/view':
            if request.headers.get('Range') == 'bytes=10-29':
                return web.Response(status=206, body=PAYLOAD[10:30], headers={
                    'Content-Range': f'bytes 10-29/{len(PAYLOAD)}',
                    'Accept-Ranges': 'bytes', 'Content-Type': 'application/octet-stream'})
            return web.Response(body=PAYLOAD)
        if p == '/upload/image':
            fields = await request.multipart()
            image = await fields.next()
            data = await image.read()
            return web.json_response({'name': image.filename, 'bytes': len(data),
                                      'sha256': hashlib.sha256(data).hexdigest()})
        if p == '/ws':
            ws = web.WebSocketResponse(protocols=('json',), compress=False)
            await ws.prepare(request)
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    if msg.data == 'close-from-server':
                        await ws.close(code=1000)
                        break
                    await ws.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await ws.send_bytes(msg.data)
            return ws
        data = await request.read()
        return web.json_response({'method': request.method, 'bytes': len(data),
                                  'sha256': hashlib.sha256(data).hexdigest(),
                                  'path': request.raw_path})


def basic(password=PASSWORD):
    return {'Authorization': 'Basic ' + base64.b64encode(('workbench:' + password).encode()).decode()}


class PublicGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h = await Harness().start()
    async def asyncTearDown(self):
        await self.h.close()

    async def test_none_homepage_needs_no_password(self):
        r=await self.h.client.get('/')
        self.assertEqual(r.status,200)
        self.assertEqual(await r.read(), HTML)
        self.assertNotIn('WWW-Authenticate',r.headers)
        self.assertEqual(r.headers['X-Workbench-Auth'],'none')

    async def test_none_accepts_stale_browser_basic_header(self):
        r=await self.h.client.get('/',headers=basic('stale-password'))
        self.assertEqual(r.status,200)
        self.assertNotIn('Authorization',self.h.captured[-1]['headers'])

    async def test_workbench_sidebar_routes_still_available(self):
        r=await self.h.client.get('/_workbench/catalog')
        self.assertEqual(r.status,200)
        self.assertEqual((await r.json())['workspace'],'qwen')
        r=await self.h.client.get('/_workbench/')
        self.assertEqual(await r.read(),HTML)

    async def test_health_reports_version_and_mode_without_password(self):
        r=await self.h.client.get('/healthz')
        data=await r.json()
        self.assertEqual(data['access_version'],'1.2.0')
        self.assertEqual(data['auth_mode'],'none')
        self.assertNotIn(PASSWORD,str(data))

    async def test_ready_route(self):
        r=await self.h.client.get('/readyz')
        self.assertTrue((await r.json())['comfy_ready'])

    async def test_html_and_javascript_mime_types(self):
        for path, expected in [('/', 'text/html'),('/assets/test.js','application/javascript')]:
            r=await self.h.client.get(path)
            self.assertEqual(r.content_type,expected)
            self.assertEqual(r.headers['X-Content-Type-Options'],'nosniff')

    async def test_browser_encoding_is_normalized_to_identity(self):
        r=await self.h.client.get('/', headers={'Accept-Encoding':'gzip, deflate, br, zstd'})
        self.assertEqual(await r.read(),HTML)
        self.assertEqual(self.h.captured[-1]['headers']['Accept-Encoding'],'identity')
        self.assertNotIn('Content-Encoding',r.headers)

    async def test_forced_upstream_gzip_preserves_bytes_and_headers(self):
        r=await self.h.client.get('/gzip')
        data=await r.read()
        self.assertEqual(r.headers['Content-Encoding'],'gzip')
        self.assertEqual(int(r.headers['Content-Length']),len(data))
        self.assertEqual(gzip.decompress(data),HTML)

    async def test_bodyless_get_has_no_transfer_encoding(self):
        r=await self.h.client.get('/')
        await r.read()
        headers=self.h.captured[-1]['headers']
        self.assertNotIn('Transfer-Encoding',headers)

    async def test_head_preserves_length_without_sending_body(self):
        r=await self.h.client.head('/')
        self.assertEqual(await r.read(),b'')
        self.assertEqual(int(r.headers['Content-Length']),len(HTML))
        self.assertNotIn('Transfer-Encoding',r.headers)

    async def test_204_and_304_empty_response_framing(self):
        for path, status in [('/empty',204),('/not-modified',304)]:
            r=await self.h.client.get(path)
            self.assertEqual(r.status,status)
            self.assertEqual(await r.read(),b'')
            self.assertNotIn('Transfer-Encoding',r.headers)

    async def test_repeated_set_cookie_headers_are_preserved(self):
        r=await self.h.client.get('/cookies')
        self.assertEqual(len(r.headers.getall('Set-Cookie')),2)

    async def test_unknown_length_response_stream(self):
        r=await self.h.client.get('/stream')
        self.assertEqual(await r.read(),PAYLOAD)
        self.assertEqual(r.headers['X-Workbench-Gateway'],'1.2.0')

    async def test_video_image_range_response(self):
        r=await self.h.client.get('/view',headers={'Range':'bytes=10-29'})
        self.assertEqual(r.status,206)
        self.assertEqual(await r.read(),PAYLOAD[10:30])
        self.assertEqual(r.headers['Content-Range'],f'bytes 10-29/{len(PAYLOAD)}')

    async def test_redirect_is_returned_not_followed_by_gateway(self):
        r=await self.h.client.get('/redirect',allow_redirects=False)
        self.assertEqual(r.status,302)
        self.assertEqual(r.headers['Location'],'/json')
        self.assertEqual(len(self.h.captured),1)

    async def test_json_prompt_post_byte_exact(self):
        data=b'{"prompt":{"one":{"class_type":"test"}},"token":"test-only"}'
        r=await self.h.client.post('/prompt',data=data,headers={'Content-Type':'application/json'})
        out=await r.json()
        self.assertEqual(out['sha256'],hashlib.sha256(data).hexdigest())

    async def test_multipart_image_upload_byte_exact(self):
        form=aiohttp.FormData()
        form.add_field('image',PAYLOAD,filename='input.png',content_type='image/png')
        r=await self.h.client.post('/upload/image',data=form)
        out=await r.json()
        self.assertEqual(out['name'],'input.png')
        self.assertEqual(out['sha256'],hashlib.sha256(PAYLOAD).hexdigest())

    async def test_compressed_request_is_not_silently_decompressed(self):
        data=gzip.compress(b'{"hello":"world"}')
        r=await self.h.client.post('/echo',data=data,headers={'Content-Encoding':'gzip'})
        self.assertEqual((await r.json())['sha256'],hashlib.sha256(data).hexdigest())
        self.assertEqual(self.h.captured[-1]['headers']['Content-Encoding'],'gzip')

    async def test_encoded_paths_and_query_survive(self):
        from yarl import URL
        target='/echo/image%20one.png?filename=a%2Fb.png&other=%2520'
        r=await self.h.client.get(URL(target,encoded=True))
        self.assertEqual((await r.json())['path'],target)

    async def test_non_basic_native_auth_is_preserved(self):
        r=await self.h.client.get('/echo',headers={'Authorization':'Bearer unit-test-only'})
        await r.read()
        self.assertEqual(self.h.captured[-1]['headers']['Authorization'],'Bearer unit-test-only')

    async def test_client_session_does_not_share_browser_cookies(self):
        self.assertIsInstance(self.h.fixture.session.cookie_jar,aiohttp.DummyCookieJar)
        self.assertFalse(self.h.fixture.session.trust_env)

    async def test_cross_origin_post_still_blocked_without_password(self):
        r=await self.h.client.post('/prompt',headers={'Origin':'https://untrusted.example'})
        self.assertEqual(r.status,403)
        self.assertEqual(len(self.h.captured),0)

    async def test_cross_site_metadata_blocked_even_when_origin_missing(self):
        r=await self.h.client.post('/prompt',headers={'Sec-Fetch-Site':'cross-site'})
        self.assertEqual(r.status,403)

    async def test_same_origin_post_works(self):
        origin=str(self.h.client.make_url('/')).rstrip('/')
        r=await self.h.client.post('/prompt',data=b'{}',headers={'Origin':origin,'Sec-Fetch-Site':'same-origin'})
        self.assertEqual(r.status,200)
        self.assertNotIn('Origin',self.h.captured[-1]['headers'])

    async def test_runpod_public_origin_supported_without_trusting_arbitrary_forwarded_host(self):
        with patch.dict(os.environ,{'RUNPOD_POD_ID':'test123'}):
            r=await self.h.client.post('/prompt',data=b'{}',headers={'Origin':'https://test123-8188.proxy.runpod.net'})
            self.assertEqual(r.status,200)
            r=await self.h.client.post('/prompt',headers={'Origin':'https://untrusted.example','X-Forwarded-Host':'untrusted.example'})
            self.assertEqual(r.status,403)

    async def test_bad_and_null_origins_are_blocked(self):
        for origin in ['null','ftp://example.com','https://test:bad','https://user:pass@example.com']:
            r=await self.h.client.post('/prompt',headers={'Origin':origin})
            self.assertEqual(r.status,403)

    async def test_cross_site_navigation_does_not_reach_loopback_as_cross_site(self):
        r=await self.h.client.get('/',headers={'Sec-Fetch-Site':'cross-site'})
        self.assertEqual(r.status,200)
        self.assertNotIn('Sec-Fetch-Site',self.h.captured[-1]['headers'])

    async def test_websocket_text_and_binary_progress(self):
        origin=str(self.h.client.make_url('/')).rstrip('/')
        async with self.h.client.ws_connect('/ws',headers={'Origin':origin}) as ws:
            await ws.send_str('progress:50')
            msg=await ws.receive(timeout=3)
            self.assertEqual(msg.data,'progress:50')
            await ws.send_bytes(b'preview-png-bytes')
            msg=await ws.receive(timeout=3)
            self.assertEqual(msg.type,WSMsgType.BINARY)
            self.assertEqual(msg.data,b'preview-png-bytes')

    async def test_websocket_subprotocol_and_server_close(self):
        async with self.h.client.ws_connect('/ws',protocols=('json',)) as ws:
            self.assertEqual(ws.protocol,'json')
            await ws.send_str('close-from-server')
            msg=await ws.receive(timeout=3)
            self.assertIn(msg.type,(WSMsgType.CLOSE,WSMsgType.CLOSED))
            self.assertEqual(ws.close_code,1000)

    async def test_cross_origin_websocket_rejected(self):
        with self.assertRaises(aiohttp.WSServerHandshakeError) as error:
            await self.h.client.ws_connect('/ws',headers={'Origin':'https://untrusted.example'})
        self.assertEqual(error.exception.status,403)

    async def test_upstream_down_returns_actionable_503(self):
        await self.h.up.close()
        r=await self.h.client.get('/')
        self.assertEqual(r.status,503)
        self.assertIn('ComfyUI is not responding',await r.text())
        self.assertEqual(r.headers['Retry-After'],'3')

    async def test_known_oversized_upload_rejected(self):
        self.h.app._client_max_size=16
        r=await self.h.client.post('/echo',data=b'x'*32)
        self.assertEqual(r.status,413)

    async def test_chunked_oversized_upload_rejected(self):
        self.h.app._client_max_size=16
        async def chunks():
            yield b'x'*10
            yield b'x'*10
        r=await self.h.client.post('/echo',data=chunks())
        self.assertEqual(r.status,413)


class BasicGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h=await Harness().start('basic')
    async def asyncTearDown(self):
        await self.h.close()

    async def test_basic_still_protects_existing_deployments(self):
        r=await self.h.client.get('/')
        self.assertEqual(r.status,401)
        self.assertIn('Basic realm=',r.headers['WWW-Authenticate'])

    async def test_valid_password_still_works(self):
        r=await self.h.client.get('/',headers=basic())
        self.assertEqual(r.status,200)
        self.assertEqual(await r.read(),HTML)
        self.assertEqual(r.headers['X-Workbench-Auth'],'basic')

    async def test_wrong_password_rejected(self):
        r=await self.h.client.get('/',headers=basic('wrong'))
        self.assertEqual(r.status,401)

    async def test_health_remains_public_in_basic_mode(self):
        r=await self.h.client.get('/healthz')
        self.assertEqual(r.status,200)

    async def test_missing_password_is_not_an_implicit_opt_out(self):
        with self.assertRaises(ValueError):
            self.h.runtime.make_app('',self.h.fixture,auth_mode='basic')

    async def test_invalid_auth_mode_rejected(self):
        for mode in ('off','false','','NONE','disable'):
            with self.assertRaises(ValueError):
                self.h.runtime.make_app('',self.h.fixture,auth_mode=mode)

    async def test_default_mode_is_basic(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(self.h.gateway.auth_mode(),'basic')

    async def test_environment_none_accepts_blank_password(self):
        with patch.dict(os.environ,{'WB_AUTH_MODE':'none'}):
            app=self.h.runtime.make_app('',self.h.fixture)
            self.assertEqual(app[self.h.gateway.AUTH_MODE_KEY],'none')

    async def test_dynamic_hop_headers_are_stripped(self):
        incoming=CIMultiDict([('Connection','X-Hop, Keep-Alive'),('X-Hop','secret'),
                               ('Keep-Alive','timeout=10'),('Content-Length','123'),
                               ('Content-Type','text/html'),('Set-Cookie','one=1'),('Set-Cookie','two=2')])
        out=self.h.gateway.response_headers(incoming)
        self.assertNotIn('X-Hop',out)
        self.assertNotIn('Connection',out)
        self.assertEqual(out['Content-Length'],'123')
        self.assertEqual(len(out.getall('Set-Cookie')),2)


if __name__ == '__main__':
    unittest.main()
