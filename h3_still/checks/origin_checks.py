"""Proxy-origin regression tests, including real aiohttp login/upload/WebSockets."""
import asyncio
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
try:
    from aiohttp import ClientSession, DummyCookieJar, WSServerHandshakeError, web
    from aiohttp.test_utils import TestServer, make_mocked_request
    from multidict import CIMultiDict
except ImportError:
    raise unittest.SkipTest('Run again after aiohttp is installed; Docker runs these checks after dependency installation.')

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from access_gateway import Gateway, COOKIE, PREAUTH, issue_cookie, password_hash, same_origin
from origin_policy import normalize_origin, format_origin, public_origins

C='https://testpod42-8188.proxy.runpod.net'
J='https://testpod42-8888.proxy.runpod.net'

class OriginPolicyChecks(unittest.TestCase):
    def request(self, origin=None, referer=None, **extra):
        headers={'Host':'10.0.0.7:8188', **extra}
        if origin is not None:headers['Origin']=origin
        if referer is not None:headers['Referer']=referer
        return make_mocked_request('POST','/__h3/login-v3',headers=headers)
    def test_same_public_origin_with_internal_host(self):
        self.assertTrue(same_origin(self.request(C),C))
    def test_missing_forwarded_headers_not_required(self):
        self.assertTrue(same_origin(self.request(J),J))
    def test_runpod_auto_configuration(self):
        self.assertEqual(public_origins({'RUNPOD_POD_ID':'testpod42'}),{'comfy':C,'jupyter':J})
    def test_explicit_https_overrides(self):
        x=public_origins({'RUNPOD_POD_ID':'testpod42','H3_COMFY_PUBLIC_ORIGIN':'https://Images.Example.org:443/'})
        self.assertEqual(x,{'comfy':'https://images.example.org','jupyter':J})
    def test_empty_env_allows_direct_only(self):
        self.assertEqual(public_origins({}),{})
    def test_invalid_pod_identifier_rejected(self):
        for pod in ['a.proxy.runpod.net','../other','x:80','a\nb']:
            with self.subTest(pod=pod),self.assertRaises(ValueError):public_origins({'RUNPOD_POD_ID':pod})
    def test_non_https_override_rejected(self):
        with self.assertRaises(ValueError):public_origins({'H3_FILES_PUBLIC_ORIGIN':'http://example.org'})
    def test_default_port_equivalence(self):
        self.assertTrue(same_origin(self.request(C+':443'),C))
    def test_hostname_case_normalized(self):
        self.assertTrue(same_origin(self.request(C.upper()),C))
    def test_referer_fallback(self):
        self.assertTrue(same_origin(self.request(referer=C+'/__h3/files?dir=a'),C))
    def test_null_not_replaced_by_referer(self):
        self.assertFalse(same_origin(self.request('null',C+'/__h3/login-v3'),C))
    def test_missing_headers_rejected(self):
        self.assertFalse(same_origin(self.request(),C))
    def test_explicit_cross_site_metadata_rejected(self):
        self.assertFalse(same_origin(self.request(C,**{'Sec-Fetch-Site':'cross-site'}),C))
    def test_port_separation(self):
        self.assertFalse(same_origin(self.request(J),C))
    def test_scheme_mismatch(self):
        self.assertFalse(same_origin(self.request(C.replace('https:','http:')),C))
    def test_unrelated_site_rejected(self):
        self.assertFalse(same_origin(self.request('https://evil.invalid'),C))
    def test_other_pod_rejected(self):
        self.assertFalse(same_origin(self.request(C.replace('testpod42','anotherpod')),C))
    def test_origin_domain_suffix_attack_rejected(self):
        self.assertFalse(same_origin(self.request(C+'.evil.invalid'),C))
    def test_forwarded_headers_never_grant_trust(self):
        request=self.request('https://evil.invalid', **{'X-Forwarded-Host':'evil.invalid',
            'Forwarded':'host=evil.invalid;proto=https','X-Forwarded-Proto':'https'})
        self.assertFalse(same_origin(request,C))
    def test_forged_host_does_not_override_config(self):
        request=make_mocked_request('POST','/',headers={'Host':'evil.invalid','Origin':'https://evil.invalid'})
        self.assertFalse(same_origin(request,C))
    def test_duplicate_origin_rejected(self):
        headers=CIMultiDict([('Host','local:8188'),('Origin',C),('Origin','https://evil.invalid')])
        self.assertFalse(same_origin(make_mocked_request('POST','/',headers=headers),C))
    def test_userinfo_and_malformed_origins_rejected(self):
        for value in ['https://user@example.org','https://x@'+C[8:],C+',https://evil.invalid',
                      C+'/path',C+'?x=1',C+'#x',C+':bad',C+':0',C+':65536',C+':',
                      'https://*.example.org','https://a\nb','https://a\\b']:
            with self.subTest(value=value),self.assertRaises(ValueError):normalize_origin(value)
    def test_ipv6_origin(self):
        self.assertEqual(format_origin(normalize_origin('https://[::1]:443')),'https://[::1]')
    def test_direct_compatibility(self):
        r=make_mocked_request('POST','/',headers={'Host':'localhost:8765','Origin':'http://localhost:8765'})
        self.assertTrue(same_origin(r))
    def test_launcher_persists_trusted_config(self):
        from service_launcher import make_config
        with tempfile.TemporaryDirectory() as name, patch.dict('os.environ',{'RUNPOD_POD_ID':'testpod42'},clear=True):
            root=Path(name);file,_=make_config('a-real-test-password',root/'runtime',root)
            self.assertEqual(json.loads(file.read_text())['public_origins'],{'comfy':C,'jupyter':J})

class GatewayProxyChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.password='example-test-password-not-a-secret'
        self.headers={'Host':'10.0.0.7:8188','Origin':C}
        async def backend(request):
            if request.headers.get('Upgrade','').lower()=='websocket':
                if request.headers.get('Origin')!='https://'+request.host:
                    raise web.HTTPForbidden(text='Backend origin/host mismatch')
                ws=web.WebSocketResponse();await ws.prepare(request)
                await ws.send_str('canonical-host-accepted')
                async for msg in ws:
                    if msg.type==web.WSMsgType.TEXT:await ws.send_str(msg.data)
                return ws
            return web.json_response({'host':request.host,'origin':request.headers.get('Origin'),
                'forwarded_host':request.headers.get('X-Forwarded-Host'),
                'scheme':request.headers.get('X-Forwarded-Proto'),
                'forwarded':request.headers.get('Forwarded'),
                'authorization':request.headers.get('Authorization')})
        app=web.Application();app.router.add_route('*','/{path:.*}',backend)
        self.upstream=TestServer(app);await self.upstream.start_server()
        config={'session_key':'test-signing-secret','password_hash':password_hash(self.password),
            'secure_cookies':True,'workspace':str(self.root),'public_origins':{'comfy':C,'jupyter':J},
            'comfy_upstream':str(self.upstream.make_url('')).rstrip('/'),
            'jupyter_upstream':str(self.upstream.make_url('')).rstrip('/'), 'jupyter_token':'private-test-jupyter-token'}
        self.gateway=Gateway(config,'comfy');self.server=TestServer(self.gateway.application());await self.server.start_server()
        self.session=ClientSession(cookie_jar=DummyCookieJar())
    async def asyncTearDown(self):
        await self.session.close();await self.server.close();await self.upstream.close();self.temp.cleanup()
    async def login(self,password=None,source=C):
        async with self.session.get(self.server.make_url('/__h3/login-v3'),headers={'Host':'internal.invalid:8188'}) as r:
            text=await r.text();nonce=re.search(r'name="nonce" value="([^"]+)"',text).group(1)
            self.assertTrue(r.cookies[PREAUTH]['secure']);self.assertTrue(r.cookies[PREAUTH]['httponly'])
        headers={**self.headers,'Origin':source,'Cookie':PREAUTH+'='+nonce}
        return await self.session.post(self.server.make_url('/__h3/login-v3'),headers=headers,
                   data={'nonce':nonce,'password':password or self.password},allow_redirects=False)
    def auth_headers(self,source=C,**extra):
        return {**self.headers,'Origin':source,'Cookie':COOKIE+'='+issue_cookie('test-signing-secret'),**extra}
    async def test_01_login_through_rewritten_host(self):
        async with await self.login() as r:
            self.assertEqual(r.status,303,await r.text());self.assertIn(COOKIE,r.cookies)
            self.assertTrue(r.cookies[COOKIE]['secure']);self.assertIn('no-store',r.headers['Cache-Control'])
            self.assertNotIn('WWW-Authenticate',r.headers)
    async def test_02_wrong_password_still_rejected(self):
        async with await self.login(password='wrong-test-password') as r:
            self.assertEqual(r.status,200);self.assertIn('Incorrect password',await r.text());self.assertNotIn(COOKIE,r.cookies)
    async def test_03_wrong_origin_rejected_even_with_valid_nonce(self):
        async with await self.login(source='https://evil.invalid') as r:self.assertEqual(r.status,403)
    async def test_04_missing_nonce_still_rejected(self):
        async with self.session.post(self.server.make_url('/__h3/login-v3'),headers=self.headers,
             data={'nonce':'fake','password':self.password}) as r:
            self.assertIn('expired',await r.text());self.assertNotIn(COOKIE,r.cookies)
    async def test_05_post_to_comfy_uses_public_host(self):
        headers=self.auth_headers(**{'X-Forwarded-Host':'evil.invalid','X-Forwarded-Proto':'http',
                                    'Forwarded':'host=evil.invalid;proto=http','Authorization':'injected'})
        async with self.session.post(self.server.make_url('/prompt'),headers=headers,json={}) as r:
            self.assertEqual(r.status,200,await r.text());data=await r.json()
            self.assertEqual(data['host'],C[8:]);self.assertEqual(data['forwarded_host'],C[8:])
            self.assertEqual(data['scheme'],'https');self.assertIsNone(data['forwarded']);self.assertIsNone(data['authorization'])
    async def test_06_chunked_upload_uses_same_origin_policy(self):
        h=self.auth_headers();data=b'one video upload chunk'
        async with self.session.post(self.server.make_url('/__h3/upload/init'),headers=h,
                                    json={'dir':'','name':'test.bin','size':len(data)}) as r:
            self.assertEqual(r.status,200,await r.text());uid=(await r.json())['id']
        async with self.session.patch(self.server.make_url('/__h3/upload/chunk/'+uid+'?offset=0'),headers=h,data=data) as r:
            self.assertEqual(r.status,200,await r.text())
        async with self.session.post(self.server.make_url('/__h3/upload/finish/'+uid),headers=h) as r:
            self.assertEqual(r.status,200,await r.text())
        self.assertEqual((self.root/'test.bin').read_bytes(),data)
    async def test_07_upload_cross_origin_rejected(self):
        async with self.session.post(self.server.make_url('/__h3/upload/init'),headers=self.auth_headers('https://evil.invalid'),
            json={'dir':'','name':'evil.bin','size':0}) as r:self.assertEqual(r.status,403)
        self.assertFalse((self.root/'evil.bin').exists())
    async def test_08_websocket_rewritten_host_passes_backend(self):
        async with self.session.ws_connect(self.server.make_url('/ws'),headers=self.auth_headers()) as ws:
            self.assertEqual((await ws.receive(timeout=5)).data,'canonical-host-accepted')
            await ws.send_str('echo-proof');self.assertEqual((await ws.receive(timeout=5)).data,'echo-proof')
    async def test_09_websocket_cross_origin_rejected(self):
        with self.assertRaises(WSServerHandshakeError) as cm:
            await self.session.ws_connect(self.server.make_url('/ws'),headers=self.auth_headers('https://evil.invalid'))
        self.assertEqual(cm.exception.status,403)
    async def test_10_utm_does_not_grant_access(self):
        async with self.session.get(self.server.make_url('/system_stats?utm_source=anything'),headers={'Host':'internal:8188'}) as r:
            self.assertEqual(r.status,401);self.assertIn('no-store',r.headers['Cache-Control'])
    async def test_11_jupyter_canonical_host_and_token(self):
        self.gateway.service='jupyter';self.gateway.public_origin=J
        async with self.session.post(self.server.make_url('/jupyter/api/terminals'),headers=self.auth_headers(J),json={}) as r:
            self.assertEqual(r.status,200,await r.text());data=await r.json()
            self.assertEqual(data['host'],J[8:]);self.assertEqual(data['authorization'],'token private-test-jupyter-token')
    async def test_12_diagnostics_has_no_secrets(self):
        async with self.session.get(self.server.make_url('/__h3/diagnostics'),headers=self.auth_headers()) as r:
            text=await r.text();self.assertIn(C,text)
            self.assertNotIn(self.password,text);self.assertNotIn('private-test-jupyter-token',text)

if __name__=='__main__':unittest.main()
