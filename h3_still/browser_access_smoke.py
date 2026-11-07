"""Actual Chromium + JupyterLab through the production cookie/proxy gateway.

Uses an HTTPS-to-HTTP proxy that rewrites Host, real Jupyter, and no model weights.
The Jupyter file browser and browser-terminal shell are exercised, not just /login.
"""
from pathlib import Path
import importlib.metadata as md
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE/'runtime'))
from service_launcher import jupyter_settings,private_json,write_jupyter_proxy_config
from access_gateway import password_hash,COOKIE
from recovery_launcher import terminal_environment,stop_process


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]


def protocol_checks(client, cbase, jbase, password, token, work):
    import re
    import ssl
    import websocket
    def login(base):
        r=client.get(base+'/__h3/login',timeout=10)
        nonce=re.search(r'name="nonce" value="([^"]+)"',r.text).group(1)
        r=client.post(base+'/__h3/login',data={'nonce':nonce,'password':password},headers={'Origin':base},timeout=10)
        assert r.status_code==200
    response=client.get(cbase+'/__h3/login',timeout=10)
    nonce=re.search(r'name="nonce" value="([^"]+)"',response.text).group(1)
    wrong=client.post(cbase+'/__h3/login',data={'nonce':nonce,'password':'wrong-password-123'},headers={'Origin':cbase},timeout=10)
    assert 'Incorrect password' in wrong.text
    assert client.get(cbase+'/system_stats',timeout=5).status_code==401
    login(cbase)
    assert client.get(cbase+'/system_stats',timeout=5).status_code==200
    assert client.post(cbase+'/prompt',json={},headers={'Origin':'https://evil.invalid'},timeout=5).status_code==403
    login(jbase)
    assert 'browser-proof.txt' in client.get(jbase+'/__h3/files',timeout=5).text
    headers={'Origin':jbase}
    content=b'chunked upload verified\n'*1000
    init=client.post(jbase+'/__h3/upload/init',json={'dir':'','name':'chunked-proof.txt','size':len(content)},headers=headers,timeout=5)
    assert init.status_code==200,init.text
    upload_id=init.json()['id']
    offset=0
    for chunk in (content[:7500],content[7500:]):
        response=client.patch(jbase+'/__h3/upload/chunk/'+upload_id+'?offset='+str(offset),data=chunk,headers=headers,timeout=5)
        assert response.status_code==200,response.text
        offset=response.json()['offset']
    assert client.post(jbase+'/__h3/upload/finish/'+upload_id,headers=headers,timeout=5).status_code==200
    assert (work/'chunked-proof.txt').read_bytes()==content
    assert client.get(jbase+'/__h3/download?path=chunked-proof.txt',timeout=5).content==content
    duplicate=client.post(jbase+'/__h3/upload/init',json={'dir':'','name':'chunked-proof.txt','size':len(content)},headers=headers,timeout=5)
    assert duplicate.status_code==409
    assert client.get(jbase+'/__h3/download?path=../gateway.json',timeout=5).status_code==400
    page=client.get(jbase+'/jupyter/lab/workspaces/h3-studio-v2?reset',timeout=15)
    assert page.status_code==200,page.text[:1000]
    assert token not in page.text
    # Get the actual assets named by the real Lab page through the same gateway.
    assets=re.findall(r'(?:src|href)="([^" ]+\.(?:js|css)(?:\?[^" ]*)?)"',page.text)
    checked=0
    for asset in assets:
        if asset.startswith('/jupyter/'):
            result=client.get(jbase+asset,timeout=20)
            assert result.status_code==200,(asset,result.status_code)
            checked+=1
    assert checked>0,'No Jupyter frontend assets checked'
    for endpoint in ('contents','kernelspecs','sessions','terminals','me'):
        response=client.get(jbase+'/jupyter/api/'+endpoint,timeout=10)
        assert response.status_code==200,(endpoint,response.status_code,response.text[:400])
    terminal=client.post(jbase+'/jupyter/api/terminals',json={},headers=headers,timeout=10)
    assert terminal.status_code==200,terminal.text
    name=terminal.json()['name']
    cookies='; '.join(k+'='+v for k,v in client.cookies.get_dict().items())
    ws=websocket.create_connection(jbase.replace('https:','wss:')+'/jupyter/terminals/websocket/'+name,
        cookie=cookies,origin=jbase,sslopt={'cert_reqs':ssl.CERT_NONE},timeout=10,
        http_no_proxy=['127.0.0.1','localhost'])
    try:
        ws.send(json.dumps(['stdin',"printf '%s' h3-proxy-shell-ok > proxy-shell-proof.txt\r"]))
        end=time.monotonic()+15
        while time.monotonic()<end and not (work/'proxy-shell-proof.txt').exists():
            ws.settimeout(1)
            try:ws.recv()
            except websocket.WebSocketTimeoutException:pass
        assert (work/'proxy-shell-proof.txt').read_text()=='h3-proxy-shell-ok'
    finally:ws.close()
    client.delete(jbase+'/jupyter/api/terminals/'+name,headers=headers,timeout=5).raise_for_status()
    # Comfy event WebSockets also pass through the same cookie gate.
    ws=websocket.create_connection(cbase.replace('https:','wss:')+'/ws',cookie=cookies,origin=cbase,
        sslopt={'cert_reqs':ssl.CERT_NONE},timeout=5,http_no_proxy=['127.0.0.1','localhost'])
    try:assert ws.recv()=='h3-ws-ready'
    finally:ws.close()
    return {'real_jupyter_assets_loaded':checked,'jupyter_api_and_contents':True,
            'real_terminal_shell_over_secure_proxy':True,'chunked_upload_download':True,
            'duplicate_upload_preserved':True,'cross_origin_rejected':True}


def main():
    import requests
    import urllib3
    from playwright.sync_api import sync_playwright
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    with tempfile.TemporaryDirectory(prefix='h3-browser-') as temp:
        root=Path(temp);work=root/'work';work.mkdir();runtime=root/'runtime';runtime.mkdir()
        (work/'browser-proof.txt').write_text('Jupyter file browser is operational.')
        ports=[port() for _ in range(6)]
        cp,jp,cu,ju,cg,jg=ports
        cbase='https://127.0.0.1:'+str(cp);jbase='https://127.0.0.1:'+str(jp)
        password='browser-test-password-not-a-user-secret'
        token='private-jupyter-test-token-'+os.urandom(16).hex()
        cert=root/'cert.pem';key=root/'key.pem'
        subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-keyout',str(key),'-out',str(cert),
                        '-days','1','-nodes','-subj','/CN=localhost'],check=True,capture_output=True)
        gate=private_json(runtime/'gateway.json',dict(password_hash=password_hash(password),session_key=os.urandom(32).hex(),
            workspace=str(work),secure_cookies=True,comfy_upstream='http://127.0.0.1:'+str(cu),
            jupyter_upstream='http://127.0.0.1:'+str(ju),jupyter_token=token,comfy_port=cg,jupyter_port=jg,
            public_origins={'comfy':cbase,'jupyter':jbase},listen='127.0.0.1'))
        boundary=root/'test_tls_boundary.py'
        boundary.write_text("from aiohttp import web, ClientSession, WSMsgType\nimport asyncio, json, ssl, sys\nfrom multidict import CIMultiDict\nHOP={'connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailer','transfer-encoding','upgrade','content-length'}\nasync def main():\n ports,cert,key=json.loads(sys.argv[1]),sys.argv[2],sys.argv[3]\n tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);tls.load_cert_chain(cert,key)\n runners=[]\n async with ClientSession(auto_decompress=False,trust_env=False) as client:\n  for public,inner in ports:\n   def factory(public,inner):\n    async def proxy(r):\n     h={k:v for k,v in r.headers.items() if k.lower() not in HOP|{'host'} and not k.lower().startswith('x-forwarded-')}\n     h['Host']='10.42.0.7:'+str(inner)\n     h['X-Forwarded-Host']='127.0.0.1:'+str(public)\n     h['X-Forwarded-Proto']='https'\n     url='http://127.0.0.1:'+str(inner)+str(r.rel_url)\n     if r.headers.get('Upgrade','').lower()=='websocket':\n      protocols=[p.strip() for p in r.headers.get('Sec-WebSocket-Protocol','').split(',') if p.strip()]\n      h={k:v for k,v in h.items() if not k.lower().startswith('sec-websocket')}\n      try:up=await client.ws_connect(url,headers=h,protocols=protocols)\n      except Exception as e:raise web.HTTPBadGateway(text='Test boundary WebSocket: '+str(e))\n      down=web.WebSocketResponse(protocols=[up.protocol] if up.protocol else ());await down.prepare(r)\n      async def relay(a,b):\n       async for m in a:\n        if m.type==WSMsgType.TEXT:await b.send_str(m.data)\n        elif m.type==WSMsgType.BINARY:await b.send_bytes(m.data)\n        else:break\n      jobs=[asyncio.create_task(relay(down,up)),asyncio.create_task(relay(up,down))]\n      try:await asyncio.wait(jobs,return_when=asyncio.FIRST_COMPLETED)\n      finally:\n       for job in jobs:job.cancel()\n       await asyncio.gather(*jobs,return_exceptions=True);await down.close();await up.close()\n      return down\n     async with client.request(r.method,url,headers=h,data=r.content.iter_chunked(262144) if r.can_read_body else None,allow_redirects=False) as up:\n      headers=CIMultiDict((k,v) for k,v in up.headers.items() if k.lower() not in HOP)\n      down=web.StreamResponse(status=up.status,headers=headers);await down.prepare(r)\n      async for block in up.content.iter_chunked(262144):await down.write(block)\n      await down.write_eof();return down\n    return proxy\n   app=web.Application(client_max_size=4*1024**3);app.router.add_route('*','/{path:.*}',factory(public,inner))\n   runner=web.AppRunner(app,access_log=None);await runner.setup()\n   await web.TCPSite(runner,'127.0.0.1',public,ssl_context=tls).start();runners.append(runner)\n  try:await asyncio.Event().wait()\n  finally:\n   for runner in runners:await runner.cleanup()\nasyncio.run(main())\n")
        jup=write_jupyter_proxy_config(runtime,work,token,ju)
        fake=root/'fake_comfy.py'
        fake.write_text('''from aiohttp import web
import sys
async def handler(r):
 if r.path=='/ws':
  if r.headers.get('Origin') != 'https://'+r.host:raise web.HTTPForbidden(text='Fixture Host/Origin mismatch')
  ws=web.WebSocketResponse();await ws.prepare(r);await ws.send_str('h3-ws-ready')
  async for m in ws:
   if m.type==web.WSMsgType.TEXT:await ws.send_str(m.data)
  return ws
 if r.path=='/':return web.Response(text='<html><body>Comfy proxy fixture</body></html>',content_type='text/html')
 return web.json_response({'ok':True,'path':r.path})
a=web.Application();a.router.add_route('*','/{path:.*}',handler);web.run_app(a,host='127.0.0.1',port=int(sys.argv[1]),print=None)
''')
        env=terminal_environment();env['JUPYTER_CONFIG_DIR']=str(runtime);env['JUPYTER_CONFIG_PATH']=str(runtime)
        env['JUPYTER_RUNTIME_DIR']=str(runtime/'run')
        children=[];streams=[]
        def start(name,cmd):
            log=(root/(name+'.log')).open('w');streams.append(log)
            child=subprocess.Popen(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child)
        try:
            start('fake',[sys.executable,str(fake),str(cu)])
            start('jupyter',[sys.executable,'-m','jupyterlab','--config',str(jup)])
            start('gateway',[sys.executable,str(PACKAGE/'runtime/access_gateway.py'),'--config',str(gate)])
            start('boundary',[sys.executable,str(boundary),json.dumps([[cp,cg],[jp,jg]]),str(cert),str(key)])
            client=requests.Session();client.trust_env=False;client.verify=False
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if any(c.poll() is not None for c in children):raise RuntimeError('A test service exited.')
                try:
                    if client.get(cbase+'/__h3/health',timeout=2).status_code==200:
                        r=client.get('http://127.0.0.1:%s/jupyter/api/status'%ju,headers={'Authorization':'token '+token},timeout=2)
                        if r.status_code==200:break
                except requests.RequestException:pass
                time.sleep(.3)
            else:raise RuntimeError('Readiness failed.')
            for path in ('/system_stats','/system_stats?utm_source=test','/ws?clientId=test'):
                r=client.get(cbase+path,timeout=5);assert r.status_code==401,(path,r.status_code)
                assert 'WWW-Authenticate' not in r.headers
                assert 'no-store' in r.headers['Cache-Control']
            root_response=client.get(cbase+'/',timeout=5)
            assert root_response.status_code==200 and 'Sign in' in root_response.text
            protocol_report=protocol_checks(client,cbase,jbase,password,token,work)
            http_only='--http-only' in sys.argv
            errors=[]
            if not http_only:
                with sync_playwright() as p:
                    browser=p.chromium.launch(executable_path=shutil.which('chromium') or None,args=['--no-sandbox'],headless=True)
                    context=browser.new_context(ignore_https_errors=True)
                    page=context.new_page();errors=[]
                    page.on('pageerror',lambda e:errors.append(str(e)))
                    page.goto(cbase+'/');page.get_by_label('Password').fill('wrong-password-1234');page.get_by_role('button',name='Sign in',exact=True).click()
                    page.get_by_text('Incorrect password.',exact=True).wait_for()
                    page.get_by_label('Password').fill(password);page.get_by_role('button',name='Sign in',exact=True).click()
                    page.get_by_text('Comfy proxy fixture',exact=True).wait_for()
                    cookies=context.cookies();assert any(c['name']==COOKIE and c['secure'] and c['httpOnly'] for c in cookies)
                    assert page.evaluate("async()=> (await fetch('/system_stats')).status")==200
                    result=page.evaluate("""()=>new Promise((resolve,reject)=>{let w=new WebSocket(location.origin.replace('https:','wss:')+'/ws');w.onmessage=e=>{resolve(e.data);w.close()};w.onerror=reject;setTimeout(()=>reject('timeout'),10000)})""")
                    assert result=='h3-ws-ready'
                    # Test cross-origin state change with a valid cookie, including a random UTM.
                    session='; '.join(c['name']+'='+c['value'] for c in cookies)
                    r=client.post(cbase+'/prompt?utm_source=test',headers={'Cookie':session,'Origin':'https://evil.invalid'},json={},timeout=5)
                    assert r.status_code==403
                    page.goto(jbase+'/__h3/login');page.get_by_label('Password').fill(password);page.get_by_role('button',name='Sign in',exact=True).click()
                    page.get_by_role('link',name='Open file browser',exact=True).click()
                    page.get_by_text('browser-proof.txt',exact=True).wait_for()
                    uploaded=root/'uploaded-test.txt';uploaded.write_text('browser upload passed')
                    page.locator('input[type=file]').set_input_files(str(uploaded))
                    page.get_by_role('button',name='Upload',exact=True).click()
                    page.get_by_text('uploaded-test.txt',exact=True).wait_for()
                    assert (work/'uploaded-test.txt').read_text()=='browser upload passed'
                    page.goto(jbase+'/jupyter/lab/workspaces/h3-studio-v2?reset')
                    page.wait_for_selector('.jp-FileBrowser',state='visible',timeout=90000)
                    page.get_by_text('browser-proof.txt',exact=True).first.wait_for(timeout=60000)
                    assert page.evaluate("async()=> (await fetch('/jupyter/api/contents')).status")==200
                    assert token not in page.content(),'Private token appeared in HTML.'
                    # Open and operate a real shell THROUGH the rendered browser terminal.
                    page.locator('.jp-LauncherCard').filter(has_text='Terminal').first.click(timeout=20000)
                    terminal=page.locator('.xterm-helper-textarea').last
                    terminal.wait_for(state='attached',timeout=20000)
                    terminal.focus()
                    page.keyboard.type("printf '%s' h3-browser-shell-ok > browser-shell-proof.txt",delay=5)
                    page.keyboard.press('Enter')
                    deadline=time.monotonic()+20
                    while time.monotonic()<deadline and not (work/'browser-shell-proof.txt').exists():
                        page.wait_for_timeout(300)
                    assert (work/'browser-shell-proof.txt').read_text()=='h3-browser-shell-ok'
                    screenshot=PACKAGE/'browser-access-check.png'
                    page.screenshot(path=str(screenshot),full_page=False)
                    context.close();browser.close()
            report=dict(jupyterlab=md.version('jupyterlab'),jupyter_server=md.version('jupyter_server'),
                https_cookie_login=True,no_basic_auth_challenge=True,wrong_password_rejected=True,
                unauthenticated_query_strings_denied=True,cross_origin_post_blocked=True,
                comfy_websocket_proxy='passed using local fixture upstream',
                lightweight_filebrowser_rendered=not http_only,browser_upload=not http_only,
                actual_jupyterlab_filebrowser_initialized=not http_only,actual_browser_terminal_command=not http_only,
                private_jupyter_token_not_in_html=True,browser_page_errors=errors,
                tls_terminated_before_gateway=True,rewritten_host_proxy_tested=True,
                deployed_runpod_proxy_tested=False,gpu_inference=False,protocol_tests=protocol_report,
                browser_note="Browser UI test deferred to CI; local managed Chromium blocks navigation" if http_only else "Executed")
            (PACKAGE/'browser-access-report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
        except Exception:
            for log in root.glob('*.log'):
                print(log.name,log.read_text(errors='replace')[-6000:].replace(token,'<redacted>'),file=sys.stderr)
            raise
        finally:
            for child in reversed(children):stop_process(child)
            for stream in streams:stream.close()

if __name__=='__main__':main()
