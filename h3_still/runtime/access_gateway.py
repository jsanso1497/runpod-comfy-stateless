"""Form/cookie authentication for ComfyUI and Jupyter behind the RunPod HTTPS proxy.

No HTTP Basic challenge. Internal services bind to loopback; Jupyter also has an
unexposed random token. Cookies, origin checks and no-store headers cover both
HTTP and WebSocket paths. The small file browser does not depend on JupyterLab JS.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
from collections import defaultdict, deque
import hashlib
import hmac
import html
import json
import os
from pathlib import Path
import secrets
import shutil
import ssl
import time
from urllib.parse import quote, urlencode, urlsplit

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

VERSION = '2026-11-07.2'
COOKIE = 'h3_studio_v2_session'
PREAUTH = 'h3_studio_v2_login'
TTL = 12 * 3600
NO_CACHE = {'Cache-Control': 'private, no-store, no-cache, max-age=0, must-revalidate',
            'Pragma': 'no-cache', 'Expires': '0', 'Surrogate-Control': 'no-store',
            'CDN-Cache-Control': 'no-store', 'Cloudflare-CDN-Cache-Control': 'no-store',
            'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'same-origin'}
HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
       'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length'}


def password_hash(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1)
    return base64.urlsafe_b64encode(salt + digest).decode()


def password_ok(password, encoded):
    try:
        data = base64.urlsafe_b64decode(encoded)
        digest = hashlib.scrypt(password.encode(), salt=data[:16], n=16384, r=8, p=1)
        return hmac.compare_digest(data[16:], digest)
    except (ValueError, TypeError):
        return False


def sign(secret, payload):
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_cookie(secret, purpose='session', ttl=TTL):
    value = '%s.%s.%s' % (purpose, int(time.time()) + ttl, secrets.token_hex(24))
    return value + '.' + sign(secret, value)


def valid_cookie(secret, value, purpose='session'):
    try:
        p, expires, nonce, signature = value.split('.')
        return (p == purpose and int(expires) >= time.time() and len(nonce) == 48 and
                hmac.compare_digest(signature, sign(secret, '.'.join((p, expires, nonce)))))
    except (AttributeError, ValueError):
        return False


def same_origin(request):
    source = request.headers.get('Origin') or request.headers.get('Referer')
    if not source:
        return False
    try:
        origin = urlsplit(source)
        return origin.scheme in ('http', 'https') and origin.netloc.lower() == request.host.lower()
    except ValueError:
        return False


def document(title, body):
    return '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>%s</title>
<style>body{font:16px system-ui,sans-serif;max-width:1050px;margin:6vh auto;padding:24px;background:#f5f6f8;color:#1b2430}main{background:white;padding:28px;border:1px solid #ddd;border-radius:12px}h1{font-size:26px}input,button{font:inherit;padding:10px;margin:6px 0}button,.button{display:inline-block;background:#185586;color:white;border:0;border-radius:6px;padding:12px;text-decoration:none;cursor:pointer}a{color:#185586}table{width:100%%;border-collapse:collapse}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd;overflow-wrap:anywhere}code,pre{overflow-wrap:anywhere;white-space:pre-wrap}small{color:#586473}.error{color:#9e2121}</style></head><body><main>%s</main></body></html>''' % (html.escape(title), body)


class Gateway:
    def __init__(self, config, service):
        self.config, self.service = config, service
        self.secret = config['session_key']
        self.secure = config.get('secure_cookies', True)
        self.attempts = defaultdict(deque)
        self.uploads = {}
        self.root = Path(config['workspace']).resolve()
        self.client = None

    def authenticated(self, request):
        return valid_cookie(self.secret, request.cookies.get(COOKIE))

    def set_cookie(self, response, name, value, max_age):
        response.set_cookie(name, value, httponly=True, secure=self.secure,
                            samesite='Lax', path='/', max_age=max_age)

    def login_page(self, request, message=''):
        nonce = issue_cookie(self.secret, 'login', 600)
        body = '<h1>H3 Still Studio</h1><p>Sign in to %s using your <code>comfy_password</code>.</p>' % (
            'ComfyUI' if self.service == 'comfy' else 'Files, JupyterLab and terminal')
        if message:
            body += '<p class="error">%s</p>' % html.escape(message)
        body += '<form method="post" action="/__h3/login"><input type="hidden" name="nonce" value="%s">' % nonce
        body += '<label>Password<br><input type="password" name="password" autocomplete="current-password" required autofocus maxlength="72"></label><br><button type="submit">Sign in</button></form><p><small>Studio %s. No browser username/password popup is used.</small></p>' % VERSION
        response = web.Response(text=document('H3 Studio sign in', body), content_type='text/html')
        self.set_cookie(response, PREAUTH, nonce, 600)
        return response

    async def login(self, request):
        if not same_origin(request):
            raise web.HTTPForbidden(text='Login origin check failed. Open this service in its own HTTPS tab.')
        data = await request.post()
        nonce = str(data.get('nonce', ''))
        if nonce != request.cookies.get(PREAUTH) or not valid_cookie(self.secret, nonce, 'login'):
            return self.login_page(request, 'This login form expired. Enter your password again.')
        key = (request.remote or 'proxy')[:128]
        if len(self.attempts) > 4096:
            self.attempts.clear()
        q = self.attempts[key]
        now = time.monotonic()
        while q and q[0] < now - 60:
            q.popleft()
        if len(q) >= 8:
            raise web.HTTPTooManyRequests(text='Too many attempts. Try again after one minute.')
        q.append(now)
        password = str(data.get('password', ''))
        if not 16 <= len(password.encode()) <= 72 or not await asyncio.to_thread(password_ok, password, self.config['password_hash']):
            return self.login_page(request, 'Incorrect password.')
        self.attempts.pop(key, None)
        target = '/?h3v=' + VERSION + '&fresh=' + secrets.token_hex(8) if self.service == 'comfy' else '/__h3/home'
        response = web.HTTPSeeOther(location=target)
        self.set_cookie(response, COOKIE, issue_cookie(self.secret), TTL)
        response.del_cookie(PREAUTH, path='/')
        return response

    def home(self):
        body = '<h1>H3 Studio files and terminal</h1><p>The lightweight file browser works independently of JupyterLab.</p>'
        body += '<p><a class="button" href="/__h3/files">Open file browser</a> <a class="button" href="/jupyter/lab/workspaces/h3-studio-v2?reset">Open JupyterLab</a></p>'
        body += '<p>For a terminal: open JupyterLab, then <strong>Launcher &gt; Terminal</strong>. '
        body += 'No RunPod Web Terminal toggle or second password is needed.</p><p><a href="/__h3/diagnostics">Connection diagnostics</a></p>'
        body += '<form method="post" action="/__h3/logout"><button>Sign out</button></form>'
        return web.Response(text=document('H3 Studio', body), content_type='text/html')

    def file_path(self, value=''):
        rel = Path(value)
        if rel.is_absolute() or '..' in rel.parts or '\\' in value:
            raise web.HTTPBadRequest(text='Invalid file path.')
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root):
            raise web.HTTPForbidden(text='Path leaves workspace.')
        return path

    def files(self, request):
        rel = request.query.get('dir', '')
        directory = self.file_path(rel)
        if not directory.is_dir():
            raise web.HTTPNotFound(text='Directory not found.')
        try:
            page = max(0, min(10000, int(request.query.get('page', 0))))
        except ValueError:
            page = 0
        items = sorted((p for p in directory.iterdir() if not p.name.startswith('.')), key=lambda p: (not p.is_dir(), p.name.lower()))
        body = '<h1>Files</h1><p><a href="/__h3/home">Studio home</a> | <a href="/jupyter/lab/workspaces/h3-studio-v2?reset">JupyterLab</a></p><p><code>%s</code></p>' % html.escape(str(directory))
        if directory != self.root:
            parent = directory.parent.relative_to(self.root).as_posix()
            body += '<p><a href="/__h3/files?%s">Up one folder</a></p>' % urlencode({'dir': parent})
        body += '<form id="upload-form" method="post" enctype="multipart/form-data" action="/__h3/upload?%s"><input type="file" name="file" required><button>Upload</button></form>' % html.escape(urlencode({'dir': rel}))
        body += '<p id="upload-status" role="status"></p><small>Large files use 4 MB chunks to avoid one oversized proxy request. Existing files are never overwritten.</small>'
        body += r"""<script>
const form=document.getElementById('upload-form'),status=document.getElementById('upload-status');
form.addEventListener('submit',async event=>{
 event.preventDefault();const file=form.querySelector('input[type=file]').files[0];if(!file)return;
 const button=form.querySelector('button');button.disabled=true;
 async function send(url,options){let r=await fetch(url,options);if(!r.ok)throw new Error(await r.text());return r.json();}
 try{
  const dir=new URL(location.href).searchParams.get('dir')||'';
  const init=await send('/__h3/upload/init',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dir,name:file.name,size:file.size})});
  const step=4*1024*1024;
  for(let offset=0;offset<file.size;){
   const result=await send('/__h3/upload/chunk/'+init.id+'?offset='+offset,{method:'PATCH',headers:{'Content-Type':'application/octet-stream'},body:file.slice(offset,Math.min(offset+step,file.size))});
   offset=result.offset;status.textContent='Uploaded '+Math.round(offset/file.size*100)+'%';
  }
  await send('/__h3/upload/finish/'+init.id,{method:'POST'});location.reload();
 }catch(error){status.textContent='Upload stopped: '+error.message;}finally{button.disabled=false;}
});</script>"""
        body += '<table><tr><th>Name</th><th>Size</th><th>Action</th></tr>'
        for path in items[page*200:(page+1)*200]:
            try:
                resolved = path.resolve()
                if not resolved.is_relative_to(self.root):
                    continue
                relative = path.relative_to(self.root).as_posix()
                if path.is_dir():
                    link = '/__h3/files?' + urlencode({'dir': relative})
                    body += '<tr><td><a href="%s">%s/</a></td><td>Folder</td><td></td></tr>' % (html.escape(link), html.escape(path.name))
                else:
                    link = '/__h3/download?' + urlencode({'path': relative})
                    body += '<tr><td>%s</td><td>%.2f MB</td><td><a href="%s">Download</a></td></tr>' % (html.escape(path.name), path.stat().st_size / 1e6, html.escape(link))
            except OSError:
                continue
        body += '</table>'
        if (page+1)*200 < len(items):
            body += '<p><a href="/__h3/files?%s">Next page</a></p>' % html.escape(urlencode({'dir': rel, 'page': page+1}))
        return web.Response(text=document('Studio file browser', body), content_type='text/html')

    async def upload(self, request):
        directory = self.file_path(request.query.get('dir', ''))
        if not directory.is_dir():
            raise web.HTTPNotFound(text='Choose an existing folder first.')
        reader = await request.multipart()
        part = await reader.next()
        if not part or part.name != 'file' or not part.filename:
            raise web.HTTPBadRequest(text='Choose a file to upload.')
        filename = part.filename
        if filename != Path(filename).name or filename in ('.', '..') or '\\' in filename or '\0' in filename:
            raise web.HTTPBadRequest(text='Invalid filename.')
        target = directory / filename
        if target.exists() or target.is_symlink():
            raise web.HTTPConflict(text='That filename already exists. Rename your upload; the existing file was not changed.')
        temporary = directory / ('.upload-' + secrets.token_hex(16))
        size = 0
        try:
            with temporary.open('xb') as stream:
                while True:
                    chunk = await part.read_chunk(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 4 * 1024 ** 3:
                        raise web.HTTPRequestEntityTooLarge(max_size=4*1024**3, actual_size=size)
                    stream.write(chunk)
            # Exclusive publication: a concurrent upload cannot overwrite another file.
            os.link(temporary, target)
        except FileExistsError:
            raise web.HTTPConflict(text='A file with that name appeared during upload; it was not overwritten.')
        finally:
            temporary.unlink(missing_ok=True)
        raise web.HTTPSeeOther(location='/__h3/files?' + urlencode({'dir': request.query.get('dir', '')}))

    async def chunk_upload(self, request):
        now = time.monotonic()
        for key, row in list(self.uploads.items()):
            if row['updated'] < now - 3600:
                row['temporary'].unlink(missing_ok=True)
                self.uploads.pop(key, None)
        owner = request.cookies.get(COOKIE, '')
        if request.path == '/__h3/upload/init':
            data = await request.json()
            name, size = data.get('name'), data.get('size')
            if (not isinstance(name, str) or not name or name != Path(name).name or name in ('.','..')
                    or '\\' in name or '\0' in name or not isinstance(size, int) or not 0 <= size <= 4*1024**3):
                raise web.HTTPBadRequest(text='Invalid upload name or size (4 GB maximum).')
            directory = self.file_path(str(data.get('dir','')))
            if not directory.is_dir():raise web.HTTPNotFound(text='Upload directory not found.')
            target = directory / name
            if target.exists() or target.is_symlink():raise web.HTTPConflict(text='That filename exists. Rename the upload first.')
            if len(self.uploads) >= 16:raise web.HTTPTooManyRequests(text='Too many unfinished uploads. Retry after their cleanup interval.')
            if shutil.disk_usage(directory).free < size + 64*1024**2:raise web.HTTPInsufficientStorage(text='Not enough persistent disk space.')
            upload_id = secrets.token_hex(16)
            temporary = directory / ('.upload-' + upload_id)
            temporary.open('xb').close()
            self.uploads[upload_id] = dict(temporary=temporary,target=target,size=size,offset=0,
                                         owner=owner,updated=now,lock=asyncio.Lock())
            return web.json_response({'id':upload_id,'offset':0})
        upload_id = request.path.rsplit('/',1)[-1]
        row = self.uploads.get(upload_id)
        if not row or not hmac.compare_digest(row['owner'],owner):raise web.HTTPNotFound(text='Upload session expired.')
        async with row['lock']:
            row['updated'] = now
            if request.path.startswith('/__h3/upload/finish/'):
                if row['offset'] != row['size']:raise web.HTTPConflict(text='Upload is incomplete.')
                # Recheck confinement immediately before publishing the file.
                if not row['target'].parent.resolve().is_relative_to(self.root):raise web.HTTPForbidden()
                try:os.link(row['temporary'],row['target'])
                except FileExistsError:raise web.HTTPConflict(text='Destination exists; it was not overwritten.')
                row['temporary'].unlink();self.uploads.pop(upload_id,None)
                return web.json_response({'ok':True})
            try:offset=int(request.query.get('offset','-1'))
            except ValueError:raise web.HTTPBadRequest(text='Invalid chunk offset.')
            if offset != row['offset']:raise web.HTTPConflict(text='Chunk offset mismatch; restart this upload with a new filename.')
            written=0
            try:
                with row['temporary'].open('r+b') as stream:
                    stream.seek(offset)
                    async for chunk in request.content.iter_chunked(256*1024):
                        written+=len(chunk)
                        if written > 8*1024**2 or offset+written > row['size']:
                            raise web.HTTPBadRequest(text='Chunk exceeds upload bounds.')
                        stream.write(chunk)
            except BaseException:
                with row['temporary'].open('r+b') as stream:stream.truncate(offset)
                raise
            row['offset']+=written
            return web.json_response({'offset':row['offset']})

    async def diagnostics(self, request):
        rows = []
        for name, url in [('ComfyUI', self.config['comfy_upstream'] + '/system_stats'),
                          ('Jupyter', self.config['jupyter_upstream'] + '/jupyter/api/status')]:
            try:
                headers = {'Authorization': 'token ' + self.config['jupyter_token']} if name == 'Jupyter' else {}
                async with self.client.get(url, headers=headers, timeout=ClientTimeout(total=4)) as result:
                    rows.append((name, str(result.status)))
            except Exception:
                rows.append((name, 'not ready'))
        body = '<h1>Connection diagnostics</h1><p>Studio %s. A 200 means the internal HTTP service answered, not that GPU generation was tested.</p>' % VERSION
        body += ''.join('<p><strong>%s:</strong> %s</p>' % pair for pair in rows)
        body += '<p><a href="/__h3/files">Files</a> | <a href="/__h3/home">Home</a></p>'
        body += '<p>Logs: <code>h3-still/logs</code>. GPU billing continues until the Pod is stopped.</p>'
        return web.Response(text=document('Connection diagnostics', body), content_type='text/html')

    async def proxy(self, request):
        upstream = self.config['comfy_upstream'] if self.service == 'comfy' else self.config['jupyter_upstream']
        url = upstream + str(request.rel_url)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP | {'authorization', 'cookie', 'accept-encoding', 'x-forwarded-host', 'x-forwarded-proto'}}
        headers['Host'] = request.host
        headers['X-Forwarded-Proto'] = 'https' if self.secure else 'http'
        # Private token is attached only server-side, never exposed to browser JS or URLs.
        if self.service == 'jupyter':
            headers['Authorization'] = 'token ' + self.config['jupyter_token']
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            if not same_origin(request):
                raise web.HTTPForbidden(text='WebSocket origin check failed.')
            protocols = [p.strip() for p in request.headers.get('Sec-WebSocket-Protocol', '').split(',') if p.strip()]
            for name in list(headers):
                if name.lower().startswith('sec-websocket'):
                    del headers[name]
            try:
                other = await self.client.ws_connect(url, headers=headers, protocols=protocols,
                    autoclose=True, autoping=True, max_msg_size=64*1024**2, heartbeat=25)
            except Exception:
                raise web.HTTPBadGateway(text='Backend WebSocket is not ready. Open Connection diagnostics.')
            browser = web.WebSocketResponse(protocols=[other.protocol] if other.protocol else (),
                                           heartbeat=25, max_msg_size=64*1024**2)
            await browser.prepare(request)
            async def relay(source, target):
                async for message in source:
                    if message.type == WSMsgType.TEXT:
                        await target.send_str(message.data)
                    elif message.type == WSMsgType.BINARY:
                        await target.send_bytes(message.data)
                    elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                        break
            tasks = [asyncio.create_task(relay(browser, other)), asyncio.create_task(relay(other, browser))]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await browser.close()
                await other.close()
            return browser
        try:
            result = await self.client.request(request.method, url, headers=headers,
                data=request.content.iter_chunked(1024*1024) if request.can_read_body else None,
                allow_redirects=False, timeout=ClientTimeout(total=None, sock_connect=10, sock_read=3600))
        except Exception:
            return web.Response(status=503, text=document('Service starting',
                '<h1>Service is starting or stopped</h1><p>The login and file browser remain available.</p><p><a href="/__h3/diagnostics">Connection diagnostics</a></p>'), content_type='text/html')
        async with result:
            response_headers = {k: v for k, v in result.headers.items()
                                if k.lower() not in HOP | {'set-cookie', 'www-authenticate', 'content-encoding'}}
            response_headers.update(NO_CACHE)
            response = web.StreamResponse(status=result.status, headers=response_headers)
            await response.prepare(request)
            async for chunk in result.content.iter_chunked(256*1024):
                await response.write(chunk)
            await response.write_eof()
            return response

    async def dispatch(self, request):
        path = request.path
        if path == '/__h3/health':
            return web.json_response({'gateway': 'ready', 'version': VERSION})
        if path == '/__h3/login':
            return await self.login(request) if request.method == 'POST' else self.login_page(request)
        if not self.authenticated(request):
            if request.method in ('GET', 'HEAD') and ('text/html' in request.headers.get('Accept', '') or path == '/'):
                return self.login_page(request)
            return web.json_response({'error': 'Sign in at /__h3/login'}, status=401)
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and not same_origin(request):
            raise web.HTTPForbidden(text='Request origin check failed.')
        if path == '/__h3/logout' and request.method == 'POST':
            response = web.HTTPSeeOther(location='/__h3/login')
            response.del_cookie(COOKIE, path='/')
            return response
        if path == '/__h3/home' or (self.service == 'jupyter' and path == '/'):
            return self.home()
        if path == '/__h3/files':
            return self.files(request)
        if path.startswith('/__h3/upload/') and request.method in ('POST','PATCH'):
            return await self.chunk_upload(request)
        if path == '/__h3/upload' and request.method == 'POST':
            return await self.upload(request)
        if path == '/__h3/download':
            target = self.file_path(request.query.get('path', ''))
            if not target.is_file():
                raise web.HTTPNotFound(text='File not found.')
            return web.FileResponse(target, headers={'Content-Disposition': "attachment; filename*=UTF-8''" + quote(target.name)})
        if path == '/__h3/diagnostics':
            return await self.diagnostics(request)
        if path.startswith('/__h3/'):
            raise web.HTTPNotFound()
        if self.service == 'jupyter' and not path.startswith('/jupyter/'):
            raise web.HTTPSeeOther(location='/__h3/home')
        return await self.proxy(request)

    def application(self):
        @web.middleware
        async def security(request, handler):
            try:
                response = await handler(request)
            except web.HTTPException as error:
                response = error
            if not response.prepared:
                response.headers.update(NO_CACHE)
                response.headers['X-Frame-Options'] = 'SAMEORIGIN'
            return response
        app = web.Application(middlewares=[security], client_max_size=4*1024**3)
        app.router.add_route('*', '/{path:.*}', self.dispatch)
        async def context(app):
            self.client = ClientSession(auto_decompress=True, trust_env=False)
            yield
            await self.client.close()
        app.cleanup_ctx.append(context)
        return app


async def serve(config):
    runners = []
    tls = None
    if config.get('test_tls_cert'):
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(config['test_tls_cert'], config['test_tls_key'])
    try:
        for name, default_port in [('comfy', 8188), ('jupyter', 8888)]:
            gateway = Gateway(config, name)
            runner = web.AppRunner(gateway.application(), access_log=None)
            await runner.setup()
            await web.TCPSite(runner, config.get('listen', '0.0.0.0'), config.get(name + '_port', default_port), ssl_context=tls).start()
            runners.append(runner)
        print('H3 sign-in gateways ready on ports 8188 and 8888. Use your comfy_password; no username.', flush=True)
        await asyncio.Event().wait()
    finally:
        for runner in runners:
            await runner.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(serve(json.loads(args.config.read_text())))

if __name__ == '__main__':
    main()
