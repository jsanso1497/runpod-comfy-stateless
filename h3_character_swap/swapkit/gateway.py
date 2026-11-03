"""Authenticated streaming gateway to one fixed loopback ComfyUI instance."""
from __future__ import annotations
import asyncio
import base64
import hmac
import html
import json
import os
import re
from urllib.parse import urlsplit
from aiohttp import web, ClientSession, ClientTimeout, DummyCookieJar, WSMsgType, ClientError
from multidict import CIMultiDict
from yarl import URL
from .config import ROOT, upload_mb

UPSTREAM = "http://127.0.0.1:8189"
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"}


def authorized(header, password):
    try:
        scheme, token = header.split(" ", 1)
        username, value = base64.b64decode(token, validate=True).decode().split(":", 1)
        return scheme.lower() == "basic" and hmac.compare_digest(username, "workbench") and hmac.compare_digest(value.encode(), password.encode())
    except (ValueError, UnicodeError):
        return False


def same_origin(request):
    if request.method in ("GET", "HEAD") and request.headers.get("Upgrade", "").lower() != "websocket":
        return True
    if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
        return False
    origin = request.headers.get("Origin")
    if not origin:
        return True
    p = urlsplit(origin)
    if p.scheme not in ("http", "https") or not p.netloc or p.username or p.password or p.path not in ("", "/") or p.query or p.fragment:
        return False
    allowed = {request.host.lower()}
    pod = os.environ.get("RUNPOD_POD_ID", "")
    if re.fullmatch(r"[A-Za-z0-9]+", pod):
        allowed.add(pod.lower() + "-8188.proxy.runpod.net")
    return p.netloc.lower() in allowed


def filtered(headers, request=False):
    blocked = set(HOP)
    for value in headers.getall("Connection", []):
        blocked.update(x.strip().lower() for x in value.split(","))
    if request:
        blocked |= {"authorization", "host", "origin", "accept-encoding", "expect", "forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "x-h3-filebrowser-user"}
    out = CIMultiDict()
    for k,v in headers.items():
        if k.lower() in blocked or (request and (k.lower().startswith("sec-websocket-") or k.lower().startswith("sec-fetch-"))):
            continue
        out.add(k,v)
    if request:
        out["Accept-Encoding"] = "identity"
    return out


async def proxy(request, *, upstream_base=UPSTREAM, identity=None):
    session = request.app["session"]
    outgoing = filtered(request.headers, True)
    if identity is not None:
        outgoing["X-H3-Filebrowser-User"] = identity
    url = URL(upstream_base + request.rel_url.raw_path_qs, encoded=True)
    if request.headers.get("Upgrade", "").lower() == "websocket":
        protocols = tuple(x.strip() for x in request.headers.get("Sec-WebSocket-Protocol", "").split(",") if x.strip())
        try:
            up = await session.ws_connect(url, headers=outgoing, protocols=protocols, heartbeat=30, max_msg_size=0, compress=0)
        except (ClientError, OSError, asyncio.TimeoutError):
            return web.Response(status=503, text="ComfyUI is starting. See /_swap/status")
        down = web.WebSocketResponse(protocols=(up.protocol,) if up.protocol else (), heartbeat=30, max_msg_size=0, compress=False)
        await down.prepare(request)
        async def relay(a,b):
            async for msg in a:
                if msg.type == WSMsgType.TEXT: await b.send_str(msg.data)
                elif msg.type == WSMsgType.BINARY: await b.send_bytes(msg.data)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR): break
        tasks = [asyncio.create_task(relay(up,down)), asyncio.create_task(relay(down,up))]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks: t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await up.close(); await down.close()
        return down
    limit = request.app["upload_limit"]
    if request.content_length and request.content_length > limit:
        raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=request.content_length)
    async def body():
        total = 0
        async for part in request.content.iter_chunked(1024*1024):
            total += len(part)
            if total > limit:
                raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=total)
            yield part
    downstream = None
    try:
        async with session.request(request.method,url,headers=outgoing,data=body() if request.can_read_body else None,allow_redirects=False) as upstream:
            downstream=web.StreamResponse(status=upstream.status,headers=filtered(upstream.headers))
            await downstream.prepare(request)
            if request.method != "HEAD" and upstream.status not in (204,304):
                async for chunk in upstream.content.iter_chunked(1024*1024):
                    await downstream.write(chunk)
            await downstream.write_eof()
            return downstream
    except (ClientError,OSError,asyncio.TimeoutError):
        if downstream and downstream.prepared:
            if request.transport: request.transport.close()
            return downstream
        return web.Response(status=503,text="ComfyUI is starting or stopped. Check Pod logs and /_swap/status.",headers={"Retry-After":"5"})


def make_app(password, status):
    if len(password) < 16 or password.lstrip().startswith("{{"):
        raise ValueError("WB_PASSWORD must resolve to a password of at least 16 characters")
    @web.middleware
    async def security(request,handler):
        if request.path == "/healthz":
            return web.json_response({"gateway":"running", "ready":status.get("ready",False)})
        if not authorized(request.headers.get("Authorization", ""), password):
            return web.Response(status=401,text="H3 swap authentication required",headers={"WWW-Authenticate":'Basic realm="H3 character replacement", charset="UTF-8"',"Cache-Control":"no-store"})
        if not same_origin(request):
            return web.Response(status=403,text="Cross-origin writes are blocked")
        if request.path == "/prompt" and request.method == "POST" and not status.get("ready"):
            return web.json_response({"error":"Model preparation is not complete. Open /_swap/status before queuing."},status=503)
        return await handler(request)
    app=web.Application(middlewares=[security],client_max_size=upload_mb()*1024**2,handler_args={"auto_decompress":False})
    app["upload_limit"] = upload_mb()*1024**2
    async def on_start(app):
        app["session"]=ClientSession(auto_decompress=False,cookie_jar=DummyCookieJar(),trust_env=False,timeout=ClientTimeout(total=None,sock_connect=10))
    async def on_cleanup(app): await app["session"].close()
    async def headers(request,response):
        response.headers["Referrer-Policy"]="no-referrer"
        response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["Cache-Control"]="no-store"
    app.on_startup.append(on_start);app.on_cleanup.append(on_cleanup);app.on_response_prepare.append(headers)
    async def get_status(request): return web.json_response(dict(status))
    async def index(request):
        files=sorted((ROOT/"workflows").glob("*.json"))
        items="".join(f'<li><a href="/_swap/workflow/{html.escape(p.name)}">{html.escape(p.name)}</a></li>' for p in files if not p.name.endswith(".api.json"))
        return web.Response(content_type="text/html",text='<h1>H3 Character Replacement</h1><p><a href="/">Open ComfyUI</a> | <a href="/_swap/status">Preparation status</a> | <a href="/_swap/chain">Low-memory chain review</a> | <a href="/files/">FileBrowser</a> | <a href="/_swap/recovery">Saved workflows</a></p><p>Workflows are also in the ComfyUI Workflows sidebar. Review source masks and prompt drafts before rendering.</p><ul>'+items+'</ul>')
    async def graph(request):
        name=request.match_info["name"]
        if name not in {p.name for p in (ROOT/"workflows").glob("*.json")}:
            raise web.HTTPNotFound()
        return web.FileResponse(ROOT/"workflows"/name)
    from .filebrowser import enabled as files_enabled, UPSTREAM as FILES_UPSTREAM, USERNAME as FILES_USER
    async def filebrowser_proxy(request):
        if not files_enabled():
            raise web.HTTPNotFound(text="FileBrowser is disabled")
        if not status.get("filebrowser", {}).get("ready"):
            return web.Response(status=503, text="FileBrowser is starting or unavailable. Check /_swap/status and h3-swap/logs/filebrowser.log", headers={"Retry-After":"5"})
        return await proxy(request, upstream_base=FILES_UPSTREAM, identity=FILES_USER)
    async def files_redirect(request):
        if not files_enabled():
            raise web.HTTPNotFound()
        raise web.HTTPTemporaryRedirect(location="/files/")
    from .files_builtin import backend as files_backend, install as install_builtin_files
    if files_backend() == 'builtin':
        install_builtin_files(app, status)
    else:
        app.router.add_get("/files", files_redirect)
        app.router.add_route("*", "/files/{path:.*}", filebrowser_proxy)
    from .economy_web import install_routes as install_economy_routes
    install_economy_routes(app)
    from .recovery_web import install_routes as install_recovery_routes
    install_recovery_routes(app)
    from .disk_web import install_routes
    install_routes(app)
    app.router.add_get("/_swap",index)
    app.router.add_get("/_swap/status",get_status)
    app.router.add_get("/_swap/workflow/{name}",graph)
    async def front(request):
        # A ready HTTP root gives RunPod a useful service page even while H3's
        # backend is importing or its weights are downloading. Files stay usable.
        if (request.method == "GET" and request.path == "/" and
                files_backend() == "builtin" and not status.get("ready")):
            return web.Response(content_type="text/html", text=STARTUP_PAGE)
        return await proxy(request)
    app.router.add_route("*","/{path:.*}",front)
    return app


STARTUP_PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 starting</title>
<style>body{font:18px system-ui;background:#131922;color:#e6edf6;max-width:850px;margin:70px auto;padding:20px}a{color:#9dccff}pre{white-space:pre-wrap}</style>
<h1>Your workspace is running</h1><p>H3 generation is still preparing. File access is available now.</p>
<p><a href="/files/">Open files: upload or download</a> | <a href="/_swap/status">Startup details</a></p>
<pre id="status">Checking model preparation...</pre>
<script>async function update(){try{let r=await fetch('/_swap/status');let s=await r.json();document.getElementById('status').textContent=JSON.stringify(s,null,2);if(s.ready)location.reload()}catch(e){document.getElementById('status').textContent='Connection interrupted. Refresh after the Pod is running.'}}update();setInterval(update,5000)</script></html>"""
