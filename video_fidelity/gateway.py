"""Authenticated streaming proxy. ComfyUI itself binds to loopback only."""
from __future__ import annotations
import asyncio
import base64
import hmac
import os
import re
from urllib.parse import urlsplit
from aiohttp import web, ClientSession, ClientTimeout, WSMsgType
from fv_common import terminate

HOP={'connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailer',
     'transfer-encoding','upgrade','content-length'}


def validate_password(password: str) -> str:
    if len(password.encode('utf-8')) < 16 or len(password.encode('utf-8')) > 128 or '{{' in password:
        raise ValueError('Set COMFY_PASSWORD to a resolved, unique password of 16-128 UTF-8 bytes.')
    return password


def authorized(header: str, password: str) -> bool:
    try:
        scheme,data=header.split(' ',1)
        if scheme.lower()!='basic':
            return False
        value=base64.b64decode(data,validate=True)
        return hmac.compare_digest(value,('workbench:'+password).encode())
    except (ValueError,TypeError):
        return False


def same_origin(origin: str, host: str, pod_id: str = '', public_origin: str = '') -> bool:
    """Accept local same-origin requests or THIS RunPod pod's HTTPS origin.

    RunPod can rewrite the Host header when proxying to port 8188. Never
    trust X-Forwarded-Host blindly: it may be supplied by a client. Instead,
    derive the one allowed external origin from the server-owned pod ID.
    An explicit FAITHFUL_PUBLIC_ORIGIN can be used when the pod ID is absent.
    """
    if not origin:
        return True
    try:
        url = urlsplit(origin)
        if (url.scheme not in ('http', 'https') or not url.netloc or
                url.username is not None or url.password is not None or
                url.path or url.query or url.fragment):
            return False
        if url.netloc.lower() == host.lower():
            return True
        normalized = origin.lower()
        if (pod_id and re.fullmatch(r'[a-z0-9]{6,64}', pod_id.lower()) and
                normalized == f'https://{pod_id.lower()}-8188.proxy.runpod.net'):
            return True
        # Optional explicit operator configuration for nonstandard Pod setups.
        if (public_origin and
                re.fullmatch(r'https://[a-z0-9]{6,64}-8188\.proxy\.runpod\.net',
                             public_origin.lower()) and
                normalized == public_origin.lower()):
            return True
    except ValueError:
        return False
    return False


def create_app(password: str, comfy_process=None) -> web.Application:
    password=validate_password(password)
    @web.middleware
    async def guard(request,handler):
        if request.path=='/_health':
            alive=comfy_process is None or comfy_process.poll() is None
            return web.Response(text='ready' if alive else 'unavailable',status=200 if alive else 503)
        if not authorized(request.headers.get('Authorization',''),password):
            return web.Response(status=401,text='Authentication required',
                headers={'WWW-Authenticate':'Basic realm="Faithful Video", charset="UTF-8"','Cache-Control':'no-store'})
        if not same_origin(request.headers.get('Origin',''), request.host,
                           pod_id=os.environ.get('RUNPOD_POD_ID',''),
                           public_origin=os.environ.get('FAITHFUL_PUBLIC_ORIGIN','')):
            raise web.HTTPForbidden(text='Cross-origin request refused')
        return await handler(request)
    app=web.Application(middlewares=[guard],client_max_size=20*1024**3)
    async def context(app):
        app['client']=ClientSession(timeout=ClientTimeout(total=None,sock_connect=20),auto_decompress=False)
        async def watch():
            while True:
                await asyncio.sleep(2)
                if comfy_process is not None and comfy_process.poll() is not None:
                    raise web.GracefulExit()
        task=asyncio.create_task(watch())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
            await app['client'].close()
            if comfy_process is not None:
                terminate(comfy_process,group=True)
    app.cleanup_ctx.append(context)
    async def proxy(request):
        target='http://127.0.0.1:8189'+request.raw_path
        headers={k:v for k,v in request.headers.items()
                 if k.lower() not in HOP|{'authorization','host','origin','referer','accept-encoding'}}
        client=request.app['client']
        if request.headers.get('Upgrade','').lower()=='websocket':
            async with client.ws_connect(target,headers=headers,max_msg_size=64*1024**2) as upstream:
                downstream=web.WebSocketResponse(max_msg_size=64*1024**2)
                await downstream.prepare(request)
                async def relay(src,dst):
                    async for msg in src:
                        if msg.type==WSMsgType.TEXT:await dst.send_str(msg.data)
                        elif msg.type==WSMsgType.BINARY:await dst.send_bytes(msg.data)
                        elif msg.type in (WSMsgType.CLOSE,WSMsgType.CLOSED,WSMsgType.ERROR):break
                tasks=[asyncio.create_task(relay(downstream,upstream)),asyncio.create_task(relay(upstream,downstream))]
                try:
                    await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in tasks:task.cancel()
                    await asyncio.gather(*tasks,return_exceptions=True)
                    await downstream.close()
                return downstream
        async def body():
            size=0
            async for chunk in request.content.iter_chunked(1024*1024):
                size+=len(chunk)
                if size>20*1024**3:
                    raise web.HTTPRequestEntityTooLarge(max_size=20*1024**3,actual_size=size)
                yield chunk
        async with client.request(request.method,target,headers=headers,
                                  data=body() if request.can_read_body else None,
                                  allow_redirects=False) as upstream:
            response=web.StreamResponse(status=upstream.status,
                headers={k:v for k,v in upstream.headers.items() if k.lower() not in HOP})
            response.headers['X-Content-Type-Options']='nosniff'
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(1024*1024):
                await response.write(chunk)
            await response.write_eof()
            return response
    app.router.add_route('*','/{path:.*}',proxy)
    return app
