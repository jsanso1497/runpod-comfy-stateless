"""Browser-safe transport for the fixed, loopback ComfyUI upstream.

No model loading, third-party relay, dependency installation, or account key
handling lives here. WB_AUTH_MODE=none is an explicit public-access opt-in.
"""
from __future__ import annotations

import asyncio
import os
import re
from urllib.parse import urlsplit

from aiohttp import (
    ClientError, ClientSession, ClientTimeout, DummyCookieJar, WSMsgType, web,
)
from multidict import CIMultiDict
from yarl import URL

ACCESS_VERSION = '1.2.0'
COMFY_UPSTREAM = 'http://127.0.0.1:8189'
AUTH_MODE_KEY = web.AppKey('workbench.auth_mode', str)
PASSWORD_KEY = web.AppKey('workbench.password', str)
HOP_HEADERS = frozenset({
    'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'te', 'trailer', 'trailers', 'transfer-encoding', 'upgrade',
})


def auth_mode(value: str | None = None) -> str:
    """Never silently fall back to public access on a typo or missing password."""
    mode = os.environ.get('WB_AUTH_MODE', 'basic') if value is None else value
    if mode not in ('basic', 'none'):
        raise ValueError('WB_AUTH_MODE must be basic or none.')
    return mode


def _origin_authority(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            return None
        # Access .port to reject malformed ports rather than raising in middleware.
        parsed.port
        return parsed.netloc.lower()
    except ValueError:
        return None


def same_origin(request: web.Request) -> bool:
    """Retain browser cross-origin write and WebSocket protection in both modes.

    RunPod may forward an internal Host. Its public origin is derived from the
    actual Pod ID, not an arbitrary X-Forwarded-Host supplied by a client.
    This is a CSRF guard, not authentication in no-login mode.
    """
    is_ws = request.headers.get('Upgrade', '').lower() == 'websocket'
    if request.method in ('GET', 'HEAD') and not is_ws:
        return True
    if request.headers.get('Sec-Fetch-Site', '').lower() == 'cross-site':
        return False
    origin = request.headers.get('Origin')
    if not origin:
        return True  # Native clients and the Comfy API can omit Origin.
    authority = _origin_authority(origin)
    if authority is None:
        return False
    allowed = {request.host.lower()}
    pod_id = os.environ.get('RUNPOD_POD_ID', '')
    if re.fullmatch(r'[A-Za-z0-9]+', pod_id):
        allowed.add(f'{pod_id.lower()}-8188.proxy.runpod.net')
    return authority in allowed


def _hop_names(headers) -> set[str]:
    blocked = set(HOP_HEADERS)
    for value in headers.getall('Connection', []):
        blocked.update(token.strip().lower() for token in value.split(',') if token.strip())
    return blocked


def request_headers(request: web.Request, *, websocket: bool = False) -> CIMultiDict:
    blocked = _hop_names(request.headers) | {
        'host', 'origin', 'expect', 'accept-encoding',
        'forwarded', 'x-forwarded-host', 'x-forwarded-proto', 'x-forwarded-for',
    }
    if websocket:
        blocked.add('content-length')
    headers = CIMultiDict()
    for name, value in request.headers.items():
        key = name.lower()
        if key in blocked or key.startswith('sec-websocket-') or key.startswith('sec-fetch-'):
            continue
        # Do not pass a cached Workbench Basic password to ComfyUI. Preserve
        # non-Basic authorization for native endpoint integrations, if present.
        if key == 'authorization' and value.partition(' ')[0].lower() == 'basic':
            continue
        headers.add(name, value)
    # Normalize browser/curl differences. ComfyUI can serve uncompressed assets;
    # the external HTTPS proxy remains free to compress the final response.
    headers['Accept-Encoding'] = 'identity'
    return headers


def response_headers(headers) -> CIMultiDict:
    blocked = _hop_names(headers)
    result = CIMultiDict()
    for name, value in headers.items():
        if name.lower() not in blocked:
            result.add(name, value)
    # Preserve Content-Length, Content-Encoding, Content-Range, and repeated
    # Set-Cookie headers. The bytes are streamed without decompression/rewrite.
    return result


def session() -> ClientSession:
    """No shared user cookies; no environment proxy or implicit .netrc auth."""
    return ClientSession(
        timeout=ClientTimeout(total=None, sock_connect=10),
        auto_decompress=False,
        cookie_jar=DummyCookieJar(),
        trust_env=False,
        headers={'Accept-Encoding': 'identity'},
    )


async def on_prepare(request: web.Request, response: web.StreamResponse) -> None:
    """Set headers BEFORE they are sent, including for streamed/WS responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Workbench-Gateway'] = ACCESS_VERSION
    response.headers['X-Workbench-Auth'] = request.app.get(AUTH_MODE_KEY, request.app.get('auth_mode', 'basic'))


def unavailable(status: int = 503) -> web.Response:
    return web.Response(
        status=status,
        text='ComfyUI is not responding yet. Refresh this page after it starts. '
             'Check the Pod logs if it remains unavailable.',
        headers={'Retry-After': '3', 'Cache-Control': 'no-store'},
    )


async def _websocket(request: web.Request, client: ClientSession, url: URL) -> web.StreamResponse:
    protocols = tuple(p.strip() for p in request.headers.get('Sec-WebSocket-Protocol', '').split(',') if p.strip())
    try:
        upstream = await client.ws_connect(
            url, headers=request_headers(request, websocket=True),
            protocols=protocols, heartbeat=30, max_msg_size=0, compress=0,
        )
    except (ClientError, OSError, asyncio.TimeoutError):
        return unavailable()
    downstream = web.WebSocketResponse(
        heartbeat=30, max_msg_size=0, compress=False,
        protocols=(upstream.protocol,) if upstream.protocol else (),
    )
    tasks = []
    try:
        await downstream.prepare(request)

        async def relay(source, destination):
            async for message in source:
                if message.type == WSMsgType.TEXT:
                    await destination.send_str(message.data)
                elif message.type == WSMsgType.BINARY:
                    await destination.send_bytes(message.data)
                elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break

        tasks = [asyncio.create_task(relay(downstream, upstream)),
                 asyncio.create_task(relay(upstream, downstream))]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        code = upstream.close_code or downstream.close_code or 1000
        if code in (1005, 1006, 1015):
            code = 1001
        await upstream.close()
        await downstream.close(code=code)
    return downstream


async def proxy(request: web.Request, client: ClientSession,
                upstream_base: str = COMFY_UPSTREAM) -> web.StreamResponse:
    """Forward to a code-selected loopback server, never to a user-supplied URL."""
    url = URL(upstream_base.rstrip('/') + request.rel_url.raw_path_qs, encoded=True)
    if request.headers.get('Upgrade', '').lower() == 'websocket':
        return await _websocket(request, client, url)

    limit = request.client_max_size
    if limit and request.content_length is not None and request.content_length > limit:
        raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=request.content_length)
    too_large = False

    async def body():
        nonlocal too_large
        received = 0
        async for chunk in request.content.iter_chunked(1024 * 1024):
            received += len(chunk)
            if limit and received > limit:
                too_large = True
                raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=received)
            yield chunk

    downstream = None
    try:
        async with client.request(
            request.method, url, headers=request_headers(request),
            # A GET/HEAD without a body must NOT become a chunked-body request.
            data=body() if request.can_read_body else None,
            allow_redirects=False,
        ) as upstream:
            downstream = web.StreamResponse(
                status=upstream.status, headers=response_headers(upstream.headers),
            )
            await downstream.prepare(request)
            if request.method != 'HEAD' and upstream.status not in (204, 304):
                async for chunk in upstream.content.iter_chunked(1024 * 1024):
                    await downstream.write(chunk)
            await downstream.write_eof()
            return downstream
    except (ClientError, OSError, asyncio.TimeoutError):
        if downstream is not None and downstream.prepared:
            # Never append a second HTTP status/header block to a partial body.
            if request.transport is not None:
                request.transport.close()
            return downstream
        if too_large:
            raise web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=limit + 1)
        return unavailable()
