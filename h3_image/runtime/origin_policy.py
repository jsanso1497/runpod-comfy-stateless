"""Exact browser-origin matching across TLS-terminating proxies.

Public targets come only from server configuration or RUNPOD_POD_ID. Never
trust a browser-supplied X-Forwarded-Host/Forwarded header as configuration.
"""
import ipaddress
import os
import re
from urllib.parse import urlsplit

SERVICES = {'comfy': (8188, 'H3_COMFY_PUBLIC_ORIGIN'),
            'jupyter': (8888, 'H3_FILES_PUBLIC_ORIGIN')}


def normalize_origin(value, allow_path=False):
    """Return (scheme, hostname, effective port), rejecting ambiguous input."""
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError('Missing or invalid origin.')
    if any(ord(c) <= 32 or ord(c) == 127 for c in value) or '\\' in value or ',' in value:
        raise ValueError('Origin contains invalid characters.')
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or not parsed.hostname:
        raise ValueError('Origin must use HTTP or HTTPS and an explicit host.')
    if parsed.username is not None or parsed.password is not None or '@' in parsed.netloc:
        raise ValueError('User information is not allowed in origins.')
    if not allow_path and (parsed.path not in ('', '/') or parsed.query or parsed.fragment):
        raise ValueError('Use the service origin only, with no path or query.')
    host = parsed.hostname.lower()
    if ':' in host:
        host = str(ipaddress.IPv6Address(host))
    else:
        host = host.encode('idna').decode('ascii')
        labels = host.split('.')
        if any(not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label)
               for label in labels):
            raise ValueError('Invalid origin hostname.')
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == 'https' else 80)
    if not 1 <= port <= 65535 or parsed.netloc.endswith(':'):
        raise ValueError('Invalid origin port.')
    return parsed.scheme, host, port


def format_origin(parts):
    scheme, host, port = parts
    authority = '[' + host + ']' if ':' in host else host
    if port != (443 if scheme == 'https' else 80):
        authority += ':' + str(port)
    return scheme + '://' + authority


def public_origins(environ=None):
    """Infer only this Pod's two exact origins; explicit HTTPS overrides win."""
    env = os.environ if environ is None else environ
    pod = env.get('RUNPOD_POD_ID', '')
    if pod and not re.fullmatch(r'[A-Za-z0-9]{1,64}', pod):
        raise ValueError('RUNPOD_POD_ID has an unexpected format.')
    result = {}
    for service, (port, variable) in SERVICES.items():
        origin = env.get(variable, '')
        if not origin and pod:
            origin = 'https://%s-%s.proxy.runpod.net' % (pod.lower(), port)
        if origin:
            parsed = normalize_origin(origin)
            if parsed[0] != 'https':
                raise ValueError(variable + ' must use HTTPS.')
            result[service] = format_origin(parsed)
    return result


def source_origin(request):
    # An explicit invalid Origin (including null) never falls back to Referer.
    origins = request.headers.getall('Origin', [])
    if origins:
        if len(origins) != 1:
            raise ValueError('Multiple Origin headers are not allowed.')
        return normalize_origin(origins[0])
    referers = request.headers.getall('Referer', [])
    if len(referers) != 1:
        raise ValueError('No unambiguous Origin or Referer.')
    return normalize_origin(referers[0], allow_path=True)


def matches_origin(request, target):
    try:
        if request.headers.get('Sec-Fetch-Site', '').lower() == 'cross-site':
            return False
        return source_origin(request) == normalize_origin(target)
    except (ValueError, TypeError, UnicodeError):
        return False
