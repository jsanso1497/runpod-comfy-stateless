"""Verified, loopback-only handoff between Reference Pack/Ollama and ComfyUI.

Also copied into the pinned Reference Pack by patch_refpack_local.py. This file
has no package-relative imports, so either package imports the identical logic.
"""
from __future__ import annotations
import functools
import inspect
import time
from urllib.parse import urlsplit
import requests

BASE = 'http://127.0.0.1:11434'


def session():
    s = requests.Session()
    s.trust_env = False
    return s


def local_base(value):
    p = urlsplit(value)
    if p.scheme != 'http' or p.hostname not in ('127.0.0.1', 'localhost') or p.port != 11434:
        raise ValueError('This build uses the private Ollama server: http://127.0.0.1:11434/v1 only.')
    if p.username or p.password or p.query or p.fragment or p.path.rstrip('/') != '/v1':
        raise ValueError('Use api_base=http://127.0.0.1:11434/v1 with no credentials, query or extra path.')
    return BASE + '/v1'


def api(s, path, body=None, timeout=30):
    method = s.get if body is None else s.post
    kw = {} if body is None else {'json': body}
    try:
        with method(BASE + path, timeout=(5, timeout), allow_redirects=False, **kw) as r:
            if r.status_code != 200:
                raise RuntimeError(f'Private Ollama {path}: HTTP {r.status_code}. Check /workspace/h3-portrait/ollama-status.json.')
            data = r.json()
    except requests.RequestException as exc:
        raise RuntimeError('Private Ollama is unavailable at 127.0.0.1:11434. '
                           'Check /workspace/h3-portrait/ollama-status.json before generating.') from exc
    if not isinstance(data, dict):
        raise RuntimeError('Private Ollama returned a malformed response.')
    return data


def canonical(model):
    return model if ':' in model.rsplit('/', 1)[-1] else model + ':latest'


def ensure_installed(s, model):
    if not isinstance(model, str) or not model.strip():
        raise ValueError('local_model_slug is empty. Select the installed Instruct model.')
    tags = api(s, '/api/tags').get('models', [])
    if not any(canonical(x.get('name') or x.get('model', '')) == canonical(model) for x in tags):
        raise RuntimeError(f'Prompt model is not ready: {model}. Wait for H3 PORTRAIT OLLAMA READY; '
                           'an in-progress download does not appear in ollama list.')
    info = api(s, '/api/show', {'model': model})
    if info.get('remote_host') or info.get('remote_model') or 'vision' not in info.get('capabilities', []):
        raise ValueError('Reference Pack requires an installed LOCAL vision model, not a cloud model.')
    if 'thinking' in model.rsplit(':', 1)[-1].lower():
        raise ValueError('Reference Pack uses the Instruct model, not the Thinking director.')


def require_idle(s=None):
    """Do not silently evict a different user request from the private daemon."""
    if s is None:
        with session() as own:
            return require_idle(own)
    running = api(s, '/api/ps').get('models', [])
    if running:
        names = ', '.join(x.get('name') or x.get('model', '?') for x in running)
        raise RuntimeError('Ollama is still using GPU memory: ' + names +
                           '. Stop that request/model before H3 rendering; H3 was not started.')
    return True


def unload(s, model, timeout=90, clock=time.monotonic, sleep=time.sleep):
    api(s, '/api/generate', {'model': model, 'stream': False, 'keep_alive': 0}, timeout=min(timeout, 60))
    deadline = clock() + timeout
    while clock() < deadline:
        running = api(s, '/api/ps', timeout=10).get('models', [])
        if not running:
            print('H3 GPU HANDOFF VERIFIED: Ollama /api/ps is empty.', flush=True)
            return
        if any(canonical(x.get('name') or x.get('model', '')) != canonical(model) for x in running):
            raise RuntimeError('Another Ollama request started during GPU handoff. H3 was not started.')
        sleep(.25)
    raise RuntimeError('Ollama did not release the prompt model within the handoff timeout. H3 was not started.')


def release_comfy():
    import comfy.model_management as mm
    mm.unload_all_models()
    mm.soft_empty_cache()
    mm.throw_exception_if_processing_interrupted()


def guard_refpack(original):
    """The native media path stays unchanged; wrap actual prompt execution."""
    signature = inspect.signature(original)
    @functools.wraps(original)
    def guarded(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        provider = bound.arguments.get('prompt_provider', 'local')
        if provider not in ('local', 'none'):
            raise ValueError('Only local or none is allowed; hosted prompt models are disabled.')
        if provider == 'none':
            return original(*bound.args, **bound.kwargs)
        base = local_base(bound.arguments.get('api_base', ''))
        model = bound.arguments.get('local_model_slug', '')
        bound.arguments['api_base'] = base
        if 'openrouter_api_key' in bound.arguments:
            bound.arguments['openrouter_api_key'] = ''
        with session() as s:
            ensure_installed(s, model)
            # A previous attempt at this same prompt may have left its runner resident.
            running = api(s, '/api/ps').get('models', [])
            if any(canonical(r.get('name') or r.get('model', '')) != canonical(model) for r in running):
                require_idle(s)
            release_comfy()
            try:
                return original(*bound.args, **bound.kwargs)
            finally:
                # Includes successful prompts, empty completions, errors and cancellation.
                # Failure here deliberately blocks downstream native H3 encoding.
                unload(s, model)
    guarded._h3_handoff_version = '1.5.3'
    return guarded
