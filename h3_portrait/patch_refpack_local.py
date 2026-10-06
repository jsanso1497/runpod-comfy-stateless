#!/usr/bin/env python3
"""Idempotent, source-checked patch of pinned Hearmeman Reference Pack 0.3.5."""
from pathlib import Path
import hashlib
import json
import sys


def replace_checked(text, old, new):
    if old in text:
        if text.count(old) != 1:
            raise ValueError('Ambiguous upstream patch anchor: '+old[:90])
        return text.replace(old, new, 1)
    if new in text:
        return text
    raise ValueError('Reference Pack source drift: '+old[:90])


def patch(root):
    root = Path(root).resolve()
    paths = {n: root/'minimax_refpack'/n for n in ('endpoint.py','nodes.py','prompt.py')}
    if any(not p.is_file() or p.is_symlink() for p in paths.values()):
        raise ValueError('Reference Pack source layout changed.')
    texts = {name: p.read_text() for name,p in paths.items()}
    e = texts['endpoint.py']
    for old,new in (
        ('PROVIDERS = ("openrouter", "local", "none")', 'PROVIDERS = ("local", "none")'),
        ('DEFAULT_PROVIDER = "openrouter"', 'DEFAULT_PROVIDER = "local"'),
        ('if raw is True:\n        return "openrouter"', 'if raw is True:\n        return "local"'),
        ('if text == "true":\n        return "openrouter"', 'if text == "true":\n        return "local"'),
    ): e = replace_checked(e,old,new)
    old = '''    return Endpoint(
        provider="openrouter",
        chat_url=f"{OPENROUTER_BASE}/chat/completions",
        models_url=f"{OPENROUTER_BASE}/models",
        accepts=_OPENROUTER_ACCEPTS,
        chat_timeout=_OPENROUTER_TIMEOUTS["chat"],
        classify_timeout=_OPENROUTER_TIMEOUTS["classify"],
        models_timeout=_OPENROUTER_TIMEOUTS["models"],
        sends_reasoning=True,
        requires_key=True,
        is_openrouter=True,
    )'''
    e = replace_checked(e,old,'    raise ValueError("Hosted prompt providers are disabled in this local-only H3 Portrait build.")')
    p = texts['prompt.py']
    for old,new in (
        ('_DEFAULT_ENDPOINT = _endpoint.resolve("openrouter")', '_DEFAULT_ENDPOINT = _endpoint.resolve("local", "http://127.0.0.1:11434/v1")'),
        ('DEFAULT_MODEL = "google/gemini-3-flash-preview"', 'DEFAULT_MODEL = "(hosted models disabled)"'),
        ('CLASSIFIER_MODEL = "google/gemini-2.5-flash-lite"', 'CLASSIFIER_MODEL = "(hosted models disabled)"'),
    ): p = replace_checked(p,old,new)
    # The reference writer must not inherit a proxy for private media requests.
    marker = '# H3_LOCAL_HTTP_153'
    if marker not in p:
        p = replace_checked(p,'import requests','import requests\n'+marker+'\n_LOCAL_HTTP = requests.Session()\n_LOCAL_HTTP.trust_env = False')
        p = p.replace('requests.post(', '_LOCAL_HTTP.post(').replace('requests.get(', '_LOCAL_HTTP.get(')
    n = texts['nodes.py']
    n = replace_checked(n,'''                "openrouter_model": (prompt.available_models(), {
                    "default": prompt.DEFAULT_MODEL,''','''                "openrouter_model": (["(hosted models disabled)"], {
                    "default": "(hosted models disabled)",''')
    n = replace_checked(n,'''        provider = _provider_of(prompt_provider, use_openrouter)
        model = _model_for(provider, openrouter_model, local_model_slug)''','''        provider = _provider_of(prompt_provider, use_openrouter)
        if provider not in ("local", "none"):
            raise ValueError("Hosted prompt providers are disabled in this local-only H3 Portrait build.")
        model = _model_for(provider, openrouter_model, local_model_slug)''')
    if '# H3_VERIFIED_HANDOFF_153' not in n:
        n += '''\n# H3_VERIFIED_HANDOFF_153
from .local_runtime import guard_refpack
MiniMaxH3ReferencePack.build = guard_refpack(MiniMaxH3ReferencePack.build)
'''
    updated = {'endpoint.py': e, 'nodes.py': n, 'prompt.py': p,
               'local_runtime.py': (Path(__file__).parent/'node/local_runtime.py').read_text()}
    for name, value in updated.items(): compile(value,name,'exec')
    for name, value in updated.items():
        dest = root/'minimax_refpack'/name
        temp = dest.with_suffix('.py.tmp')
        temp.write_text(value); temp.replace(dest)
    manifest={'version':'1.5.3','sha256':{name:hashlib.sha256(value.encode()).hexdigest() for name,value in updated.items()}}
    (root/'H3_LOCAL_PATCH.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify(root)
    print('H3 REFERENCE PACK LOCAL-ONLY PATCH APPLIED | 1.5.3 verified GPU handoff',flush=True)


def verify(root):
    root=Path(root)
    record=json.loads((root/'H3_LOCAL_PATCH.json').read_text())
    if record.get('version')!='1.5.3' or set(record.get('sha256',{}))!={'endpoint.py','nodes.py','prompt.py','local_runtime.py'}:
        raise ValueError('Missing or incompatible Reference Pack safety manifest.')
    for name,digest in record['sha256'].items():
        if hashlib.sha256((root/'minimax_refpack'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Reference Pack source changed after verification: '+name)
    expected=(Path(__file__).parent/'node/local_runtime.py').read_bytes()
    if (root/'minimax_refpack/local_runtime.py').read_bytes()!=expected:
        raise ValueError('Reference Pack and H3 Portrait memory-handoff helpers differ.')
    print('H3 REFERENCE PACK 1.5.3 INTEGRITY VERIFIED',flush=True)


if __name__ == '__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--check': verify(sys.argv[2])
    elif len(sys.argv)==2: patch(sys.argv[1])
    else: raise SystemExit('Usage: patch_refpack_local.py [--check] /path/to/ComfyUI-MiniMaxRefPack')
