#!/usr/bin/env python3
"""Patch pinned Hearmeman Reference Pack 0.3.5 into local-only mode."""
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
endpoint = root / 'minimax_refpack/endpoint.py'
nodes = root / 'minimax_refpack/nodes.py'
prompt_file = root / 'minimax_refpack/prompt.py'
if not endpoint.is_file() or not nodes.is_file() or not prompt_file.is_file():
    raise SystemExit('Reference Pack source layout changed.')

e = endpoint.read_text()
for old, new in {
    'PROVIDERS = ("openrouter", "local", "none")': 'PROVIDERS = ("local", "none")',
    'DEFAULT_PROVIDER = "openrouter"': 'DEFAULT_PROVIDER = "local"',
    'if raw is True:\n        return "openrouter"': 'if raw is True:\n        return "local"',
    'if text == "true":\n        return "openrouter"': 'if text == "true":\n        return "local"',
}.items():
    if old not in e:
        raise SystemExit('Reference Pack endpoint source drift: ' + old)
    e = e.replace(old, new, 1)

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
new = '''    raise ValueError("Hosted prompt providers are disabled in this local-only H3 Portrait build.")'''
if old not in e:
    raise SystemExit('Reference Pack OpenRouter endpoint block changed.')
e = e.replace(old, new, 1)
endpoint.write_text(e)

p = prompt_file.read_text()
for old, new in {
    '_DEFAULT_ENDPOINT = _endpoint.resolve("openrouter")': '_DEFAULT_ENDPOINT = _endpoint.resolve("local", "http://127.0.0.1:11434/v1")',
    'DEFAULT_MODEL = "google/gemini-3-flash-preview"': 'DEFAULT_MODEL = "(hosted models disabled)"',
    'CLASSIFIER_MODEL = "google/gemini-2.5-flash-lite"': 'CLASSIFIER_MODEL = "(hosted models disabled)"',
}.items():
    if old not in p:
        raise SystemExit('Reference Pack prompt source drift: ' + old)
    p = p.replace(old, new, 1)
prompt_file.write_text(p)

n = nodes.read_text()
old = '''                "openrouter_model": (prompt.available_models(), {
                    "default": prompt.DEFAULT_MODEL,'''
new = '''                "openrouter_model": (["(hosted models disabled)"], {
                    "default": "(hosted models disabled)",'''
if old not in n:
    raise SystemExit('Reference Pack OpenRouter model widget changed.')
n = n.replace(old, new, 1)

old = '''        provider = _provider_of(prompt_provider, use_openrouter)
        model = _model_for(provider, openrouter_model, local_model_slug)'''
new = '''        provider = _provider_of(prompt_provider, use_openrouter)
        if provider not in ("local", "none"):
            raise ValueError("Hosted prompt providers are disabled in this local-only H3 Portrait build.")
        model = _model_for(provider, openrouter_model, local_model_slug)'''
if old not in n:
    raise SystemExit('Reference Pack provider build block changed.')
n = n.replace(old, new, 1)
nodes.write_text(n)
print('H3 REFERENCE PACK LOCAL-ONLY PATCH APPLIED', flush=True)
