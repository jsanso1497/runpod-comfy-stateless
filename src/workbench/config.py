"""Strict configuration resolution. Secrets are never included in public status."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get('WB_ROOT', Path(__file__).resolve().parents[2]))
FAMILIES = ('qwen', 'flux', 'h3', 'restoration')
FEATURES = ('sam', 'seedvr2', 'ollama')
SECRET_KEYS = ('HF_TOKEN', 'CIVITAI_TOKEN', 'hf_token', 'civit_token', 'WB_LORA_URLS', 'WB_CHECKPOINTS', 'WB_PASSWORD')

def read_catalog(name: str):
    return json.loads((ROOT / 'catalog' / name).read_text(encoding='utf-8'))

def csv(value: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in value.split(',') if x.strip()))

def workspace() -> str:
    value = os.environ.get('WB_WORKSPACE', 'qwen')
    if value not in FAMILIES:
        raise ValueError('WB_WORKSPACE must be qwen, flux, h3, or restoration.')
    return value

def model_config(name: str | None = None) -> dict:
    target = name or workspace()
    return next(m for m in read_catalog('models.json')['models'] if m['id'] == target)

def selected_tasks(name: str | None = None) -> list[str]:
    model = model_config(name)
    ids = csv(os.environ.get('WB_TASKS', '')) or list(model['default_tasks'])
    tasks = {t['id']:t for t in read_catalog('tasks.json')['tasks']}
    for id in ids:
        if id not in tasks or model['id'] not in tasks[id]['workspaces']:
            raise ValueError(f'Task {id!r} is not compatible with {model["id"]}. Use the configurator to generate separate workspaces.')
    return ids

def requested_groups(name: str | None = None) -> list[str]:
    model = model_config(name)
    raw = csv(os.environ.get('WB_TOOLBOX', ''))
    if 'standard' in raw:
        raw = [x for x in raw if x != 'standard']
    if set(raw) - set(model['toolbox']):
        raise ValueError('An optional toolbox feature is not supported by this workspace.')
    tasks = {t['id']:t for t in read_catalog('tasks.json')['tasks']}
    groups = set(model['default_groups']) | set(raw)
    for id in selected_tasks(model['id']):
        groups.update(tasks[id]['requires'])
    return sorted(groups)

def data_root() -> Path:
    value = Path(os.environ.get('WB_DATA_ROOT', '/workspace')).expanduser()
    if not value.is_absolute():
        raise ValueError('WB_DATA_ROOT must be an absolute path.')
    return value

def state_root() -> Path:
    return data_root() / 'workbench'

def model_root() -> Path:
    return data_root() / 'models'

def resolve_secret_aliases() -> None:
    # Secret *names* in RunPod need not match variable names inside the container.
    for target, alias in [('HF_TOKEN','hf_token'), ('CIVITAI_TOKEN','civit_token')]:
        if not os.environ.get(target) and os.environ.get(alias):
            os.environ[target] = os.environ[alias]
    for key in SECRET_KEYS:
        if os.environ.get(key, '').lstrip().startswith('{{'):
            raise ValueError(f'{key} contains an unresolved RunPod Secret reference.')

def child_environment() -> dict[str,str]:
    env = dict(os.environ)
    for key in SECRET_KEYS:
        env.pop(key, None)
    env['WB_ROOT'] = str(ROOT)
    env['PYTHONPATH'] = str(ROOT / 'src') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env

def safe_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}', value):
        raise ValueError('Asset names must use letters, digits, periods, hyphens, or underscores.')
    return value

def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative,str) or not relative or '\\' in relative or '\x00' in relative:
        raise ValueError('Invalid relative path.')
    p = Path(relative)
    if p.is_absolute() or any(part in ('..','') for part in p.parts):
        raise ValueError('Path must stay within its storage root.')
    root = root.resolve()
    target = root / p
    if target.is_symlink() or not target.resolve().is_relative_to(root) or target.resolve() == root:
        raise ValueError('Path escapes storage or is a symlink.')
    return target

def atomic_json(path: Path, value, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Refusing a symlinked state file.')
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temp=Path(handle.name)
        json.dump(value,handle,indent=2);handle.write('\n')
    temp.chmod(0o600 if private else 0o644)
    temp.replace(path)
