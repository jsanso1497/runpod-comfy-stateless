"""Strict configuration resolution. Secrets are never included in public status."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

ROOT = Path(os.environ.get('WB_ROOT', Path(__file__).resolve().parents[2]))
FAMILIES = ('qwen', 'flux', 'h3', 'restoration', 'krea')
FEATURES = ('sam', 'seedvr2', 'ollama')
SECRET_KEYS = ('HF_TOKEN', 'CIVITAI_TOKEN', 'hf_token', 'civit_token', 'WB_LORA_URLS', 'WB_CHECKPOINTS', 'WB_PASSWORD')

def read_catalog(name: str):
    value = json.loads((ROOT / 'catalog' / name).read_text(encoding='utf-8'))
    if name == 'assets.json':
        value.setdefault('ollama_by_workspace', {})['h3'] = dict(value['ollama'])
    addon = ROOT / 'catalog' / 'tasks.green-suit.json'
    if name == 'tasks.json' and addon.is_file():
        ids = {task['id'] for task in value['tasks']}
        for task in json.loads(addon.read_text(encoding='utf-8'))['tasks']:
            if task['id'] in ids:
                raise ValueError('Duplicate add-on task ID: ' + task['id'])
            if task['id'] != 'QGS1' or task['workspaces'] != ['qwen']:
                raise ValueError('Unexpected green-suit add-on task or workspace.')
            if set(task['requires']) - {'qwen', 'sam', 'seedvr2'}:
                raise ValueError('Unsupported green-suit task dependency.')
            for variant in task['variants']:
                safe_path(ROOT, variant['file'])
            value['tasks'].append(task)
            ids.add(task['id'])
    addon = ROOT / 'catalog' / 'krea.json'
    if addon.is_file() and name in ('models.json', 'tasks.json', 'assets.json'):
        krea = json.loads(addon.read_text(encoding='utf-8'))
        if krea['model']['id'] != 'krea' or krea['task']['workspaces'] != ['krea']:
            raise ValueError('Invalid Krea add-on workspace.')
        if name == 'models.json':
            if any(m['id'] == 'krea' for m in value['models']):
                raise ValueError('Duplicate Krea workspace.')
            value['models'].append(krea['model'])
            for key, row in krea['external_nodes'].items():
                if key in value['external_nodes']:
                    raise ValueError('Duplicate external-node identifier: ' + key)
                value['external_nodes'][key] = row
        elif name == 'tasks.json':
            if any(t['id'] == krea['task']['id'] for t in value['tasks']):
                raise ValueError('Duplicate Krea task.')
            for variant in krea['task']['variants']:
                safe_path(ROOT, variant['file'])
            value['tasks'].append(krea['task'])
        else:
            ids = {a['id'] for a in value['assets']}
            destinations = {a['destination'] for a in value['assets']}
            for asset in krea['assets']:
                if asset['id'] in ids or asset['destination'] in destinations:
                    raise ValueError('Duplicate Krea asset.')
                if asset['family'] != 'krea' or asset['group'] != 'krea':
                    raise ValueError('Invalid Krea asset family.')
                value['assets'].append(asset)
                ids.add(asset['id']); destinations.add(asset['destination'])
            value.setdefault('ollama_by_workspace', {})['krea'] = krea['ollama']
            if os.environ.get('WB_WORKSPACE') == 'krea':
                value['ollama'] = krea['ollama']
    if name in ('models.json', 'tasks.json', 'assets.json'):
        from .qwen_prompt_catalog import apply
        value = apply(name, value)
    if name == 'tasks.json':
        from .prompt_studio_catalog import append_tasks
        append_tasks(value, ROOT)
    from .identity_catalog import apply as apply_identity
    value = apply_identity(name, value, ROOT)
    from .qwen_people_catalog import apply as apply_people
    value = apply_people(name, value, ROOT)
    from .moss_catalog import apply as apply_moss
    return apply_moss(name, value, ROOT)

def csv(value: str) -> list[str]:
    return list(dict.fromkeys(x.strip() for x in value.split(',') if x.strip()))

def workspace() -> str:
    value = os.environ.get('WB_WORKSPACE', 'qwen')
    if value not in FAMILIES:
        raise ValueError('WB_WORKSPACE must be one of: ' + ', '.join(FAMILIES))
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
