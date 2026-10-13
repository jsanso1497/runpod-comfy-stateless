"""Read the explicit prompt-only task; expose its shared helper in Qwen and Krea."""
from __future__ import annotations
import json
from pathlib import Path


def append_tasks(value: dict, root: Path) -> None:
    path = root / 'catalog' / 'tasks.prompt-studio.json'
    if not path.is_file():
        return
    from .config import safe_path
    addon = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(addon, dict) or addon.get('schema_version') != 1:
        raise ValueError('Invalid prompt studio task catalog.')
    tasks = addon.get('tasks')
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError('The prompt studio catalog must contain only U11.')
    task = tasks[0]
    if task.get('id') != 'U11' or task.get('workspaces') != ['qwen'] or task.get('requires') != ['ollama']:
        raise ValueError('Unsupported prompt studio task or dependencies.')
    if any(t['id'] == 'U11' for t in value['tasks']):
        raise ValueError('Duplicate prompt studio task U11.')
    for variant in task['variants']:
        safe_path(root, variant['file'])
    task['workspaces'] = ['qwen', 'krea']
    value['tasks'].append(task)
