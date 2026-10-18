"""Register QKEEP1 only when its explicit add-on catalog is present."""
from __future__ import annotations
import copy
import json
from pathlib import Path

TASK = 'QKEEP1'
WORKFLOW = 'workflows/qwen/QKEEP1/Qwen21_Keep_One_Person_Protected_Overlay_BF16.json'


def apply(name, value, root):
    root = Path(root)
    path = root / 'catalog/tasks.qwen-people.json'
    if name not in ('models.json', 'tasks.json') or not path.is_file():
        return value
    addon = json.loads(path.read_text(encoding='utf-8'))
    rows = addon.get('tasks', [])
    if addon.get('schema_version') != 1 or len(rows) != 1:
        raise ValueError('Invalid QKEEP1 add-on catalog.')
    row = rows[0]
    if (row.get('id') != TASK or row.get('workspaces') != ['qwen'] or
            row.get('requires') != ['qwen', 'sam', 'qwen_stock_pe'] or
            len(row.get('variants', [])) != 1 or row['variants'][0].get('file') != WORKFLOW):
        raise ValueError('Unexpected QKEEP1 task or dependencies.')
    if name == 'models.json':
        for model in value['models']:
            if model.get('id') == 'qwen':
                if 'qwen_people' not in model.setdefault('local_nodes', []):
                    model['local_nodes'].append('qwen_people')
    else:
        if any(t.get('id') == TASK for t in value['tasks']):
            raise ValueError('Duplicate QKEEP1 task.')
        value['tasks'].append(copy.deepcopy(row))
    return value
