"""Two experimental MOSS tasks inside Qwen; not Qwen checkpoints.

No MOSS registry entries are given family='qwen', so the Workbench Qwen
checkpoint/LoRA dropdowns and private-library architecture rules stay intact.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path

TASK = 'QMOSS'
I2I_TASK = 'QMOSS_I2I'
I2I_WORKFLOW = 'workflows/experimental/moss/QMOSS_I2I/MOSS_Image_I2I.json'
TASKS = (TASK, I2I_TASK)
GROUP = 'moss_image'
WORKFLOW = 'workflows/experimental/moss/QMOSS/MOSS_Image_Stock.json'
COMFY_COMMIT = 'bab80ef793ac42ad3f25589a6b0507ca537b1dd1'
PREVIOUS_COMFY_COMMIT = '52f98af2e2e42c421070a3e147c161c47cdeaf22'
MODEL_REPO = 'Comfy-Org/MOSS-Image'
MODEL_REVISION = 'f28da8f93449c16cfa7efa07dcd1303a1c0c44d1'

# Exact ComfyUI template selections, not substitutions with Qwen Image weights.
ASSET_SPECS = (
    ('moss-stock-sft-int8', 'diffusion_models/moss_image_sft_int8_convrot.safetensors',
     'b4d1475bc7eeffd7b809e82ea13a1fba1fb85c4e18365af3813722faa5c0fe4f', 5413698958, True),
    ('moss-stock-qwen35-w4a8', 'text_encoders/qwen3.5_9b_w4a8_gptq.safetensors',
     '4cc6a1543ad2061806da46f35f90cadb4801dcba246c44782cf42ec0feb6407c', 7343591088, True),
    ('moss-stock-vae-small', 'vae/flux2-vae_small.safetensors',
     'd8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b', 249521340, False),
)
MODEL_FILES = frozenset(row[1].split('/', 1)[1] for row in ASSET_SPECS)


def asset_rows() -> list[dict]:
    return [dict(id=ident, family='moss', group=GROUP, repo_id=MODEL_REPO,
                 revision=MODEL_REVISION, filename=filename, destination=filename,
                 sha256=sha, estimated_bytes=size,
                 integrity_source='Official Comfy-Org LFS SHA256 and immutable source revision',
                 verification='Provider pointer verified; full weights and GPU inference not tested here')
            for ident, filename, sha, size, _ in ASSET_SPECS]


def is_stock_quantized_asset(asset: dict) -> bool:
    """An exception only for the two hash-pinned official stock MOSS files.

    Private entries, alternate repositories, renamed destinations, changed
    revisions and new hashes cannot use this exception. The VAE remains strict.
    """
    keys = ('id', 'family', 'group', 'repo_id', 'revision', 'filename', 'destination', 'sha256')
    for expected, spec in zip(asset_rows(), ASSET_SPECS):
        if spec[4] and all(asset.get(key) == expected[key] for key in keys):
            return True
    return False


def is_stock_model_selection(task_id: str, value) -> bool:
    return task_id in TASKS and isinstance(value, str) and value in MODEL_FILES


def apply(name: str, value: dict, root: Path) -> dict:
    """Run last in read_catalog; honor the explicit installed addon only."""
    addon_path = Path(root) / 'catalog/tasks.moss.json'
    if name not in ('models.json', 'tasks.json', 'assets.json') or not addon_path.is_file():
        return value
    addon = json.loads(addon_path.read_text(encoding='utf-8'))
    rows = addon.get('tasks', [])
    if addon.get('schema_version') != 1 or [r.get('id') for r in rows] != list(TASKS):
        raise ValueError('Invalid MOSS addon catalog.')
    for row, expected_id, expected_path in zip(rows, TASKS, (WORKFLOW, I2I_WORKFLOW)):
        if (row.get('id') != expected_id or row.get('family') != 'moss' or
            row.get('workspaces') != ['qwen'] or row.get('requires') != [GROUP] or
            len(row.get('variants', [])) != 1 or row['variants'][0].get('file') != expected_path):
            raise ValueError('Unexpected MOSS task or dependencies.')
    if name == 'tasks.json':
        if any(t.get('id') in TASKS for t in value['tasks']):
            raise ValueError('Duplicate MOSS task.')
        value['tasks'].extend(copy.deepcopy(rows))
    elif name == 'assets.json':
        ids = {a['id'] for a in value['assets']}
        destinations = {a['destination'] for a in value['assets']}
        for asset in asset_rows():
            if asset['id'] in ids or asset['destination'] in destinations:
                raise ValueError('MOSS asset destination or ID is already registered.')
            value['assets'].append(asset)
            ids.add(asset['id']); destinations.add(asset['destination'])
    else:
        for model in value['models']:
            if model.get('id') != 'qwen':
                continue
            current = model.get('comfy_commit')
            if current not in (PREVIOUS_COMFY_COMMIT, COMFY_COMMIT):
                raise ValueError('Review the Qwen ComfyUI pin before applying the MOSS addon.')
            model['comfy_commit'] = COMFY_COMMIT
            if GROUP not in model.setdefault('toolbox', []):
                model['toolbox'].append(GROUP)
            model['experimental_moss'] = {
                'task': TASK, 'tasks': list(TASKS), 'comfy_commit': COMFY_COMMIT,
                'note': 'Stock MOSS model precision is independent of Qwen BF16. GPU validation pending.'}
    return value
