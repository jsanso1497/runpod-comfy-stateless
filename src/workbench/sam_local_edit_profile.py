"""Opt-in minimal Qwen/SAM catalog. Other images retain their existing behavior."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

VERSION = 'approved-local-edit-1'
TASK_ID = 'QLOCAL'
WORKFLOW_PATH = 'workflows/qwen/00_Approved/00_SAM_Guided_Local_Edit.json'
COMFY_COMMIT = 'bab80ef793ac42ad3f25589a6b0507ca537b1dd1'
ALLOWED_GROUPS = frozenset({'qwen', 'sam'})
LOCAL_NODES = ['qwen_photo', 'runtime_compat', 'sam_local_edit_ui']
EXPECTED_ASSETS = {
    'diffusion_models/qwen_image_2.1_bf16.safetensors',
    'text_encoders/qwen3vl_8b_bf16.safetensors',
    'vae/qwen_image_2.1_vae_bf16.safetensors',
    'checkpoints/sam3.1_multiplex_fp16.safetensors',
}


def enabled() -> bool:
    value = os.environ.get('WB_APPROVED_ONLY', '0')
    if value not in ('0', '1'):
        raise ValueError('WB_APPROVED_ONLY must be 0 or 1.')
    return value == '1' and os.environ.get('WB_WORKSPACE', 'qwen') == 'qwen'


def _json(root: Path, name: str):
    return json.loads((root / 'catalog' / name).read_text(encoding='utf-8'))


def approved_tasks(root: Path) -> list[dict]:
    catalog = _json(Path(root), 'approved.workflows.json')
    rows = catalog.get('tasks', [])
    if catalog.get('schema_version') != 1 or len(rows) != 1:
        raise ValueError('Expected exactly one approved workflow.')
    task = rows[0]
    if (task.get('id') != TASK_ID or task.get('approved') is not True
            or task.get('workspaces') != ['qwen']
            or set(task.get('requires', [])) != ALLOWED_GROUPS
            or len(task.get('variants', [])) != 1
            or task['variants'][0].get('file') != WORKFLOW_PATH):
        raise ValueError('Approved workflow or dependencies changed; review the profile.')
    return copy.deepcopy(rows)


def catalog_override(name: str, root: Path):
    """Run BEFORE the general add-on pipeline, so it cannot add extra downloads."""
    if not enabled() or name not in ('tasks.json', 'models.json', 'assets.json'):
        return None
    root = Path(root)
    if name == 'tasks.json':
        return {'schema_version': 1, 'tasks': approved_tasks(root)}
    value = _json(root, name)
    if name == 'models.json':
        matches = [m for m in value['models'] if m['id'] == 'qwen']
        if len(matches) != 1:
            raise ValueError('Exactly one base Qwen model configuration is required.')
        model = matches[0]
        model.update(
            name='SAM-Guided Local Edit', short='Local Edit',
            description='Only the user-approved SAM-guided local-edit workflow.',
            comfy_commit=COMFY_COMMIT, external_nodes=[], local_nodes=list(LOCAL_NODES),
            default_tasks=[TASK_ID], default_groups=['qwen', 'sam'], toolbox=[],
            storage_gb=100, ram_gb=64, vram_gb=48,
            hardware_note='Planning target: one 48 GB GPU and 64 GB system RAM. '
                          'Not a certified memory minimum; keep approved processing defaults.',
            approved_only=True,
        )
        model.pop('experimental_moss', None)
        value['models'] = [model]
        value['external_nodes'] = {}
        return value
    assets = [a for a in value['assets'] if a.get('group') in ALLOWED_GROUPS]
    if len(assets) != 4 or {a['destination'] for a in assets} != EXPECTED_ASSETS:
        raise ValueError('The approved profile requires exactly the inspected Qwen/SAM asset set.')
    value['assets'] = assets
    empty = {'group': 'ollama', 'models': [], 'estimated_bytes': 0,
             'note': 'No prompt-enhancer model or server in this image.'}
    value['ollama'] = dict(empty)
    value['ollama_by_workspace'] = {'qwen': dict(empty)}
    return value


def sync_approved(source_root, data_root, tasks, workspace):
    """Put the approved graph first alphabetically, preserving saved user edits."""
    if workspace != 'qwen':
        raise ValueError('The approved local-edit profile is a Qwen workspace.')
    source_root, data_root = Path(source_root).resolve(), Path(data_root).resolve()
    rows = [t for t in tasks if workspace in t.get('workspaces', [])]
    if len(rows) != 1 or rows[0].get('id') != TASK_ID:
        raise ValueError('Unapproved tasks reached the approved factory sync.')
    source = source_root / WORKFLOW_PATH
    if source.is_symlink() or not source.resolve().is_relative_to(source_root):
        raise ValueError('Approved workflow source escapes its root.')
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    folder = data_root / 'comfy-user/default/workflows/00_Approved'
    dest = folder / '00_SAM_Guided_Local_Edit.json'
    if not dest.resolve().is_relative_to(data_root) or dest.is_symlink():
        raise ValueError('Approved workflow destination escapes the data volume.')
    # Never remove old workflows or silently replace a user-edited copy.
    if dest.exists() and dest.read_bytes() != raw:
        dest = folder / ('00_SAM_Guided_Local_Edit__factory_' + digest[:12] + '.json')
    if not dest.resolve().is_relative_to(data_root) or dest.is_symlink():
        raise ValueError('Approved alternate destination escapes the data volume.')
    if dest.exists() and dest.read_bytes() != raw:
        raise ValueError('Both approved factory copies were edited; preserve them and rename one.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        fd, temporary = tempfile.mkstemp(dir=dest.parent, prefix='.approved-')
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(raw)
            os.chmod(temporary, 0o644)
            os.replace(temporary, dest)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {'directory': str(folder.relative_to(data_root)), 'profile': VERSION,
            'workflows': [{'task': TASK_ID, 'path': str(dest.relative_to(data_root)),
                           'source_sha256': digest}]}
