"""Developer recipes are additive tasks, not replacements for saved Workbench graphs."""
from __future__ import annotations
import copy

VERSION = '1.1.1'
KREA_REV = '89e9e7a09ee2e5c9331e952063d79b1b8a703280'
QREPO = 'ausboss/Qwen-Image-2.1-Consistency-LoRA'
OREPO = 'ausboss/Qwen-Image-2.1-Outfit-Swap-Consistency-LoRA'
QREV = '8f05b0fa027d517fa396fb31b71e0eaf48110e89'
OREV = '226c0a4edcee08328f2369f42f14b305e9671d9c'
FILES = {
    'KDEV': [
        ('Turbo: one image', '01_Krea_Turbo_Single_Image_BF16'),
        ('Turbo: scene + person', '02_Krea_Turbo_Scene_Person_BF16'),
        ('Raw: one image', '03_Krea_Raw_Single_Image_BF16'),
        ('Raw: scene + person', '04_Krea_Raw_Scene_Person_BF16'),
    ],
    'QDEV': [
        ('Keep framing: Consistency 1500', '01_Qwen_Consistency_1500_BF16'),
        ('Tighter framing: Consistency 2000', '02_Qwen_Consistency_2000_BF16'),
        ('Edit then realign: no LoRA', '03_Qwen_Realign_No_LoRA_BF16'),
        ('Outfit swap', '04_Qwen_Outfit_Swap_BF16'),
    ],
}


def workflow_path(task, name):
    family = {'KDEV': 'krea', 'QDEV': 'qwen'}[task]
    return f'workflows/developer/{family}/{task}/{name}.json'


def task_rows():
    result = []
    for task, family in [('KDEV', 'krea'), ('QDEV', 'qwen')]:
        sources = (['https://github.com/lbouaraba/comfyui-krea2edit',
                    'https://huggingface.co/conradlocke/krea2-identity-edit'] if task == 'KDEV' else
                   ['https://huggingface.co/' + QREPO, 'https://huggingface.co/' + OREPO])
        result.append({
            'id': task, 'family': family, 'workspaces': [family], 'category': 'tasks',
            'title': ('Developer Identity Edit: Raw/Turbo, one image or scene + person' if task == 'KDEV' else
                      'Developer consistency, realignment and outfit-swap recipes'),
            'description': ('BF16-adapted developer baselines. No prompt-enhancer dependency. '
                            'Krea includes sRGB image I/O and two bypassed optional LoRA slots; select compatible '
                            'Civitai checkpoints in the native diffusion loader. These are not GPU-certified.'
                            if task == 'KDEV' else
                            'BF16-adapted AusBoss recipes. 1500 and 2000 are alternatives, not a stack. '
                            'No prompt enhancer added. Consistency adapters are not dedicated face-swap models.'),
            'sources': sources, 'requires': ['krea'] if task == 'KDEV' else ['qwen', 'qwen_identity'],
            'status': 'developer-derived source candidate; GPU validation pending',
            'variants': [{'label': label, 'file': workflow_path(task, name),
                          'source': 'Developer JSON adapted for Workbench BF16; original archived in image'}
                         for label, name in FILES[task]],
            'required_nodes': (['Krea2EditModelPatch', 'Krea2EditGroundedEncode'] if task == 'KDEV' else []),
        })
    return result


def asset_rows():
    result = []
    for ident, repo, rev, filename, sha in [
        ('qwen-consistency-1500', QREPO, QREV, 'qwen-image-2.1-consistency.safetensors',
         '4f44ada1be2189cc23b3d010f9603543403f48454e2f76842a3d30109b20bd63'),
        ('qwen-consistency-2000', QREPO, QREV, 'qwen-image-2.1-consistency-2000.safetensors',
         'a0bf043edc4695b1661a0e768e49a7fdff9a656b7d4313203d590e0262b64beb'),
        ('qwen-outfit-swap', OREPO, OREV, 'qwen-image-2.1-outfit-swap.safetensors',
         '4ddfeac5695adaed1f0f78b23fda40919fd2ed6a4af790ced6c1e043dc4ae1f0'),
    ]:
        result.append({'id': ident, 'family': 'qwen', 'group': 'qwen_identity', 'repo_id': repo,
                       'revision': rev, 'filename': filename, 'destination': 'loras/' + filename,
                       'sha256': sha, 'estimated_bytes': 160_000_000,
                       'integrity_source': 'Pinned developer file checksum; enforced on the Pod',
                       'verification': 'GPU inference not certified'})
    return result


def apply(name, value, root=None):
    """Append after existing Qwen/Krea catalog extensions. Do not alter defaults."""
    if name == 'tasks.json':
        known = {t['id'] for t in value['tasks']}
        for task in task_rows():
            if task['id'] in known:
                raise ValueError('Duplicate identity developer task: ' + task['id'])
            value['tasks'].append(task)
    elif name == 'assets.json':
        ids = {a['id'] for a in value['assets']}
        destinations = {a['destination'] for a in value['assets']}
        for row in asset_rows():
            if row['id'] in ids or row['destination'] in destinations:
                raise ValueError('Duplicate identity asset destination.')
            value['assets'].append(row)
        for row in value['assets']:
            if row['id'] == 'krea-krea2_identity_edit_v1_2':
                row['revision'] = KREA_REV
    from .qwen_stock_catalog import apply as apply_stock
    return apply_stock(name, value)
