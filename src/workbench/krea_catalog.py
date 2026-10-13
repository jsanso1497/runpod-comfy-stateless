"""Add Krea parity without changing Qwen assets or private library registrations."""
from __future__ import annotations
from .qwen_prompt_local import helper_config

MASKED_FILE = 'workflows/krea/K01/Krea_Masked_Photo_HQ.json'


def apply(name: str, value: dict) -> dict:
    helper = helper_config()
    if name == 'models.json':
        for model in value['models']:
            if model['id'] != 'krea':
                continue
            for node in ('qwen_prompt',):
                if node not in model.setdefault('local_nodes', []):
                    model['local_nodes'].append(node)
            # Restoration remains a separate optional tool, never a generation pass.
            for node in ('seedvr2',):
                if node not in model.setdefault('external_nodes', []):
                    model['external_nodes'].append(node)
            for feature in ('sam', 'seedvr2', 'qwen_pe_i2i'):
                if feature not in model.setdefault('toolbox', []):
                    model['toolbox'].append(feature)
            model['description'] = ('Identity Edit, text-only Standard, and optional protected masked editing. '
                'Uses the same selected full-precision vision helper as Qwen, with reference roles, '
                'rewrite validation, and explicit model unloading.')
            model['hardware_note'] = ('Planning target, not a measured minimum. The selected full-precision '
                'vision helper runs before Krea and must unload. Both Raw and Turbo BF16 remain available. '
                'Generation stays at or below 2 MP. SAM, SeedVR2 and official PE-I2I assets are optional. '
                'Extra references use a sheet in the upstream two-image editor. GPU validation pending.')
    elif name == 'tasks.json':
        for task in value['tasks']:
            if task['id'] == 'K01':
                task['optional_prompt_enhancement'] = {
                    'model': helper['models'][0], 'precision': helper['precision'],
                    'default': 'on', 'backend': 'ollama'}
                task['description'] = ('One source and 0-4 role-labeled references, Raw/Turbo BF16, '
                    'two optional LoRAs and the shared full-precision Qwen vision helper. '
                    'Choose full-frame Identity Edit, text-only Standard, or masked crop/stitch '
                    'with manual/SAM selection and a preview-first safety gate.')
                if not any(v['file'] == MASKED_FILE for v in task['variants']):
                    task['variants'].append({'label': 'Krea Masked Photo - manual/SAM, preview, protected stitch',
                        'file': MASKED_FILE, 'source': 'Workbench Krea integration with shared photo geometry'})
    elif name == 'assets.json':
        value.setdefault('ollama_by_workspace', {})['krea'] = helper
    return value
