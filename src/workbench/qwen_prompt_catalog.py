"""Additive vision helper migration without replacing public/private catalogs."""
from __future__ import annotations
import os
from .qwen_prompt_local import helper_config


def apply(name: str, value: dict) -> dict:
    helper = helper_config()
    if name == 'models.json':
        for model in value['models']:
            if model['id'] != 'qwen':
                continue
            model['default_groups'] = [g for g in model.get('default_groups', []) if g != 'qwen_pe_i2i']
            if 'ollama' not in model['default_groups']:
                model['default_groups'].append('ollama')
            if 'ollama' not in model.setdefault('toolbox', []):
                model['toolbox'].append('ollama')
            if 'qwen_pe_i2i' not in model['toolbox']:
                model['toolbox'].append('qwen_pe_i2i')
            model['description'] = ('Reference editing, identity, wardrobe and local repair. The existing prompt enhancer offers Qwen3.8 27B BF16 vision (default) or the official Qwen Image 2.1 PE-I2I BF16. OFF defaults are preserved.')
            model['storage_gb'] = max(model.get('storage_gb', 0), 250)
            model['hardware_note'] = ('Planning target, not a measured minimum. Local Qwen3.8 adds about 56 GB; enabling qwen_pe_i2i adds about 19 GB. The active helper unloads before Qwen image generation. Both paths require a live GPU test.')
    elif name == 'tasks.json':
        for task in value['tasks']:
            if 'qwen' not in task.get('workspaces', []):
                continue
            if 'optional_prompt_enhancement' in task:
                task['optional_prompt_enhancement'] = {'model': helper['models'][0], 'precision': helper['precision'], 'default': 'off', 'backend': 'ollama'}
            if task['id'] == 'U10':
                task['requires'] = list(dict.fromkeys([g for g in task['requires'] if g != 'qwen_pe_i2i'] + ['ollama']))
                task['title'] = 'Compare Qwen3.8 vision and official I2I enhanced prompts'
                task['description'] = 'Use the selector on the existing enhancer control to preview either Qwen3.8 27B BF16 vision or official Qwen Image 2.1 PE-I2I BF16. Prepare qwen_pe_i2i from Toolbox before using official mode.'
    elif name == 'assets.json':
        value.setdefault('ollama_by_workspace', {})['qwen'] = helper
    from .krea_catalog import apply as apply_krea
    value = apply_krea(name, value)
    if name == 'assets.json':
        selected = os.environ.get('WB_WORKSPACE', 'qwen')
        if selected in value['ollama_by_workspace']:
            value['ollama'] = value['ollama_by_workspace'][selected]
    return value
