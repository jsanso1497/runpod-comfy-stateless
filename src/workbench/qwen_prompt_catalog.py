"""Additive Qwen helper migration without replacing public/private model catalogs."""
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
            model['description'] = ('Reference editing, identity, wardrobe and local repair. Existing prompt toggles now use ' + helper['label'] + ' locally; OFF defaults are preserved.')
            model['storage_gb'] = max(model.get('storage_gb', 0), 250)
            model['hardware_note'] = ('Planning target, not a measured minimum. ' + helper['label'] + ' adds approximately ' + str(helper['estimated_bytes'] // 1_000_000_000) + ' GB of model storage. The prompt model unloads before Qwen image generation. Full-precision helper residency and image quality need a live GPU test.')
    elif name == 'tasks.json':
        for task in value['tasks']:
            if 'qwen' not in task.get('workspaces', []):
                continue
            if 'optional_prompt_enhancement' in task:
                task['optional_prompt_enhancement'] = {'model': helper['models'][0], 'precision': helper['precision'], 'default': 'off', 'backend': 'ollama'}
            if task['id'] == 'U10':
                task['requires'] = list(dict.fromkeys([g for g in task['requires'] if g != 'qwen_pe_i2i'] + ['ollama']))
                task['title'] = 'Preview local vision-enhanced Qwen prompt'
                task['description'] = 'Preview the selected full-precision abliterated vision rewrite, original prompt, and validation status. No image generation. Invalid rewrites retry once before visible original-prompt fallback.'
    elif name == 'assets.json':
        value.setdefault('ollama_by_workspace', {})['qwen'] = helper
        selected = os.environ.get('WB_WORKSPACE', 'qwen')
        if selected in value['ollama_by_workspace']:
            value['ollama'] = value['ollama_by_workspace'][selected]
    return value
