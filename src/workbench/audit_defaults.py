"""Additive defaults for the already-installed Qwen + MOSS workspace."""
import os


def configure_qwen(model):
    if model.get('id') != 'qwen':
        return
    policy = os.getenv('WB_MOSS_AUTO_PREPARE', '1')
    if policy not in ('0', '1'):
        raise ValueError('WB_MOSS_AUTO_PREPARE must be 0 or 1.')
    if policy == '1':
        groups = model.setdefault('default_groups', [])
        if 'moss_image' not in groups:
            groups.append('moss_image')
    local = model.setdefault('local_nodes', [])
    if 'runtime_compat' not in local:
        local.append('runtime_compat')
