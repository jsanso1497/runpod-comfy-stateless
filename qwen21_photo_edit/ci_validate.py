#!/usr/bin/env python3
"""Dependency-free, no-model workflow/overlay CI validation.

This is separate from the CPU unit tests, which run during the Docker image build
with pinned PyTorch and ComfyUI installed. Never downloads model weights in CI.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from make_workflows import make


def verify_graph(filename: str, precision: str):
    workflow = json.loads((BASE / 'workflows' / (filename + '.json')).read_text())
    api = json.loads((BASE / 'workflows' / (filename + '.api.json')).read_text())
    expected_workflow, expected_api = make(precision)
    if workflow != expected_workflow or api != expected_api:
        raise AssertionError(f'{filename}: committed workflows are not in sync with make_workflows.py')
    nodes = {n['id']: n for n in workflow['nodes']}
    if len(nodes) != len(workflow['nodes']):
        raise AssertionError('Repeated node IDs')
    output_nodes = [n for n in nodes.values() if n['type'] == 'Q21PhotoReviewSave']
    if len(output_nodes) != 1 or output_nodes[0]['widgets_values_named']['run_edit'] is not False:
        raise AssertionError('Start in mask-preview mode, not paid GPU generation.')
    loader = next(n for n in nodes.values() if n['type'] == 'Q21PhotoModels')
    if loader['widgets_values_named']['precision'] != precision:
        raise AssertionError('Incorrect model precision defaults')
    if any(n['type'] in {'BFSHeadSwap', 'BFSBodySwap'} for n in nodes.values()):
        raise AssertionError('No BFS adapter should load by default')
    loras = next(n for n in nodes.values() if n['type'] == 'Q21PhotoLoRAs')
    if any(loras['widgets_values_named'][f'lora_{n}'] != 'None' for n in (1, 2, 3)):
        raise AssertionError('LoRA must be opt-in')
    mask = next(n for n in nodes.values() if n['type'] == 'Q21PhotoMask')
    if mask['widgets_values_named']['auto_mask'] is not False:
        raise AssertionError('SAM must be opt-in to save VRAM and avoid accidental masks')
    crop = next(n for n in nodes.values() if n['type'] == 'Q21PhotoCrop')
    if crop['widgets_values_named']['resolution'] != 2048:
        raise AssertionError('Expected approximately 2K crop budget')
    links = {row[0]: row for row in workflow['links']}
    if len(links) != len(workflow['links']):
        raise AssertionError('Repeated link IDs')
    for row in workflow['links']:
        lid, source, outslot, target, inslot, typ = row
        assert source in nodes and target in nodes, (lid, 'unknown node')
        assert nodes[source]['outputs'][outslot]['type'] == typ, (lid, 'output type mismatch')
        assert nodes[target]['inputs'][inslot]['type'] == typ, (lid, 'input type mismatch')
        assert nodes[target]['inputs'][inslot]['link'] == lid, (lid, 'wrong input link')
        assert lid in nodes[source]['outputs'][outslot]['links'], (lid, 'wrong output link')
    for node in nodes.values():
        for inp in node['inputs']:
            if inp['link'] is not None and inp['link'] not in links:
                raise AssertionError('Dangling input link')
    if set(api) != {str(n) for n in nodes if nodes[n]['type'] != 'Note'}:
        raise AssertionError('API form missing a node or including notes')
    for ident, entry in api.items():
        for key, val in entry['inputs'].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and val[0].isdigit():
                if int(val[0]) not in nodes:
                    raise AssertionError(f'API has dangling node reference: {ident}.{key}')
    print(f'PASS {filename}: UI + API graphs, mask, crop, references, LoRA, sampler, and output')


def main():
    for filename, precision in [('Qwen21_Photo_2K_SAM3_BF16', 'BF16 quality'),
                                ('Qwen21_Photo_2K_SAM3_INT8', 'INT8 lower memory')]:
        verify_graph(filename, precision)
    for name in ['node/__init__.py', 'node/geometry.py', 'download_models.py', 'install.sh',
                 'preflight.py', 'start.sh', 'pin_torch.py']:
        if not (BASE / name).is_file():
            raise AssertionError('Missing required file: '+name)
        if name.endswith('.py'):
            ast.parse((BASE / name).read_text())
    print('PASS source package is structurally complete (no runtime models tested).')


if __name__ == '__main__':
    main()
