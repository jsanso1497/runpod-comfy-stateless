#!/usr/bin/env python3
"""Offline approved-graph and minimal-profile contracts; no GPU certification."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from workbench.sam_local_edit_profile import WORKFLOW_PATH, catalog_override, LOCAL_NODES


def normalized(graph):
    graph = copy.deepcopy(graph)
    for key in ('id', 'revision', 'extra'):
        graph.pop(key, None)
    for node in graph['nodes']:
        node.pop('title', None)
        if node['id'] == 1:
            node['widgets_values'][0] = ''
            node['widgets_values_named']['image'] = ''
            node.get('properties', {}).pop('image', None)
        if node['id'] == 9:
            node['widgets_values'][1] = ''
            node['widgets_values_named']['filename_prefix'] = ''
    return graph


def validate_graph(graph):
    nodes = {n['id']: n for n in graph['nodes']}
    links = {r[0]: r for r in graph['links']}
    if len(nodes) != 16 or len(links) != 31:
        raise ValueError('Approved v2 graph topology changed.')
    if len(nodes) != len(graph['nodes']) or len(links) != len(graph['links']):
        raise ValueError('Duplicate IDs.')
    for lid, src, out, dst, inp, kind in links.values():
        a, b = nodes[src]['outputs'][out], nodes[dst]['inputs'][inp]
        if a['type'] != kind or b['type'] != kind:
            raise ValueError('Mismatched socket types.')
        if lid not in (a.get('links') or []) or b.get('link') != lid:
            raise ValueError('Inconsistent graph connection.')
    for node in nodes.values():
        for slot, port in enumerate(node.get('inputs', [])):
            lid = port.get('link')
            if lid is not None and (lid not in links or links[lid][3:5] != [node['id'], slot]):
                raise ValueError('Dangling input.')
        for slot, port in enumerate(node.get('outputs', [])):
            for lid in port.get('links') or []:
                if lid not in links or links[lid][1:3] != [node['id'], slot]:
                    raise ValueError('Dangling output.')
        named, positional = node.get('widgets_values_named'), node.get('widgets_values', [])
        if named is not None and list(named.values()) != positional:
            raise ValueError('Widget serialization disagrees for node ' + str(node['id']))
        if node.get('mode') != 0:
            raise ValueError('An approved node is muted or bypassed.')
    expected = {
        2: [True, 'watermark', 0.5, 2, -1, 'full photo', 'ignore', 0],
        3: [2048, 3072, 128, False], 13: [2048, 3072, 128, False],
        4: ['BF16 quality', 'default', 'auto', 'qwen_image_2.1_bf16.safetensors'],
        6: [12345, 'fixed', 40, 1, 'euler', 'simple', 1], 8: [0, 0],
    }
    if any(nodes[i]['widgets_values'] != values for i, values in expected.items()):
        raise ValueError('Approved inference defaults changed.')
    if nodes[9]['widgets_values'][0] is not True:
        raise ValueError('The workflow must open with generation enabled.')
    if nodes[5]['type'] != 'TextEncodeQwenImage21' or nodes[10]['type'] != 'SplitImageWithAlpha':
        raise ValueError('Native mask conditioning / RGB-only handling changed.')
    if nodes[5]['widgets_values'][2] != 0:
        raise ValueError('The prepared crop must not be resized during encoding.')
    forbidden = {'WBQwenPromptControl', 'TextGenerate', 'Q21PhotoEncode', 'Q21PhotoLoRAs'}
    if forbidden & {n['type'] for n in nodes.values()}:
        raise ValueError('An extra generation helper entered the approved graph.')
    def input_source(node_id, name):
        port = next(p for p in nodes[node_id]['inputs'] if p['name'] == name)
        return links[port['link']][1:3]
    contracts = [(5, 'images.image_1', [15, 0]), (5, 'images.image_2', [13, 1]),
                 (8, 'crop_data', [3, 0]), (8, 'generated_crop', [10, 0]),
                 (9, 'edited_result', [8, 0])]
    for node_id, name, expected_source in contracts:
        if input_source(node_id, name) != expected_source:
            raise ValueError('Mask guidance or final stitch was rewired.')


def run(root=ROOT):
    root = Path(root)
    graph = json.loads((root / WORKFLOW_PATH).read_text())
    validate_graph(graph)
    fixture = root / 'tests/fixtures/sam_local_edit/approved_v2_source.json'
    if fixture.is_file() and normalized(graph) != normalized(json.loads(fixture.read_text())):
        raise ValueError('Changes extend beyond naming, output prefix, and removal of a private upload filename.')
    before = {k: os.environ.get(k) for k in ('WB_APPROVED_ONLY', 'WB_WORKSPACE')}
    try:
        os.environ.update(WB_APPROVED_ONLY='1', WB_WORKSPACE='qwen')
        tasks = catalog_override('tasks.json', root)['tasks']
        assets = catalog_override('assets.json', root)['assets']
        models = catalog_override('models.json', root)['models']
        assert [t['id'] for t in tasks] == ['QLOCAL']
        assert len(assets) == 4 and {a['group'] for a in assets} == {'qwen', 'sam'}
        assert len(models) == 1 and models[0]['external_nodes'] == []
        assert models[0]['default_groups'] == ['qwen', 'sam']
        assert models[0]['local_nodes'] == LOCAL_NODES
    finally:
        for key, value in before.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
    return {'approved_tasks': ['QLOCAL'], 'graph_nodes': 16, 'graph_connections': 31,
            'model_assets': 4, 'model_groups': ['qwen', 'sam'],
            'scope': 'Static graph and profile contracts only; no Docker/GPU execution.'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
