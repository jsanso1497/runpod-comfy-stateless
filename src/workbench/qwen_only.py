"""Runtime-only Qwen profile: no MOSS or prompt-enhancement model preparation.

Source workflows and other workspace families are not rewritten. Native Qwen
workflows are projected on read, with their enhancer branches removed rather
than left connected to unavailable CLIP files. Main Qwen weights are unchanged.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import uuid

VERSION = 'qwen-only-1'
ENVIRONMENT_KEY = 'WB_QWEN_ONLY'
PROMPT_GROUPS = frozenset({'ollama', 'qwen_pe_i2i', 'qwen_stock_pe'})
EXCLUDED_GROUPS = PROMPT_GROUPS | {'moss_image'}
HIDDEN_TASKS = frozenset({'QMOSS', 'QMOSS_I2I', 'U10', 'U11'})


def enabled() -> bool:
    value = os.environ.get(ENVIRONMENT_KEY, '0')
    if value not in ('0', '1'):
        raise ValueError('WB_QWEN_ONLY must be 0 or 1.')
    return value == '1' and os.environ.get('WB_WORKSPACE', 'qwen') == 'qwen'


def apply_catalog(name: str, value: dict) -> dict:
    if not enabled() or name not in ('models.json', 'tasks.json', 'assets.json'):
        return value
    # read_catalog already owns a fresh object; do not modify disk catalogs.
    if name == 'models.json':
        for model in value['models']:
            if model['id'] != 'qwen':
                continue
            for key in ('default_groups', 'toolbox'):
                model[key] = [g for g in model.get(key, []) if g not in EXCLUDED_GROUPS]
            model['default_tasks'] = [t for t in model.get('default_tasks', []) if t not in HIDDEN_TASKS]
            model.pop('experimental_moss', None)
            model['qwen_only_profile'] = True
            model['description'] = ('Qwen image editing with direct prompts. SAM and SeedVR2 remain available. '
                                    'MOSS and prompt-enhancer model downloads are disabled by WB_QWEN_ONLY=1.')
            model['storage_gb'] = 150
            model['hardware_note'] = ('Qwen-only planning profile. Additional storage is needed for private '
                                      'models and outputs. GPU inference and peak memory are not certified.')
    elif name == 'tasks.json':
        tasks = []
        for task in value['tasks']:
            if 'qwen' not in task.get('workspaces', []):
                tasks.append(task)
                continue
            if task['id'] in HIDDEN_TASKS or task.get('family') == 'moss':
                # Keep a shared task for its other workspace, never display it in Qwen.
                task['workspaces'] = [w for w in task['workspaces'] if w != 'qwen']
                if task['workspaces']:
                    tasks.append(task)
                continue
            task['requires'] = [g for g in task.get('requires', []) if g not in EXCLUDED_GROUPS]
            if 'optional_prompt_enhancement' in task:
                task['optional_prompt_enhancement'] = {'default': 'off', 'enabled': False,
                                                      'reason': 'WB_QWEN_ONLY=1'}
            task['description'] = task.get('description', '') + ' Qwen-only mode: direct prompt, enhancer disabled.'
            task['required_nodes'] = [n for n in task.get('required_nodes', []) if n != 'TextGenerate']
            tasks.append(task)
        value['tasks'] = tasks
    else:
        value['assets'] = [a for a in value['assets'] if a.get('group') not in EXCLUDED_GROUPS]
        empty_helper = {'group': 'ollama', 'models': [], 'estimated_bytes': 0,
                        'note': 'Disabled by WB_QWEN_ONLY=1'}
        value['ollama'] = dict(empty_helper)
        value.setdefault('ollama_by_workspace', {})['qwen'] = dict(empty_helper)
    return value


def direct_prompt_result(prompt):
    """Same pass-through report shape as the existing enhancer's OFF branch."""
    return prompt, {'enabled': False, 'original_prompt': prompt, 'effective_prompt': prompt,
                    'status': 'OFF: disabled by WB_QWEN_ONLY=1; original prompt unchanged',
                    'cache_hit': False, 'fallback_used': False, 'policy': ENVIRONMENT_KEY}


def _rebuild_subgraph_ports(subgraph):
    links = subgraph['links']
    input_id = subgraph['inputNode']['id']
    output_id = subgraph['outputNode']['id']
    for node in subgraph['nodes']:
        for slot, port in enumerate(node.get('inputs', [])):
            incoming = [r['id'] for r in links if r['target_id'] == node['id'] and r['target_slot'] == slot]
            if len(incoming) > 1:
                raise ValueError('Native Qwen subgraph has duplicate input links.')
            port['link'] = incoming[0] if incoming else None
        for slot, port in enumerate(node.get('outputs', [])):
            port['links'] = [r['id'] for r in links if r['origin_id'] == node['id'] and r['origin_slot'] == slot]
    for slot, port in enumerate(subgraph['inputs']):
        port['linkIds'] = [r['id'] for r in links if r['origin_id'] == input_id and r['origin_slot'] == slot]
    for slot, port in enumerate(subgraph['outputs']):
        port['linkIds'] = [r['id'] for r in links if r['target_id'] == output_id and r['target_slot'] == slot]


def _strip_native_enhancer(subgraph):
    encoders = [n for n in subgraph.get('nodes', []) if n['type'] == 'TextEncodeQwenImage21']
    if not encoders or not any(n['type'] == 'TextGenerate' for n in subgraph['nodes']):
        return None
    if len(encoders) != 1 or subgraph.get('widgets'):
        raise ValueError('Unrecognized native Qwen subgraph. Refusing an unsafe prompt bypass.')
    encoder = encoders[0]
    interface = subgraph['inputs']
    prompt_inputs = [i for i, p in enumerate(interface) if p['name'] == 'prompt']
    prompt_slots = [i for i, p in enumerate(encoder['inputs']) if p['name'] == 'prompt']
    if len(prompt_inputs) != 1 or len(prompt_slots) != 1:
        raise ValueError('Native Qwen prompt interface could not be resolved.')
    incoming = [r for r in subgraph['links'] if r['target_id'] == encoder['id']
                and r['target_slot'] == prompt_slots[0]]
    if len(incoming) != 1:
        raise ValueError('Native Qwen conditioning has no unique prompt connection.')
    input_id = subgraph['inputNode']['id']
    output_id = subgraph['outputNode']['id']
    # Directly preserve the user's prompt, without its rewrite switch.
    incoming[0].update(origin_id=input_id, origin_slot=prompt_inputs[0])
    by_id = {n['id']: n for n in subgraph['nodes']}
    required = {output_id}
    todo = [output_id]
    while todo:
        target = todo.pop()
        for row in subgraph['links']:
            source = row['origin_id']
            if row['target_id'] == target and source != input_id and source not in required:
                if source not in by_id:
                    raise ValueError('Native Qwen subgraph contains an unresolved source.')
                required.add(source)
                todo.append(source)
    kept_nodes = [n for n in subgraph['nodes'] if n['id'] in required]
    if any(n['type'] == 'TextGenerate' for n in kept_nodes):
        raise ValueError('Prompt generator remains on the image path; profile cannot bypass it safely.')
    kept_ids = {n['id'] for n in kept_nodes}
    links = [r for r in subgraph['links'] if r['target_id'] in kept_ids | {output_id}
             and r['origin_id'] in kept_ids | {input_id}]
    used_inputs = {r['origin_slot'] for r in links if r['origin_id'] == input_id}
    index_map = {old: new for new, old in enumerate(i for i in range(len(interface)) if i in used_inputs)}
    names = {interface[i]['name'] for i in used_inputs}
    for row in links:
        if row['origin_id'] == input_id:
            row['origin_slot'] = index_map[row['origin_slot']]
    old_id = subgraph['id']
    new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, VERSION + '/' + old_id))
    subgraph['id'] = new_id
    subgraph['inputs'] = [p for i, p in enumerate(interface) if i in used_inputs]
    subgraph['nodes'] = kept_nodes
    subgraph['links'] = links
    subgraph['groups'] = [g for g in subgraph.get('groups', []) if g.get('title') != 'Prompt Enhancer']
    subgraph['name'] = 'Qwen Image Edit: direct prompt (no enhancer models)'
    subgraph.setdefault('extra', {})['qwen_only_profile'] = VERSION
    _rebuild_subgraph_ports(subgraph)
    return old_id, new_id, names


def _link_values(container):
    # Outer LiteGraph links are arrays; inner subgraph links are dictionaries.
    for row in container.get('links', []):
        if isinstance(row, dict):
            yield row, row['id'], row['origin_id'], row['origin_slot'], row['target_id'], row['target_slot']
        else:
            yield row, *row[:5]


def _update_instance(container, node, old_id, new_id, names):
    named = node.get('widgets_values_named', {})
    positional = node.get('widgets_values', [])
    keys = list(named)
    if positional and (len(positional) != len(keys) or any(positional[i] != named[k] for i, k in enumerate(keys))):
        raise ValueError('Native Qwen widget serialization changed; refusing to shuffle controls.')
    node['widgets_values'] = [named[k] for k in keys if k in names]
    node['widgets_values_named'] = {k: named[k] for k in keys if k in names}
    old_ports = node.get('inputs', [])
    mapping = {i: j for j, i in enumerate(i for i, p in enumerate(old_ports) if p['name'] in names)}
    links = []
    removed = set()
    for row, lid, src, out, dst, inp in _link_values(container):
        if dst == node['id']:
            if inp not in mapping:
                removed.add(lid)
                continue
            if isinstance(row, dict):
                row['target_slot'] = mapping[inp]
            else:
                row[4] = mapping[inp]
        links.append(row)
    container['links'] = links
    for other in container['nodes']:
        for port in other.get('outputs', []):
            port['links'] = [lid for lid in (port.get('links') or []) if lid not in removed]
    node['inputs'] = [p for i, p in enumerate(old_ports) if i in mapping]
    node['type'] = new_id
    properties = node.setdefault('properties', {})
    if properties.get('Node name for S&R') == old_id:
        properties['Node name for S&R'] = new_id
    properties['previewExposures'] = []
    if 'proxyWidgets' in properties:
        properties.pop('proxyWidgets')
    node['title'] = 'Qwen BF16 | Direct prompt | Enhancement disabled'


def project_workflow(graph: dict) -> dict:
    if not enabled():
        return graph
    if graph.get('extra', {}).get('qwen_only_profile') == VERSION:
        return graph
    graph = copy.deepcopy(graph)
    subgraphs = graph.get('definitions', {}).get('subgraphs', [])
    changes = [result for sub in subgraphs if (result := _strip_native_enhancer(sub)) is not None]
    for container in [graph] + subgraphs:
        for node in container.get('nodes', []):
            for old_id, new_id, names in changes:
                if node['type'] == old_id:
                    _update_instance(container, node, old_id, new_id, names)
            if node['type'] == 'WBQwenPromptControl':
                # The runtime also enforces OFF, including for old saved graphs.
                values = node.get('widgets_values', [])
                if not values or type(values[0]) is not bool:
                    raise ValueError('Unexpected Qwen enhancer control layout.')
                values[0] = False
                if 'widgets_values_named' in node:
                    node['widgets_values_named']['enabled'] = False
                node['title'] = 'Prompt enhancer disabled by Qwen-only profile'
        if container is not graph:
            _rebuild_subgraph_ports(container)
    graph.setdefault('extra', {})['qwen_only_profile'] = VERSION
    if graph.get('id'):
        graph['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, VERSION + '/' + graph['id']))
    return graph


def workflow_bytes(raw: bytes) -> bytes:
    if not enabled():
        return raw
    graph = project_workflow(json.loads(raw))
    return (json.dumps(graph, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def load_workflow(path):
    return json.loads(workflow_bytes(Path(path).read_bytes()))
