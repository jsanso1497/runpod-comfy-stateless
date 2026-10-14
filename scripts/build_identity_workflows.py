#!/usr/bin/env python3
"""Fetch pinned public developer JSON at build time and emit BF16 factory recipes.

No model downloads and no API keys. This runs in GitHub Actions before catalog
validation, and during a direct Docker build. Input snapshots and provenance are
kept inside the resulting image. Use --offline only after source snapshots exist.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from workbench.identity_catalog import FILES, workflow_path
from workbench.krea_practices import adapt_developer_runtime
import setup_identity_edits as adapter


def fetch_source(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'Workbench-Developer-Recipes/1.1'}), timeout=90) as response:
                raw = response.read(8 * 1024**2 + 1)
            if len(raw) > 8 * 1024**2:
                raise ValueError('Developer workflow exceeds the size limit.')
            graph = json.loads(raw)
            if not isinstance(graph, dict) or not isinstance(graph.get('nodes'), list):
                raise ValueError('Developer download is not a frontend workflow.')
            return raw
        except (OSError, ValueError):
            if attempt == 2:
                raise RuntimeError('Unable to fetch/parse pinned developer JSON: ' + url) from None
            time.sleep(attempt + 1)


def add_optional_krea_loras(graph):
    """Native optional adapters, bypassed by default, after Identity Edit."""
    by_id = {n['id']: n for n in graph['nodes']}
    patch = by_id[79]
    index = next(i for i, p in enumerate(patch['inputs']) if p['name'] == 'model')
    old_id = patch['inputs'][index]['link']
    old = next(l for l in graph['links'] if l[0] == old_id)
    source, source_slot = old[1], old[2]
    first_id = max(n['id'] for n in graph['nodes']) + 1
    next_link = max(l[0] for l in graph['links']) + 1
    graph['links'].remove(old)
    by_id[source]['outputs'][source_slot]['links'].remove(old_id)
    chain = [(old_id, source, source_slot, first_id, 0, 'MODEL'),
             (next_link, first_id, 0, first_id + 1, 0, 'MODEL'),
             (next_link + 1, first_id + 1, 0, 79, index, 'MODEL')]
    by_id[source]['outputs'][source_slot]['links'].append(old_id)
    patch['inputs'][index]['link'] = next_link + 1
    for j in range(2):
        graph['nodes'].append({
            'id': first_id + j, 'type': 'LoraLoaderModelOnly', 'pos': [2320 + j * 400, 1390],
            'size': [360, 140], 'flags': {}, 'order': 30 + j, 'mode': 4,
            'title': f'Optional LoRA {j + 1} - BYPASSED',
            'inputs': [{'name': 'model', 'type': 'MODEL', 'link': chain[j][0]}],
            'outputs': [{'name': 'MODEL', 'type': 'MODEL', 'links': [chain[j + 1][0]]}],
            'properties': {'Node name for S&R': 'LoraLoaderModelOnly'},
            'widgets_values': [adapter.KEDIT, 0.0],
        })
    graph['links'].extend([list(l) for l in chain])
    graph['nodes'].append({
        'id': first_id + 2, 'type': 'Note', 'pos': [2320, 1590], 'size': [800, 210],
        'flags': {}, 'order': 32, 'mode': 0, 'inputs': [], 'outputs': [], 'properties': {},
        'title': 'Workbench BF16 adaptation + optional private library',
        'widgets_values': ['The original developer JSON is archived under /opt/workbench/vendor/identity. '
            'Choose a compatible Krea 2 fine-tune in UNETLoader. Use the Raw or Turbo preset that matches it. '
            'Optional LoRA nodes are bypassed: select your own file, set its strength, then remove Bypass. '
            'Keep the main Identity Edit v1.2 adapter at 1.0. Do not select it again in an optional slot. '
            'This graph has no prompt enhancer and no SAM dependency. Workbench sRGB image I/O is used. '
            'The pixel path pre-encodes at the target resolution; redundant full-size VAE encodes are omitted. '
            'Output follows ResolutionSelector, '
            'not the source aspect ratio automatically. Set the desired ratio before Run.'],
    })
    graph['last_node_id'] = first_id + 2
    graph['last_link_id'] = next_link + 1
    return graph


def validate_topology(graph):
    """Check all ordinary top-level endpoints. Embedded schemas remain upstream."""
    nodes = {n['id']: n for n in graph['nodes']}
    if len(nodes) != len(graph['nodes']):
        raise ValueError('Duplicate node ID in developer graph.')
    links = {l[0]: l for l in graph.get('links', [])}
    if len(links) != len(graph.get('links', [])):
        raise ValueError('Duplicate link ID in developer graph.')
    for lid, row in links.items():
        if len(row) != 6:
            raise ValueError('Unexpected top-level graph link layout.')
        _, src, out, dst, inp, kind = row
        a = nodes[src]['outputs'][out]
        b = nodes[dst]['inputs'][inp]
        if lid not in (a.get('links') or []) or b.get('link') != lid:
            raise ValueError('Broken developer link/socket agreement.')
    for n in nodes.values():
        for p in n.get('inputs', []):
            if p.get('link') is not None and p['link'] not in links:
                raise ValueError('Missing input link.')
        for p in n.get('outputs', []):
            if any(l not in links for l in (p.get('links') or [])):
                raise ValueError('Missing output link.')


def build(offline=False):
    archive = ROOT / 'vendor/identity'
    archive.mkdir(parents=True, exist_ok=True)
    report = {'version': '1.1.1', 'adapter_version': adapter.VERSION, 'gpu_tested': False, 'sources': [], 'workflows': []}
    for family, task in [('qwen', 'QDEV'), ('krea', 'KDEV')]:
        originals = {}
        for name, url in adapter.recipes(family).items():
            # URL hash also prevents reuse of a snapshot from a different source pin.
            path = archive / family / (name + '-' + hashlib.sha256(url.encode()).hexdigest()[:12] + '.json')
            if offline:
                raw = path.read_bytes()
            else:
                raw = fetch_source(url)
            if family == 'krea':
                blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
                if blob != 'a707d8d27cb8504f8a63ab38babe3855ac797c59':
                    raise ValueError('Pinned Krea developer JSON did not match its Git blob ID.')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            originals[name] = json.loads(raw)
            report['sources'].append({'family': family, 'url': url, 'file': str(path.relative_to(ROOT)),
                                      'sha256': hashlib.sha256(raw).hexdigest()})
        generated = adapter.make_presets(family, originals)
        if {name for name, _ in generated} != {name for _, name in FILES[task]}:
            raise ValueError('Generated workflow names differ from the catalog.')
        for name, graph in generated:
            if family == 'krea':
                graph = add_optional_krea_loras(adapt_developer_runtime(graph))
            validate_topology(graph)
            path = ROOT / workflow_path(task, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = (json.dumps(graph, indent=2, ensure_ascii=False) + '\n').encode()
            path.write_bytes(data)
            report['workflows'].append({'file': str(path.relative_to(ROOT)),
                'sha256': hashlib.sha256(data).hexdigest(), 'required_nodes': sorted(adapter.required_types(graph))})
    (archive / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Built 8 BF16 developer-derived workflows. No model weights downloaded; GPU validation pending.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use already fetched snapshots, without network access.')
    args = parser.parse_args()
    build(args.offline)
