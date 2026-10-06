#!/usr/bin/env python3
"""Verify portrait source, image bundle and active node agree before downloads.

The manifest is created automatically inside Docker. Users never maintain hashes.
This checks file integrity and wiring policy, not model quality or GPU inference.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re

EXPECTED_VERSION = '1.4.0'
EXPECTED_PIPELINE = 'role-routed-reference-v4'
HERE = Path(__file__).resolve().parent
NODE_FILES = ('__init__.py', 'logic.py', 'ollama_client.py', 'analysis_prompt.txt',
              'system_prompt.txt', 'reference_roles.py', 'video_export.py', 'native_helpers.py',
              'web/references.js', 'web/director_controls.js')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(source):
    source = Path(source)
    required = ('VERSION', 'settings.json', 'start.sh', 'prepare_assets.py',
                'ollama_service.py', 'smoke_check.py', 'install_workflows.py',
                'workflows/H3_Portrait_Auto.json', 'workflows/H3_Ref2VA_Standard.json',
                'profiles/lite/settings.json', 'profiles/lite/models.json')
    for relative in required + tuple('node/' + x for x in NODE_FILES):
        if not (source / relative).is_file():
            raise ValueError(f'Incomplete portrait upload: missing h3_portrait/{relative}')
    version = (source / 'VERSION').read_text().strip()
    cfg = json.loads((source / 'settings.json').read_text())
    if version != EXPECTED_VERSION or cfg.get('version') != EXPECTED_VERSION:
        raise ValueError('Portrait VERSION/settings disagree or belong to an older release.')
    lite = cfg.get('profile') == 'lite'
    if cfg.get('analysis_think') is not False or cfg.get('think') is not (not lite):
        raise ValueError('Prompt-stage thinking controls disagree with the selected profile.')
    if 'instruct' not in cfg.get('analysis_model', '').lower():
        raise ValueError('An Instruct reference model is required; a Thinking toggle is insufficient.')
    if lite:
        if cfg.get('analysis_model') != cfg.get('ollama_model') or '8b-instruct' not in cfg['ollama_model']:
            raise ValueError('Lite uses the same 8B Instruct model for both stages.')
        if 'pruned' in cfg['model_files']['diffusion'] or 'int8_convrot' not in cfg['model_files']['diffusion']:
            raise ValueError('Lite must retain the full Ref2VA architecture using INT8 weights.')
    elif cfg.get('analysis_model') == cfg.get('ollama_model') or '32b-thinking' not in cfg['ollama_model']:
        raise ValueError('Full uses separate Instruct analysis and 32B Thinking direction.')
    module = ast.parse((source / 'node/ollama_client.py').read_text())
    assignments = {n.targets[0].id: ast.literal_eval(n.value)
                   for n in module.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and isinstance(n.value, ast.Constant)}
    if assignments.get('PIPELINE_REVISION') != EXPECTED_PIPELINE:
        raise ValueError('Older ollama_client.py detected. Upload the complete h3_portrait/node folder.')
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'generate')
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == '_request_json']
    calls.sort(key=lambda n: n.lineno)
    controls = [next((ast.unparse(k.value) for k in c.keywords if k.arg == 'think'), None) for c in calls]
    if controls != ['False', 'director_think']:
        raise ValueError('Expected compact analysis followed by profile-controlled direction.')
    if 'rr.route_analysis(' not in (source / 'node/ollama_client.py').read_text():
        raise ValueError('Missing reference-role routing.')
    init_module = ast.parse((source / 'node/__init__.py').read_text())
    init_version = next((ast.literal_eval(n.value) for n in init_module.body
                         if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                         and any(isinstance(t, ast.Name) and t.id == 'PACKAGE_VERSION' for t in n.targets)), None)
    if init_version != EXPECTED_VERSION:
        raise ValueError('Older node/__init__.py detected. Upload the complete portrait node folder.')
    init_calls = {n.func.attr for n in ast.walk(init_module) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                  and n.func.value.id == 'oc'}
    if not {'pipeline_info', 'pipeline_digest'} <= init_calls:
        raise ValueError('The portrait node is missing the split-model routing/cache implementation.')
    for f in (source / 'node').rglob('*.py'):
        ast.parse(f.read_text(), filename=str(f))
    print(f'H3 PORTRAIT SOURCE VERIFIED: {version} | {EXPECTED_PIPELINE}', flush=True)
    return cfg


def node_hashes(path):
    path = Path(path)
    values = {}
    for name in NODE_FILES:
        f = path / name
        if not f.is_file():
            raise ValueError(f'Missing installed portrait node file: {f}')
        values[name] = digest(f)
    return values


def compare_node(source, node):
    expected = node_hashes(Path(source) / 'node')
    observed = node_hashes(node)
    mismatched = [name for name in expected if expected[name] != observed[name]]
    if mismatched:
        raise ValueError('Portrait node copy does not match this source: ' + ', '.join(mismatched))


def asset_hashes(source):
    return {name:digest(Path(source)/name) for name in ('settings.json','models.json','install_workflows.py','workflows/H3_Portrait_Auto.json','workflows/H3_Ref2VA_Standard.json')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=HERE)
    p.add_argument('--node-copy', action='append', type=Path, default=[])
    p.add_argument('--write-manifest', type=Path)
    p.add_argument('--check-manifest', type=Path)
    p.add_argument('--source-revision', default=os.environ.get('H3_SOURCE_REVISION', 'local-uncommitted'))
    a = p.parse_args()
    verify_source(a.source)
    if not (a.source_revision == 'local-uncommitted' or re.fullmatch(r'[0-9a-f]{40}', a.source_revision)):
        raise ValueError('Build source revision must be a full Git commit or local-uncommitted.')
    hashes = node_hashes(a.source / 'node')
    for copy in a.node_copy:
        compare_node(a.source, copy)
    if a.check_manifest:
        saved = json.loads(a.check_manifest.read_text())
        if saved.get('version') != EXPECTED_VERSION or saved.get('node_sha256') != hashes:
            raise ValueError('Image release manifest does not match the installed portrait source.')
        if saved.get('source_revision') != a.source_revision:
            raise ValueError('Image source revision and release manifest disagree.')
        if saved.get('assets_sha256') != asset_hashes(a.source):
            raise ValueError('Workflow/catalog/source assets do not match the build manifest.')
        if saved.get('settings_sha256') != digest(a.source / 'settings.json'):
            raise ValueError('Image settings do not match the build manifest.')
    if a.write_manifest:
        a.write_manifest.write_text(json.dumps({
            'version': EXPECTED_VERSION, 'pipeline': EXPECTED_PIPELINE,
            'source_revision': a.source_revision, 'node_sha256': hashes,
            'settings_sha256': digest(a.source / 'settings.json'), 'assets_sha256':asset_hashes(a.source),
        }, indent=2) + '\n')
    print('H3 PORTRAIT RELEASE VERIFIED: ' + EXPECTED_VERSION +
          ' | source=' + a.source_revision, flush=True)


if __name__ == '__main__':
    main()
