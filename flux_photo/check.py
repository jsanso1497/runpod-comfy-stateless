"""Static graph check and optional real CPU ComfyUI schema smoke test. No inference."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

from assets import validate_manifest
from make_workflows import api_graph, build

HERE = Path(__file__).resolve().parent


def validate_graph(graph, schemas=None):
    nodes = {n['id']: n for n in graph['nodes']}
    links = {l[0]: l for l in graph['links']}
    if len(nodes) != len(graph['nodes']) or len(links) != len(graph['links']):
        raise ValueError('Duplicate node or link IDs.')
    if graph['last_node_id'] != max(nodes) or graph['last_link_id'] != max(links):
        raise ValueError('Incorrect last ID counters.')
    edges = {n: [] for n in nodes}
    for lid, a, ai, b, bi, kind in links.values():
        src, dst = nodes[a]['outputs'][ai], nodes[b]['inputs'][bi]
        if src['type'] != kind or dst['type'] != kind or dst['link'] != lid or lid not in src['links']:
            raise ValueError('Broken link endpoints or types.')
        edges[a].append(b)
    visited, active = set(), set()
    def visit(n):
        if n in active: raise ValueError('Workflow cycle.')
        if n in visited: return
        active.add(n)
        for child in edges[n]: visit(child)
        active.remove(n); visited.add(n)
    for n in nodes: visit(n)
    for n in nodes.values():
        for inp in n['inputs']:
            if inp['link'] is None: raise ValueError('Required socket is disconnected.')
        for out in n['outputs']:
            if any(l not in links for l in out['links']): raise ValueError('Dangling output link.')
        if n['type'] == 'Note': continue
        if schemas is not None:
            if n['type'] not in schemas: raise ValueError('Missing runtime node: ' + n['type'])
            schema = schemas[n['type']]
            fields = {**schema['input'].get('required', {}), **schema['input'].get('optional', {})}
            actual = api_graph({'nodes': [n], 'links': graph['links']})[str(n['id'])]['inputs']
            missing = set(schema['input'].get('required', {})) - set(actual)
            if missing: raise ValueError(f'{n["type"]} missing required fields: {missing}')
            if set(actual) - set(fields): raise ValueError('Unknown runtime input field.')
            if tuple(o['type'] for o in n['outputs']) != tuple(schema['output']):
                raise ValueError('Runtime output socket mismatch for ' + n['type'])
            for name, value in n['widgets_values_named'].items():
                spec = fields[name]
                if isinstance(spec[0], list) and n['type'] != 'LoadImage' and value not in spec[0]:
                    raise ValueError('Invalid dropdown choice: ' + name)
    return True


def static_check():
    for p in HERE.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    validate_manifest(json.loads((HERE/'config/models.json').read_text()))
    for native in (False, True):
        name, expected = build(native)
        actual = json.loads((HERE/'workflows'/(name+'.json')).read_text())
        if expected != actual: raise ValueError('Workflow is out of sync with its generator.')
        validate_graph(actual)
        if api_graph(actual) != json.loads((HERE/'config'/(name+'.api.json')).read_text()):
            raise ValueError('UI/API workflow mismatch.')
    print('FLUX PHOTO STATIC CHECK PASS: 2 UI workflows and 2 matching API graphs.')


def build_smoke(home):
    home = Path(home)
    logpath = HERE/'build-smoke.log'
    process = None
    with logpath.open('w') as log:
        try:
            process = subprocess.Popen([sys.executable, 'main.py', '--cpu', '--listen', '127.0.0.1',
                                        '--port', '18191', '--disable-auto-launch'], cwd=home, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 240
            schema = None
            while time.monotonic() < deadline:
                if process.poll() is not None: break
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18191/object_info', timeout=5) as r:
                        schema = json.load(r)
                    break
                except (OSError, ValueError): time.sleep(2)
            if schema is None:
                raise RuntimeError('CPU ComfyUI did not expose schemas.\n' + logpath.read_text()[-16000:])
            if 'flux2' not in schema['CLIPLoader']['input']['required']['type'][0]:
                raise ValueError('Pinned core does not expose CLIPLoader type flux2.')
            for p in (HERE/'workflows').glob('*.json'):
                validate_graph(json.loads(p.read_text()), schema)
            print('REAL COMFY CPU SCHEMA SMOKE PASS. No model download or GPU generation was performed.')
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--build-smoke',action='store_true')
    ap.add_argument('--comfy-home',type=Path)
    args=ap.parse_args();static_check()
    if args.build_smoke:
        if args.comfy_home is None: ap.error('--comfy-home is required')
        build_smoke(args.comfy_home)
