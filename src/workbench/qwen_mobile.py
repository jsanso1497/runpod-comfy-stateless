"""Additive native Qwen 2.1 App Mode graph. No custom inference nodes or PE."""
from __future__ import annotations
import copy
import json
from pathlib import Path

VERSION = 'qwen-mobile-jupyter-1.0.0'
TASK_ID = 'QAPP'
WORKFLOW_PATH = 'workflows/qwen/QAPP/00_Qwen21_Native_Mobile_App_BF16.json'
API_PATH = 'automation/qwen/QAPP_Native_Mobile_BF16.api.json'
COMFY_COMMIT = 'bab80ef793ac42ad3f25589a6b0507ca537b1dd1'
FRONTEND_VERSION = '1.55.17'
MODEL = 'qwen_image_2.1_bf16.safetensors'
ENCODER = 'qwen3vl_8b_bf16.safetensors'
VAE = 'qwen_image_2.1_vae_bf16.safetensors'
DEFAULT_PROMPT = 'Make this photograph sharper with natural lighting and realistic skin texture.'


def task_row() -> dict:
    return {
        'id': TASK_ID, 'family': 'qwen', 'workspaces': ['qwen'], 'category': 'tasks',
        'title': 'Qwen Mobile: Native BF16 Photo Edit',
        'description': 'One image and a direct instruction in native ComfyUI App Mode. '
                       'Full BF16, native sampling, no prompt enhancer, LoRA, SAM or upscaler on this graph. '
                       'Your separate SAM and other Qwen workflows remain available.',
        'requires': ['qwen'], 'status': 'Static contract tested; live GPU/browser validation pending',
        'sources': ['https://docs.comfy.org/interface/app-mode',
                    'https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1'],
        'variants': [{'label': 'Mobile App / one photo / direct prompt / BF16',
                      'file': WORKFLOW_PATH,
                      'source': 'Native core nodes and pinned ComfyUI App Mode metadata'}],
        'required_nodes': ['LoadImage', 'PrimitiveStringMultiline', 'UNETLoader', 'CLIPLoader',
                           'VAELoader', 'TextEncodeQwenImage21', 'QwenImage21Cache',
                           'KSampler', 'VAEDecode', 'SaveImage'],
    }


def apply_catalog(name: str, value: dict) -> dict:
    """Called before qwen_only. Never change existing task graphs or assets."""
    if name == 'tasks.json':
        if any(t.get('id') == TASK_ID for t in value['tasks']):
            raise ValueError('Duplicate QAPP task; review the mobile add-on.')
        value['tasks'].append(task_row())
    return value


def _node(ident, kind, title, pos, inputs, outputs, values, named):
    return {
        'id': ident, 'type': kind, 'title': title, 'pos': list(pos), 'size': [330, 220],
        'flags': {}, 'order': ident - 1, 'mode': 0,
        'inputs': [{'name': n, 'type': t, 'link': None} for n, t in inputs],
        'outputs': [{'name': n, 'type': t, 'links': []} for n, t in outputs],
        'properties': {'Node name for S&R': kind, 'cnr_id': 'comfy-core'},
        'widgets_values': values, 'widgets_values_named': named,
    }


def make_workflow() -> dict:
    n = [
        _node(1, 'LoadImage', '1. Source photo', (0, 0), [], [('IMAGE', 'IMAGE'), ('MASK', 'MASK')],
              ['', 'image'], {'image': '', 'upload': 'image'}),
        _node(2, 'PrimitiveStringMultiline', '2. Edit instruction', (0, 490), [], [('STRING', 'STRING')],
              [DEFAULT_PROMPT], {'value': DEFAULT_PROMPT}),
        _node(3, 'CLIPLoader', 'Required Qwen image encoder: BF16', (400, 0), [], [('CLIP', 'CLIP')],
              [ENCODER, 'qwen_image', 'default'],
              {'clip_name': ENCODER, 'type': 'qwen_image', 'device': 'default'}),
        _node(4, 'UNETLoader', 'Qwen Image 2.1: BF16, not Turbo', (400, 250), [], [('MODEL', 'MODEL')],
              [MODEL, 'default'], {'unet_name': MODEL, 'weight_dtype': 'default'}),
        _node(5, 'VAELoader', 'Matching Qwen 2.1 VAE', (400, 500), [], [('VAE', 'VAE')],
              [VAE], {'vae_name': VAE}),
        _node(6, 'TextEncodeQwenImage21', '3. Native conditioning and image size', (800, 0),
              [('clip', 'CLIP'), ('images.image_1', 'IMAGE'), ('vae', 'VAE'), ('prompt', 'STRING')],
              [('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'), ('latent', 'LATENT')],
              ['', '', 2048], {'prompt': '', 'negative_prompt': '', 'resolution': 2048}),
        _node(7, 'QwenImage21Cache', 'Lossless cache: auto device', (800, 490),
              [('model', 'MODEL')], [('MODEL', 'MODEL')], ['auto', 'default'],
              {'device': 'auto', 'dtype': 'default'}),
        _node(8, 'KSampler', '4. Native sampling', (1200, 0),
              [('model', 'MODEL'), ('positive', 'CONDITIONING'), ('negative', 'CONDITIONING'),
               ('latent_image', 'LATENT')], [('LATENT', 'LATENT')],
              [0, 'randomize', 25, 1.0, 'euler', 'simple', 1.0],
              {'seed': 0, 'control_after_generate': 'randomize', 'steps': 25, 'cfg': 1.0,
               'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}),
        _node(9, 'VAEDecode', 'Decode', (1600, 0), [('samples', 'LATENT'), ('vae', 'VAE')],
              [('IMAGE', 'IMAGE')], [], {}),
        _node(10, 'SaveImage', '5. Result: saved PNG', (1950, 0), [('images', 'IMAGE')], [],
              ['Qwen_Mobile/Qwen21'], {'filename_prefix': 'Qwen_Mobile/Qwen21'}),
    ]
    n[0]['size'] = [330, 430]
    n[1]['size'] = [330, 230]
    n[5]['size'] = [340, 410]
    n[5]['inputs'][1]['shape'] = 7
    n[5]['inputs'][2]['shape'] = 7
    n[5]['inputs'][3]['widget'] = {'name': 'prompt'}
    n[7]['size'] = [330, 300]
    n[9]['size'] = [500, 620]
    links = []
    by_id = {x['id']: x for x in n}

    def link(src, out, dst, inp, kind):
        lid = len(links) + 1
        links.append([lid, src, out, dst, inp, kind])
        by_id[src]['outputs'][out]['links'].append(lid)
        by_id[dst]['inputs'][inp]['link'] = lid

    for args in [(1, 0, 6, 1, 'IMAGE'), (2, 0, 6, 3, 'STRING'),
                 (3, 0, 6, 0, 'CLIP'), (5, 0, 6, 2, 'VAE'),
                 (4, 0, 7, 0, 'MODEL'), (7, 0, 8, 0, 'MODEL'),
                 (6, 0, 8, 1, 'CONDITIONING'), (6, 1, 8, 2, 'CONDITIONING'),
                 (6, 2, 8, 3, 'LATENT'), (8, 0, 9, 0, 'LATENT'),
                 (5, 0, 9, 1, 'VAE'), (9, 0, 10, 0, 'IMAGE')]:
        link(*args)
    graph = {
        'id': '148ce399-405f-4fda-948d-07be1865032b', 'revision': 0,
        'last_node_id': 10, 'last_link_id': len(links), 'nodes': n, 'links': links,
        'groups': [], 'config': {}, 'version': 0.4,
        'extra': {
            'ds': {'scale': 0.55, 'offset': [60, 60]},
            'linearMode': True,
            'linearData': {
                'inputs': [
                    [1, 'image', {'description': 'Upload one source photo. JPEG or PNG recommended.'}],
                    [2, 'value', {'height': 180, 'description': 'Describe the edit. Sent directly to Qwen without rewriting.'}],
                    [6, 'resolution', {'description': '2048: about 4 MP, preserves aspect ratio. 1536: about 2.4 MP. 1024: about 1 MP. Avoid 0 for large phone photos.'}],
                    [8, 'seed', {'description': 'Use a fixed seed while comparing prompt changes.'}],
                    [8, 'control_after_generate', {'description': 'randomize for new variations; fixed for repeatable comparisons.'}],
                ], 'outputs': [10],
            },
            'workbench_mobile': {
                'version': VERSION, 'comfy_commit': COMFY_COMMIT,
                'frontend_verified_schema': FRONTEND_VERSION,
                'default_resolution': 2048, 'prompt_enhancer': False,
                'gpu_tested': False, 'browser_tested': False,
                'notes': 'Native full-image generation is not guaranteed pixel-aligned. No SAM or crop/stitch in this simple app. '
                         'Resolution uses the native square-pixel-budget control, not an exact longest edge. '
                         'No quantization or extra model downloads. SaveImage writes PNG under output/Qwen_Mobile.',
            },
        },
    }
    validate_workflow(graph)
    return graph


def make_api(graph: dict | None = None) -> dict:
    """Reference API export for validation; the UI JSON remains the user-facing app."""
    graph = graph or make_workflow()
    lookup = {r[0]: r for r in graph['links']}
    result = {}
    for node in graph['nodes']:
        values = copy.deepcopy(node.get('widgets_values_named', {}))
        # Frontend-only upload/control widgets are not backend API inputs.
        for key in ('upload', 'control_after_generate'):
            values.pop(key, None)
        for inp in node['inputs']:
            if inp.get('link') is not None:
                edge = lookup[inp['link']]
                values[inp['name']] = [str(edge[1]), edge[2]]
        result[str(node['id'])] = {'class_type': node['type'], 'inputs': values,
                                  '_meta': {'title': node['title']}}
    return result


def validate_workflow(graph: dict) -> None:
    nodes = {n['id']: n for n in graph['nodes']}
    links = {r[0]: r for r in graph['links']}
    if len(nodes) != len(graph['nodes']) or len(links) != len(graph['links']):
        raise ValueError('Duplicate node/link IDs.')
    for lid, src, out, dst, inp, kind in links.values():
        a, b = nodes[src]['outputs'][out], nodes[dst]['inputs'][inp]
        if lid not in a['links'] or b.get('link') != lid or a['type'] != kind or b['type'] != kind:
            raise ValueError('Broken workflow link/socket contract.')
    for node in nodes.values():
        for inp in node['inputs']:
            if inp.get('link') is not None and inp['link'] not in links:
                raise ValueError('Dangling input.')
        for out in node['outputs']:
            if any(lid not in links for lid in out['links']):
                raise ValueError('Dangling output.')
    for node_id, widget, *_ in graph['extra']['linearData']['inputs']:
        if widget not in nodes[node_id]['widgets_values_named']:
            raise ValueError('App input does not resolve to a widget.')
    if graph['extra']['linearData']['outputs'] != [10] or nodes[10]['type'] != 'SaveImage':
        raise ValueError('App output is not the saved image.')
    if graph['extra']['linearMode'] is not True:
        raise ValueError('App Mode must be the default view.')
    api = make_api_unchecked(graph)
    if api['8']['inputs']['latent_image'] != ['6', 2]:
        raise ValueError('Sampling must use the native first-image latent size.')
    if {nodes[k]['widgets_values'][0] for k in (3, 4, 5)} != {MODEL, ENCODER, VAE}:
        raise ValueError('Only the three existing BF16 model files are allowed.')
    if set(n['type'] for n in nodes.values()) != set(task_row()['required_nodes']):
        raise ValueError('Unexpected generation/extension node in app.')


def make_api_unchecked(graph):
    return make_api(graph)


def write_workflows(root: Path) -> None:
    graph = make_workflow()
    for relative, value in [(WORKFLOW_PATH, graph), (API_PATH, make_api(graph))]:
        path = Path(root) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    write_workflows(args.root)
