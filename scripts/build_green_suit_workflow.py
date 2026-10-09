#!/usr/bin/env python3
"""Regenerate the add-on graph; requires the CPU dependencies used by repository CI."""
from pathlib import Path
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('green_suit_graph_nodes', ROOT/'src/nodes/everyday/green_suit.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

NODES = []
LINKS = []
API = {}

def add(identifier, node_type, title, pos, size, inputs=None, outputs=None, values=None):
    node = {'id': identifier, 'type': node_type, 'pos': pos, 'size': size, 'flags': {},
            'order': len(NODES), 'mode': 0,
            'inputs': [{'name': name, 'type': typ, 'link': None} for name, typ in (inputs or [])],
            'outputs': [{'name': name, 'type': typ, 'links': [], 'slot_index': index}
                        for index, (name, typ) in enumerate(outputs or [])],
            'properties': {'Node name for S&R': node_type}, 'widgets_values': values or [], 'title': title}
    NODES.append(node)
    return node

def custom(identifier, node_type, title, pos, size):
    cls = module.NODE_CLASS_MAPPINGS[node_type]
    schema = cls.INPUT_TYPES()
    inputs, values, params = [], [], {}
    for category in ('required', 'optional'):
        for name, field in schema.get(category, {}).items():
            typ = field[0]
            options = field[1] if len(field) > 1 else {}
            if isinstance(typ, list) or typ in ('INT', 'FLOAT', 'STRING', 'BOOLEAN') and not options.get('forceInput'):
                value = options.get('default', typ[0] if isinstance(typ, list) else '' if typ == 'STRING' else 0)
                values.append(value); params[name] = value
            else:
                inputs.append((name, typ))
    returns = list(zip(cls.RETURN_NAMES, cls.RETURN_TYPES)) if cls.RETURN_TYPES else []
    node = add(identifier, node_type, title, pos, size, inputs, returns, values)
    API[str(identifier)] = {'class_type': node_type, 'inputs': params, '_meta': {'title': title}}
    return node

def connect(start, output, end, input_name):
    socket = next(i for i, x in enumerate(end['inputs']) if x['name'] == input_name)
    identifier = len(LINKS) + 1
    typ = start['outputs'][output]['type']
    assert end['inputs'][socket]['type'] == typ
    LINKS.append([identifier, start['id'], output, end['id'], socket, typ])
    start['outputs'][output]['links'].append(identifier)
    end['inputs'][socket]['link'] = identifier
    API[str(end['id'])]['inputs'][input_name] = [str(start['id']), output]

source = add(1, 'LoadImage', 'SOURCE / IMAGE 1: original identity + wardrobe + background', [20, 90], [360, 570],
             outputs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')], values=['', 'image'])
API['1'] = {'class_type': 'LoadImage', 'inputs': {'image': 'REPLACE_WITH_UPLOADED_IMAGE.png'},
            '_meta': {'title': source['title']}}
mask = custom(2, 'WBGSBodyMask', '01 | SAM whole person minus hair', [440, 90], [370, 390])
qwen = custom(3, 'WBGSQwenSuit', '02 | Qwen Image 2.1 BF16: green lycra', [870, 90], [480, 820])
ref = custom(4, 'WBGS4KReference', '03 | 4K geometry reference: 2720 x 4080', [1410, 90], [410, 250])
partner_inputs = custom(5, 'WBGSPartnerInputs', '04a | Two ordered images for Comfy Partner', [1870, 90], [370, 170])
partner = add(12, 'OpenAIGPTImageNodeV2', '04b | Comfy Partner GPT 2.5 Sunburst (login required)', [2300, 90], [540, 850],
    inputs=[('image_1', 'IMAGE'), ('image_2', 'IMAGE')], outputs=[('IMAGE', 'IMAGE')],
    values=[module.GPT_PROMPT, 'gpt-image-2.5-sunburst', 1, 0, 'Custom', 2336, 3504, 'opaque', 'max'])
API['12'] = {'class_type': 'OpenAIGPTImageNodeV2', 'inputs': {
    'prompt': module.GPT_PROMPT, 'model': {'model': 'gpt-image-2.5-sunburst', 'size': 'Custom',
        'custom_width': 2336, 'custom_height': 3504, 'background': 'opaque', 'quality': 'max',
        'images': {}}, 'n': 1, 'seed': 0}, '_meta': {'title': partner['title']}}
gpt = custom(13, 'WBGSPartnerResult', '04c | Capture partner output', [2890, 90], [380, 170])
finish = custom(6, 'WBGSFinalize', '05 | Exact protected pixels + silhouette diagnostics', [3320, 90], [420, 240])
review = custom(7, 'WBGSReviewCompare', 'START / FINISH HERE | Run stage and compare progression', [3780, 90], [570, 1160])
connect(source, 0, mask, 'source')
connect(source, 1, mask, 'manual_protection')
connect(mask, 0, qwen, 'source_and_masks')
connect(qwen, 0, ref, 'qwen_result')
connect(ref, 0, partner_inputs, 'reference_package')
connect(partner_inputs, 0, partner, 'image_1')
connect(partner_inputs, 1, partner, 'image_2')
API['12']['inputs']['model']['images'] = {'image_1': ['5', 0], 'image_2': ['5', 1]}
connect(ref, 0, gpt, 'reference_package')
connect(partner, 0, gpt, 'gpt_image')
connect(gpt, 0, finish, 'gpt_result')
connect(mask, 0, review, 'source_and_masks')
connect(ref, 0, review, 'reference_package')
connect(finish, 0, review, 'final_package')
notes = [
    (8, 'READ FIRST | staged runs prevent accidental API charges', [20, -260], [790, 250],
     'Upload one 2:3 source image at left. The same original is GPT IMAGE 1; its generated green-suit version becomes IMAGE 2.\n\nAt the final compare node: 1 = SAM mask only. 2 = Qwen + 4K reference. 3 = full GPT edit. Mode 3 consumes Comfy Credits when the native Partner node runs. Leave both seeds fixed while reviewing.\n\nA painted mask on Load Image ADDS PROTECTION. It is not an edit mask. Check hair and the person selection before continuing. Face/skin are included in the green-suit stage.'),
    (9, 'REFERENCE RULES | no zoom, no reframing, no extra body parts', [870, -260], [950, 250],
     'Qwen preserves the original canvas and changes the person to fitted opaque green lycra. Native source dimensions are preserved after stitching. The 4K reference is 2720 x 4080, an exact 2:3 canvas. Original hair and outside-mask pixels are restored after the upscale.\n\nBoth model prompts are editable. The Qwen prompt-enhancer socket is optional and disconnected to retain precise role instructions. No reduced-precision variants or new local checkpoints are introduced.'),
    (10, 'API + OVERLAY LIMITS | read before approving an output', [1880, -260], [1040, 250],
     'GPT: gpt-image-2.5-sunburst, quality=max, native 2336 x 3504. This is the largest exact 2:3 size under the documented API limits; high-resolution output is experimental. Comfy Partner authentication uses your logged-in Comfy account and Comfy Credits; no OpenAI API key is required.\n\nPrompting cannot guarantee pixel-perfect internal pose or identity. Raw GPT, matte-constrained native, and matte-constrained 4K outputs are saved separately. Matte clipping does not fill missing limbs or fix internal pose. Loose original clothes are refitted to the target silhouette. Review the A/B overlay and the diagnostic report.'),
    (11, 'COMPARE | images, not new generations', [2980, -260], [570, 250],
     'Choose any two saved stages in the final node. Use wipe or 50% overlay. Turn on full-resolution sources for closer inspection, or open either image at full size.\n\nTo deliberately request another paid GPT image, increment new_variation. Identical image bytes and API settings reuse the local disk cache. Transport failures can still be billed; there are no automatic retries.\n\nDo not add other connected OUTPUT nodes upstream unless you intend to bypass the staged-run gate.'),
]
for identifier, title, pos, size, text in notes:
    add(identifier, 'Note', title, pos, size, values=[text])

graph = {'last_node_id': 13, 'last_link_id': len(LINKS), 'nodes': NODES, 'links': LINKS,
         'groups': [
             {'id': 1, 'title': 'SOURCE + MASK', 'bounding': [-5, 25, 845, 900], 'color': '#365b75', 'font_size': 24, 'flags': {}},
             {'id': 2, 'title': 'LOCAL QWEN + 4K REFERENCE', 'bounding': [850, 25, 995, 970], 'color': '#37634a', 'font_size': 24, 'flags': {}},
             {'id': 3, 'title': 'GPT EDIT + REGISTRATION CHECK', 'bounding': [1860, 25, 1085, 1080], 'color': '#65527b', 'font_size': 24, 'flags': {}},
             {'id': 4, 'title': 'RUN CONTROL + PROGRESSION', 'bounding': [2960, 25, 620, 1260], 'color': '#806438', 'font_size': 24, 'flags': {}}],
         'config': {}, 'extra': {'ds': {'scale': .42, 'offset': [110, 360]},
             'workbench': {'task_id': 'QGS1', 'version': module.VERSION, 'quality': 'hq',
                           'source_commit': '6b9a111740e08badc95970827e14e20efbfe3eca',
                           'validation': 'CPU/static tests only; GPU, API and frontend integration untested'}}, 'version': .4}

workflow = ROOT/'workflows/qwen/QGS1_Green_Suit_Overlay/Green_Suit_Qwen_to_GPT25_Overlay_v1_0.json'
workflow.parent.mkdir(parents=True, exist_ok=True)
workflow.write_text(json.dumps(graph, indent=2) + '\n')
automation = ROOT/'automation/QGS1/Green_Suit_Qwen_to_GPT25_Overlay_v1_0.api.json'
automation.parent.mkdir(parents=True, exist_ok=True)
automation.write_text(json.dumps(API, indent=2) + '\n')
catalog = {'schema_version': 1, 'tasks': [{'id': 'QGS1', 'title': 'Green suit template -> GPT identity overlay',
    'description': 'SAM person minus hair, Qwen BF16 green suit, SeedVR2 4K reference, native logged-in Comfy Partner GPT 2.5 edit and progression comparison. No OpenAI key.',
    'family': 'qwen', 'category': 'tasks', 'workspaces': ['qwen'],
    'requires': ['qwen', 'sam', 'seedvr2'], 'sources': [],
    'status': 'CPU/static validated; GPU and Partner login run required',
    'variants': [{'label': 'HQ + GPT Sunburst', 'file': str(workflow.relative_to(ROOT)), 'source': 'green-suit-add-on'}]}]}
(ROOT/'catalog/tasks.green-suit.json').write_text(json.dumps(catalog, indent=2) + '\n')
print('Wrote editable workflow, API graph and additive task catalog.')
