"""Build two workflows from an immutable upstream template, not a hand-made clone.

Only the main diffusion/text-encoder filenames change in the stock graph.
Prompt enhancer model, system prompt, sampling, cache, topology and defaults are
retained verbatim. The overlay graph wraps that same unmodified subgraph.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from .qwen_stock_catalog import STOCK_PATH, OVERLAY_PATH, PE_FILE

UPSTREAM_REV = '3e89967349dd95fa68ce3b36966f41d642af8ea7'
UPSTREAM_URL = ('https://raw.githubusercontent.com/Comfy-Org/workflow_templates/' +
                UPSTREAM_REV + '/templates/image_qwen_image_2_1_image_edit.json')
REPLACEMENTS = {'qwen_image_2.1_int8_convrot.safetensors': 'qwen_image_2.1_bf16.safetensors',
                'qwen3vl_8b_int8_convrot.safetensors': 'qwen3vl_8b_bf16.safetensors'}


def replace_models(value):
    if isinstance(value, dict):
        return {key: replace_models(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_models(item) for item in value]
    if isinstance(value, str):
        if value in REPLACEMENTS:
            return REPLACEMENTS[value]
        if value.startswith('https://'):
            for before, after in REPLACEMENTS.items():
                if value.endswith('/' + before):
                    return value[:-len(before)] + after
    return value


def native_node(graph):
    matches = [n for n in graph['nodes'] if
               'unet_name' in n.get('widgets_values_named', {}) and
               n.get('widgets_values_named', {}).get('clip_name_1') == PE_FILE]
    if len(matches) != 1:
        raise ValueError('Pinned stock template contract changed: expected one native edit subgraph.')
    return matches[0]


def verify_links(graph):
    nodes = {n['id']: n for n in graph['nodes']}
    links = {row[0]: row for row in graph.get('links', [])}
    if len(nodes) != len(graph['nodes']) or len(links) != len(graph.get('links', [])):
        raise ValueError('Duplicate graph node/link IDs.')
    for lid, src, out, dst, inp, kind in links.values():
        if lid not in (nodes[src]['outputs'][out].get('links') or []):
            raise ValueError('Output socket/link disagreement.')
        if nodes[dst]['inputs'][inp].get('link') != lid:
            raise ValueError('Input socket/link disagreement.')
    for node in nodes.values():
        for inp in node.get('inputs', []):
            if inp.get('link') is not None and inp['link'] not in links:
                raise ValueError('Dangling input link.')
        for out in node.get('outputs', []):
            if any(lid not in links for lid in (out.get('links') or [])):
                raise ValueError('Dangling output link.')


def make_stock(upstream):
    graph = replace_models(copy.deepcopy(upstream))
    node = native_node(graph)
    defaults = node['widgets_values_named']
    expected = {'cfg': 1, 'steps': 25, 'scheduler': 'euler', 'scheduler_1': 'simple',
                'resolution': 0, 'switch': False, 'switch_1': False,
                'device': 'auto', 'dtype': 'default', 'clip_name_1': PE_FILE}
    if any(defaults.get(key) != val for key, val in expected.items()):
        raise ValueError('Pinned upstream defaults no longer match the inspected native template.')
    graph.setdefault('extra', {})['workbench_source'] = {
        'url': UPSTREAM_URL, 'revision': UPSTREAM_REV,
        'changes': 'Only main diffusion and base text encoder changed to BF16.'}
    verify_links(graph)
    return graph


def _node(ident, kind, title, pos, inputs, outputs, widgets):
    return {'id': ident, 'type': kind, 'title': title, 'pos': list(pos), 'size': [360, 320],
            'flags': {}, 'order': 0, 'mode': 0,
            'inputs': [{'name': name, 'type': typ, 'link': None} for name, typ in inputs],
            'outputs': [{'name': name, 'type': typ, 'links': []} for name, typ in outputs],
            'properties': {'Node name for S&R': kind}, 'widgets_values': widgets}


# These defaults apply only to QOVERLAY, not to other Qwen workflows.
OVERLAY_VERSION = 'watermark-full-photo-2'
WATERMARK_PROMPT = (
    'Remove the watermark from <image1>. Remove all overlaid watermark lettering, '
    'logos, and translucent purchase-message text. Reconstruct the underlying '
    'photographic detail so it joins naturally with the surrounding photograph. '
    'Keep the original subject, face, clothing, pose, framing, camera angle, '
    'lighting, color, and background unchanged except where the watermark covers '
    'them. Return the complete opaque photograph, not an isolated watermark, '
    'cutout, transparent layer, mask, or comparison. Do not add new text or logos.'
)


def make_overlay(stock):
    """One source crop feeds both Qwen and its full-resolution RGB stitcher.

    Protection = 1 - GrowMask(edit_mask, 16). The final paste mask is
    clamp(edit_mask - protection, 0, 1), so protection is part of the actual
    output path, not an unused branch. For binary masks this retains the exact
    selected edit area. Soft-mask boundaries are conservatively restricted.
    The only output node is the existing full-photo review/save node.
    """
    graph = copy.deepcopy(stock)
    edit = native_node(graph)
    target_slot = next(i for i, socket in enumerate(edit['inputs'])
                       if socket.get('label') == 'image_1')
    source_link = next(row for row in graph['links']
                       if row[0] == edit['inputs'][target_slot]['link'])
    source = next(n for n in graph['nodes'] if n['id'] == source_link[1])
    if source['type'] != 'LoadImage':
        raise ValueError('Expected stock source LoadImage.')

    # Named and positional values must agree in both frontend serialization paths.
    values = edit['widgets_values_named']
    if len(edit['widgets_values']) < 2 or edit['widgets_values'][1] != values['prompt']:
        raise ValueError('Pinned native prompt widget layout changed.')
    values['prompt'] = WATERMARK_PROMPT
    edit['widgets_values'][1] = WATERMARK_PROMPT
    graph['nodes'] = [source, edit]
    graph['links'] = []
    graph['groups'] = []
    for node in graph['nodes']:
        for inp in node.get('inputs', []):
            inp['link'] = None
        for out in node.get('outputs', []):
            out['links'] = []
    source['pos'], source['size'] = [0, 0], [340, 430]
    source['title'] = '1 | FULL original photo - not an earlier overlay or mask'
    edit['pos'] = [1630, 0]
    edit['title'] = '6 | Remove watermark - native Qwen BF16 + stock prompt enhancer'

    mask = _node(1001, 'Q21PhotoMask', '2 | SAM ON - target: watermark', (400, 0),
                 [('source', 'IMAGE'), ('manual_mask', 'MASK')],
                 [('effective_edit_mask', 'MASK'), ('mask_report', 'STRING')],
                 [True, 'watermark', 0.5, 2, -1, 'full photo', 'ignore', 0])
    mask['size'] = [360, 410]
    grow = _node(1002, 'GrowMask', '3 | Grow edit mask by 16 source pixels', (400, 470),
                 [('mask', 'MASK')], [('MASK', 'MASK')], [16, False])
    grow['size'] = [360, 120]
    protect = _node(1003, 'InvertMask', '4 | Invert grown mask = protected outside', (400, 650),
                    [('mask', 'MASK')], [('MASK', 'MASK')], [])
    protect['size'] = [360, 85]
    gate = _node(1004, 'MaskComposite', '5 | Final edit mask minus protected area', (810, 0),
                 [('destination', 'MASK'), ('source', 'MASK')], [('MASK', 'MASK')],
                 [0, 0, 'subtract'])
    gate['size'] = [350, 190]
    crop = _node(1005, 'Q21PhotoCrop', '5B | ONE crop for generation and final stitch', (1210, 0),
                 [('source', 'IMAGE'), ('mask', 'MASK'), ('mask_report', 'STRING')],
                 [('crop_data', 'Q21_PHOTO_CROP'), ('working_crop', 'IMAGE')],
                 [2048, 3072, 128, False])
    crop['size'] = [360, 280]
    stitch = _node(1006, 'Q21PhotoStitch', '7 | Paste edited crop onto FULL original photo', (2110, 0),
                   [('crop_data', 'Q21_PHOTO_CROP'), ('generated_crop', 'IMAGE'),
                    ('reference_report', 'STRING')], [('edited_result', 'Q21_PHOTO_RESULT')],
                   [12, 0.0])
    stitch['size'] = [370, 210]
    save = _node(1007, 'Q21PhotoReviewSave', '8 | EDIT ON - FINAL FULL PHOTO (only output)', (2540, 0),
                 [('crop_data', 'Q21_PHOTO_CROP'), ('edited_result', 'Q21_PHOTO_RESULT')], [],
                 [True, 'Qwen21_Watermark_Removal/FINAL_Full_Photo'])
    save['inputs'][1]['shape'] = 7
    save['size'] = [590, 760]
    audit = _node(1008, 'PrimitiveStringMultiline', 'Stitch audit context (not the edit prompt)',
                  (2110, 300), [], [('STRING', 'STRING')], [json.dumps({
                      'workflow': 'QOVERLAY', 'revision': OVERLAY_VERSION,
                      'generation': 'native ComfyUI Qwen Image 2.1 edit',
                      'protection': 'invert(grow(edit_mask, 16)); subtract before cropping',
                      'output': 'original-resolution RGB photo composite'})])
    audit['size'] = [370, 200]
    note = _node(1009, 'Note', 'Watermark removal / full-photo output', (810, 460), [], [], [
        'SAM auto_mask and run_edit are ON. Upload the FULL original photograph, then Queue.\n'
        'SAM prompt = watermark; index -1 selects all matching instances. Detection can miss '
        'letters or select too much. Set run_edit OFF for mask review. If needed, paint and save '
        'a manual mask on the source and set auto_mask OFF.\n'
        'Protection is the edit mask grown by 16 source pixels, then inverted. It is subtracted '
        'from the edit selection BEFORE the single crop used by both Qwen and the final stitch. '
        'The paste region stays inside the selected edit mask; growth does not enlarge it.\n'
        'Keep native resolution=0 / custom_size OFF. Image models remain BF16; the stock '
        'prompt-enhancer model and settings are unchanged. refine_prompt remains optional.\n'
        'Only node 8 is an output. It saves/previews the complete RGB photo at the original '
        'dimensions, plus a diagnostic .mask.png and verification .json on disk. There is NO '
        'transparent-overlay export or output of the generated crop.\n'
        'Pixels outside the final mask are copied from the original. PNG is 8-bit sRGB. '
        'Occluded details are reconstructed, not recovered exactly; SAM and GPU quality need '
        'a real-image test.'])
    note['size'] = [1180, 370]
    graph['nodes'] = [source, mask, grow, protect, gate, crop, edit, audit, stitch, save, note]

    def link(a, slot, b, target, kind):
        ident = 1100 + len(graph['links'])
        graph['links'].append([ident, a['id'], slot, b['id'], target, kind])
        a['outputs'][slot]['links'].append(ident)
        b['inputs'][target]['link'] = ident

    link(source, 0, mask, 0, 'IMAGE')
    link(source, 1, mask, 1, 'MASK')
    link(mask, 0, grow, 0, 'MASK')
    link(grow, 0, protect, 0, 'MASK')
    link(mask, 0, gate, 0, 'MASK')
    link(protect, 0, gate, 1, 'MASK')
    link(source, 0, crop, 0, 'IMAGE')
    link(gate, 0, crop, 1, 'MASK')
    link(mask, 1, crop, 2, 'STRING')
    link(crop, 1, edit, target_slot, 'IMAGE')
    link(crop, 0, stitch, 0, 'Q21_PHOTO_CROP')
    link(edit, 0, stitch, 1, 'IMAGE')
    link(audit, 0, stitch, 2, 'STRING')
    link(crop, 0, save, 0, 'Q21_PHOTO_CROP')
    link(stitch, 0, save, 1, 'Q21_PHOTO_RESULT')
    for order, node in enumerate(graph['nodes']):
        node['order'] = order
    graph['last_node_id'] = 1009
    graph['last_link_id'] = graph['links'][-1][0]
    graph['id'] = '2c870975-65ad-4191-a8b3-990f3959efb0'
    graph['revision'] = 0
    graph['extra']['workbench_source'].update(
        overlay_version=OVERLAY_VERSION,
        changes=('QOVERLAY uses one crop and an RGB full-photo stitch/save path. '
                 'SAM ON, target watermark, run_edit ON, watermark-removal prompt. '
                 '16px grown/inverted mask gates the final selection. '
                 'No second SAM and no transparent-overlay exporter. '
                 'Native model/enhancer/sampler settings otherwise unchanged.'))
    graph['extra']['ds'] = {'scale': 0.55, 'offset': [80, 80]}
    verify_links(graph)
    if graph['definitions'] != stock['definitions']:
        raise AssertionError('The native edit subgraph or enhancer was altered.')
    return graph

def build(root, offline=False):
    root = Path(root)
    archive = root / 'vendor/qwen-stock' / (UPSTREAM_REV + '.json')
    if archive.is_file():
        raw = archive.read_bytes()
    elif offline:
        raise FileNotFoundError('Pinned upstream snapshot missing. Run once with network access.')
    else:
        for attempt in range(3):
            try:
                with urlopen(Request(UPSTREAM_URL, headers={'User-Agent': 'Workbench-Qwen-Stock/1.0'}), timeout=60) as response:
                    raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ValueError('Upstream template exceeds size limit.')
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(raw)
    upstream = json.loads(raw)
    stock = make_stock(upstream)
    overlay = make_overlay(stock)
    for relative, graph in [(STOCK_PATH, stock), (OVERLAY_PATH, overlay)]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    report = {'upstream_revision': UPSTREAM_REV, 'upstream_sha256': hashlib.sha256(raw).hexdigest(),
              'workflows': [STOCK_PATH, OVERLAY_PATH], 'gpu_tested': False}
    (archive.parent / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.offline), indent=2))
