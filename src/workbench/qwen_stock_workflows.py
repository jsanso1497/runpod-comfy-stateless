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


def make_overlay(stock):
    graph = copy.deepcopy(stock)
    edit = native_node(graph)
    target_slot = next(i for i, socket in enumerate(edit['inputs']) if socket.get('label') == 'image_1')
    source_link = next(row for row in graph['links'] if row[0] == edit['inputs'][target_slot]['link'])
    source = next(n for n in graph['nodes'] if n['id'] == source_link[1])
    if source['type'] != 'LoadImage':
        raise ValueError('Expected stock source LoadImage.')
    # Only the outer shell changes. No SaveImage or comparator may cause an
    # unmasked generation/save while mask review is OFF.
    graph['nodes'] = [source, edit]
    graph['links'] = []
    graph['groups'] = []
    for node in graph['nodes']:
        for inp in node.get('inputs', []):
            inp['link'] = None
        for out in node.get('outputs', []):
            out['links'] = []
    source['pos'], source['size'] = [0, 0], [320, 360]
    source['title'] = '1 | ORIGINAL sRGB photo - paint on MASK layer, not RGB'
    edit['pos'] = [1270, 0]
    edit['title'] = '5 | Stock Qwen edit: BF16 / original prompt enhancer'
    # IDs are above all upstream IDs, including nested subgraph node IDs.
    mask_id, protect_id, prep_id, save_id, note_id = range(1001, 1006)
    mask = _node(mask_id, 'Q21PhotoMask', '2 | Edit mask: OFF manual / ON SAM', (380, 0),
                 [('source', 'IMAGE'), ('manual_mask', 'MASK')],
                 [('effective_edit_mask', 'MASK'), ('mask_report', 'STRING')],
                 [False, 'shirt', 0.5, 2, -1, 'full photo', 'ignore', 0])
    protect = _node(protect_id, 'WBOverlayProtection', '3 | Second SAM: protect face / described element', (380, 400),
                    [('source', 'IMAGE')], [('protected_on_original', 'MASK')], [True, 'face', 0.5, -1])
    prep = _node(prep_id, 'WBOverlayPrepare', '4 | Crop only; preserve full-resolution original', (820, 0),
                 [('source', 'IMAGE'), ('edit_mask', 'MASK'), ('protection', 'MASK'), ('mask_report', 'STRING')],
                 [('full_resolution_package', 'WB_OVERLAY_CROP'), ('working_crop', 'IMAGE')], [2048, 128, 8])
    save = _node(save_id, 'WBOverlayExport', '6 | OFF: review masks / ON: export overlay + composite', (1760, 0),
                 [('package', 'WB_OVERLAY_CROP'), ('generated_crop', 'IMAGE')], [],
                 [False, 'Qwen21_Overlay/Edit', 12])
    save['size'] = [620, 740]
    note = _node(note_id, 'Note', 'Read before first Run', (820, 470), [], [], [
        '1. Load your ORIGINAL sRGB image. For a manual edit mask: right-click Load Image, '
        'Open in Mask Editor, paint the MASK layer and save. Or enable auto_mask for SAM.\n'
        '2. The separate protection node detects face (or your description) on ORIGINAL pixels. '
        'All matching instances are protected at index -1. Missing detection stops the run. '
        'Set protection enabled OFF only when no protected region is needed.\n'
        '3. First Run reviews masks: green = editable, red = protected. Check coverage. '
        'Then set run_edit ON. The crop is at most 2048 squared, not the full source canvas.\n'
        '4. Keep native resolution=0 and custom_size OFF so the crop returns at the correct size. '
        'Describe the edit in the native prompt. Additional references can connect to image_2 onward.\n'
        '5. Exports: *_overlay.png (straight RGBA, original dimensions), *_composite.png and *_mask.png. '
        'Place the overlay at x=0,y=0 over the original, Normal blend, opacity 100%. '
        'Only pixel changes survive in alpha. Face protection wins over feathering.\n'
        'PNG files are 8-bit sRGB. CPU geometry tests do not certify SAM accuracy or GPU/model quality.'])
    note['size'] = [840, 370]
    graph['nodes'].extend([mask, protect, prep, save, note])
    def link(a, slot, b, target, kind):
        ident = 1100 + len(graph['links'])
        graph['links'].append([ident, a['id'], slot, b['id'], target, kind])
        a['outputs'][slot]['links'].append(ident)
        b['inputs'][target]['link'] = ident
    link(source, 0, mask, 0, 'IMAGE'); link(source, 1, mask, 1, 'MASK')
    link(source, 0, protect, 0, 'IMAGE'); link(source, 0, prep, 0, 'IMAGE')
    link(mask, 0, prep, 1, 'MASK'); link(mask, 1, prep, 3, 'STRING')
    link(protect, 0, prep, 2, 'MASK'); link(prep, 1, edit, target_slot, 'IMAGE')
    link(prep, 0, save, 0, 'WB_OVERLAY_CROP'); link(edit, 0, save, 1, 'IMAGE')
    for order, node in enumerate(graph['nodes']):
        node['order'] = order
    graph['last_node_id'] = note_id
    graph['last_link_id'] = graph['links'][-1][0]
    graph['id'] = 'db8e4146-91d4-4c81-804a-3096050c2110'
    graph['extra']['workbench_source']['changes'] += ' Added outer mask/crop/protection/overlay shell.'
    graph['extra']['ds'] = {'scale': 0.7, 'offset': [80, 80]}
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
