"""QOVERLAY: select finished pixels from B over untouched original A.

This graph does not generate, crop, resize, VAE-re-encode, or align either image.
It deliberately reuses the existing Workbench mask and exact-copy compositor.
"""
from __future__ import annotations

VERSION = 'selective-ab-1'
WORKFLOW_PATH = 'workflows/qwen/QOVERLAY/Qwen21_Full_Resolution_Overlay_BF16.json'
TITLE = 'Selective overlay: original A + selected parts of edited B'
REQUIRED_NODES = [
    'LoadImage', 'Q21PhotoMask', 'WBMaskExpandFeather', 'WBComposite',
    'WBMaskPreview', 'MaskToImage', 'PreviewImage', 'SaveImage',
]
DESCRIPTION = (
    'Load the full original A and a finished edited image B. Select on A with SAM '
    'or a painted mask. White reveals B; black preserves A exactly. Independent '
    'expansion, finite outward feathering and inversion. Saves the complete '
    'opaque photo, not a transparent cutout. Both images must already be aligned '
    'and have identical dimensions. No Qwen generation or prompt enhancement in '
    'this graph. SAM defaults ON with target watermark; manual mode needs no SAM inference.'
)


def update_overlay_task(task):
    """Update QOVERLAY only, without changing any other task or model defaults."""
    if task.get('id') != 'QOVERLAY':
        return task
    task.update(
        title=TITLE,
        description=DESCRIPTION,
        requires=['sam'],
        required_nodes=list(REQUIRED_NODES),
        status='CPU pixel invariants tested; live SAM/ComfyUI validation pending',
        sources=[],
        variants=[{
            'label': 'A + finished B / SAM or manual selection / no generation',
            'file': WORKFLOW_PATH,
            'source': 'Workbench selective pixel compositing; not a generative template',
        }],
    )
    task.pop('optional_prompt_enhancement', None)
    return task


def verify_graph(graph):
    """Validate unique IDs, both ends of each connection, types, and ordering."""
    nodes = {n['id']: n for n in graph['nodes']}
    links = {r[0]: r for r in graph['links']}
    if len(nodes) != len(graph['nodes']) or len(links) != len(graph['links']):
        raise ValueError('Duplicate node or link ID in selective-overlay graph.')
    for lid, src, out, dst, inp, kind in links.values():
        a, b = nodes[src]['outputs'][out], nodes[dst]['inputs'][inp]
        if a['type'] != kind or b['type'] != kind:
            raise ValueError('Socket type mismatch.')
        if lid not in (a.get('links') or []) or b.get('link') != lid:
            raise ValueError('Inconsistent serialized link endpoints.')
        if nodes[src]['order'] >= nodes[dst]['order']:
            raise ValueError('Node order is not topological.')
    for node in nodes.values():
        for slot, port in enumerate(node.get('inputs', [])):
            lid = port.get('link')
            if lid is not None and (lid not in links or links[lid][3:5] != [node['id'], slot]):
                raise ValueError('Dangling input connection.')
        for slot, port in enumerate(node.get('outputs', [])):
            for lid in port.get('links') or []:
                if lid not in links or links[lid][1:3] != [node['id'], slot]:
                    raise ValueError('Dangling output connection.')
    if graph['last_node_id'] != max(nodes) or graph['last_link_id'] != max(links):
        raise ValueError('Incorrect last node/link ID.')


def make_selective_overlay():
    nodes, links = [], []

    def node(ident, kind, title, pos, size, inputs=(), outputs=(), widgets=()):
        value = {
            'id': ident, 'type': kind, 'title': title, 'pos': list(pos),
            'size': list(size), 'flags': {}, 'order': len(nodes), 'mode': 0,
            'inputs': [{'name': name, 'type': typ, 'link': None} for name, typ in inputs],
            'outputs': [{'name': name, 'type': typ, 'links': []} for name, typ in outputs],
            'properties': {'Node name for S&R': kind}, 'widgets_values': list(widgets),
        }
        nodes.append(value)
        return value

    def connect(a, out, b, inp, kind):
        lid = len(links) + 1
        links.append([lid, a['id'], out, b['id'], inp, kind])
        a['outputs'][out]['links'].append(lid)
        b['inputs'][inp]['link'] = lid

    a = node(1, 'LoadImage', '1 | ORIGINAL A - paint your reveal mask here',
             (0, 70), (380, 460), outputs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')],
             widgets=['', 'image'])
    b = node(2, 'LoadImage', '2 | EDITED B - full finished image, same size as A',
             (0, 600), (380, 460), outputs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')],
             widgets=['', 'image'])
    selection = node(3, 'Q21PhotoMask', '3 | Select on A: SAM or painted mask',
                     (450, 70), (390, 410),
                     inputs=[('source', 'IMAGE'), ('manual_mask', 'MASK')],
                     outputs=[('effective_edit_mask', 'MASK'), ('mask_report', 'STRING')],
                     widgets=[True, 'watermark', 0.5, 2, -1, 'full photo', 'ignore', 0])
    selection['inputs'][1]['shape'] = 7
    feather = node(4, 'WBMaskExpandFeather', '4 | Expand / feather / invert the reveal selection',
                   (450, 550), (390, 190), inputs=[('mask', 'MASK')],
                   outputs=[('expanded_feathered_mask', 'MASK')], widgets=[0, 12, False])
    composite = node(5, 'WBComposite', '5 | Keep A everywhere except selected pixels from B',
                     (910, 70), (420, 170),
                     inputs=[('original_a', 'IMAGE'), ('candidate_b', 'IMAGE'),
                             ('mask', 'MASK'), ('protect_mask', 'MASK')],
                     outputs=[('composited_image', 'IMAGE'), ('preservation_report', 'STRING')])
    composite['inputs'][3]['shape'] = 7
    selection_preview = node(6, 'WBMaskPreview', 'Preview reveal area on ORIGINAL A (not the result)',
                             (910, 330), (420, 90),
                             inputs=[('original_a', 'IMAGE'), ('mask', 'MASK')],
                             outputs=[('IMAGE', 'IMAGE')])
    mask_image = node(7, 'MaskToImage', 'Show the actual reveal mask',
                      (450, 800), (390, 80), inputs=[('mask', 'MASK')],
                      outputs=[('IMAGE', 'IMAGE')])
    save = node(8, 'SaveImage', '6 | FINAL PHOTO - original A plus selected parts of B',
                (1410, 70), (580, 570), inputs=[('images', 'IMAGE')],
                widgets=['Qwen21_Selective_Overlay/FINAL_A_plus_selected_B'])
    preview = node(9, 'PreviewImage', 'SELECTION ONLY - tinted area reveals B',
                   (910, 490), (420, 460), inputs=[('images', 'IMAGE')])
    mask_preview = node(10, 'PreviewImage', 'MASK ONLY - white = B / black = A / gray = blend',
                        (1410, 750), (580, 460), inputs=[('images', 'IMAGE')])
    note = node(11, 'Note', 'START HERE | Finished-image selective overlay, not AI editing',
                (0, 1130), (1300, 340), widgets=[
                    'A = untouched original. B = the complete edited version containing changes you like.\n'
                    'Both must have IDENTICAL pixel dimensions and already share the same framing. '
                    'This workflow will not silently crop, resize, shift or align either image.\n\n'
                    'SAM: leave auto_mask ON and name the object in A whose area should reveal B. '
                    'The retained default is watermark; change it to shirt, face, etc. '
                    'instance_index -1 selects all matches. SAM does not generate new pixels.\n'
                    'MANUAL: right-click ORIGINAL A, open MaskEditor, paint the parts that should '
                    'reveal B, save the mask back to the node, then set auto_mask OFF. '
                    'Do not paint on B. If SAM is unavailable, use this manual mode.\n\n'
                    'Node 4: expand_pixels grows the selection; feather_pixels adds an OUTWARD '
                    'fade. The fade is exactly zero beyond its finite edge. Invert selects the '
                    'opposite area. For a hard exact mask use expansion=0 and feather=0.\n'
                    'Node 5: an optional protect_mask input can hard-lock original pixels AFTER '
                    'feathering. It is not a second required mask.\n\n'
                    'Queue to save the full RGB photo in node 6. Other views are mask previews only. '
                    'There is no Qwen sampler, VAE, edit prompt, crop/stitch path or transparent export. '
                    'Changing the selection never regenerates B. This does not correct movement, '
                    'color changes or lighting differences already present in B.'
                ])
    connect(a, 0, selection, 0, 'IMAGE')
    connect(a, 1, selection, 1, 'MASK')
    connect(selection, 0, feather, 0, 'MASK')
    connect(a, 0, composite, 0, 'IMAGE')
    connect(b, 0, composite, 1, 'IMAGE')
    connect(feather, 0, composite, 2, 'MASK')
    connect(a, 0, selection_preview, 0, 'IMAGE')
    connect(feather, 0, selection_preview, 1, 'MASK')
    connect(feather, 0, mask_image, 0, 'MASK')
    connect(composite, 0, save, 0, 'IMAGE')
    connect(selection_preview, 0, preview, 0, 'IMAGE')
    connect(mask_image, 0, mask_preview, 0, 'IMAGE')
    graph = {
        'id': '6f2e7417-280e-4906-9ba0-de1149b88280', 'revision': 1,
        'last_node_id': 11, 'last_link_id': len(links), 'nodes': nodes, 'links': links,
        'groups': [], 'config': {}, 'version': 0.4,
        'extra': {
            'ds': {'scale': 0.58, 'offset': [80, 50]},
            'workbench': {'task': 'QOVERLAY', 'version': VERSION, 'gpu_tested': False},
            'workbench_source': {
                'overlay_version': VERSION,
                'changes': 'Finished image B is composited over original A through a selection made on A. No generation.'
            },
        },
    }
    verify_graph(graph)
    return graph
