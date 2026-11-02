"""Add inspection and saving nodes without changing the native H3 calculation."""
from __future__ import annotations
import copy
import uuid
from .graphs import set_widget, validate_graph, Graph
from .recovery import SIMPLE_PROMPT


def native_preview(original, original_api):
    graph, api = copy.deepcopy(original), copy.deepcopy(original_api)
    by_id = {n['id']: n for n in graph['nodes']}
    required = ('decode', 'append', 'start', 'reference_conditioning', 'save', 'sample')
    if not all(key in by_id and key in api for key in required):
        raise ValueError('Pinned native graph changed; refusing guessed preview wiring')
    set_widget(by_id['reference_conditioning'], 'prompt', SIMPLE_PROMPT)
    api['reference_conditioning']['inputs']['prompt'] = SIMPLE_PROMPT
    by_id['sample']['title'] = 'LIVE SAMPLING PREVIEW: approximate first frame, not final color'
    ident = 900001
    if ident in by_id:
        raise ValueError('Preview node ID conflict')
    decoder = by_id['decode']
    append = by_id['append']
    link = next(r for r in graph['links'] if r[1] == 'decode' and r[2] == 0 and r[3] == 'append')
    lid = max([r[0] for r in graph['links']] + [0]) + 1
    next_id = lid + 1
    # Reuse the original append-input link, now sourced from the pass-through.
    decoder['outputs'][0]['links'].remove(link[0])
    decoder['outputs'][0]['links'].append(lid)
    link[1] = ident
    new = {'id': ident, 'type': 'H3SwapWindowPreview',
        'title': 'DECODED RESULT: first / middle / last; Cancel during the hold',
        'pos': [decoder['pos'][0], decoder['pos'][1]+450], 'size': [650, 480],
        'flags': {}, 'order': decoder.get('order', 0)+1, 'mode': 0,
        'inputs': [{'name': 'images', 'type': 'IMAGE', 'link': lid},
                   {'name': 'iteration_index', 'type': 'INT', 'link': next_id,
                    'widget': {'name': 'iteration_index'}}],
        'outputs': [{'name': 'unchanged_frames', 'type': 'IMAGE', 'links': [link[0]]}],
        'properties': {'Node name for S&R': 'H3SwapWindowPreview'},
        'widgets_values_named': {'iteration_index': 0, 'hold_seconds': 10, 'hold_first_window_only': True},
        'widgets_values': [0, 10, True]}
    graph['nodes'].append(new)
    graph['links'].extend([[lid, 'decode', 0, ident, 0, 'IMAGE'],
                           [next_id, 'start', 0, ident, 1, 'INT']])
    by_id['start']['outputs'][0].setdefault('links', []).append(next_id)
    api[str(ident)] = {'class_type': 'H3SwapWindowPreview', 'inputs': {
        'images': ['decode', 0], 'iteration_index': ['start', 0],
        'hold_seconds': 10, 'hold_first_window_only': True}, '_meta': {'title': new['title']}}
    api['append']['inputs']['frames'] = [str(ident), 0]
    # Use the existing native VIDEO input and native video.save_to path. Only
    # output metadata/adjacent JSON saving differs, not decoded pixel handling.
    save = by_id['save']
    if any(p.get('links') for p in save.get('outputs', [])):
        raise ValueError('Native save gained consumers; review before adapting')
    save.update(type='H3SwapSaveVideo', title='FINAL MP4 + editable workflow sidecar', outputs=[])
    save['properties']['Node name for S&R'] = 'H3SwapSaveVideo'
    save['widgets_values_named'] = {'filename_prefix': 'h3_character_swap/native_preview/akatz', 'crf': 16}
    save['widgets_values'] = list(save['widgets_values_named'].values())
    api['save'] = {'class_type': 'H3SwapSaveVideo', 'inputs': {
        'video': ['video', 0], **save['widgets_values_named']}, '_meta': {'title': save['title']}}
    graph['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'H3 native preview/recovery 2026-11-01'))
    graph['last_node_id'] = max([ident]+[n['id'] for n in graph['nodes'] if type(n['id']) is int])
    graph['last_link_id'] = next_id
    graph.setdefault('extra', {})['h3_preview_note'] = (
        'TAEH3 sampling preview is approximate and shows the first latent frame. On continuations '
        'that may be the accepted boundary. Use decoded preview for actual likeness/color. '
        'Default native run is 5.17 seconds, not a disk-backed long chain.')
    validate_graph(graph)
    return graph, api


def simple_chain():
    from .config import ROOT
    import json
    graph = json.loads((ROOT/'workflows/06_H3_LowMemory_Chain_Control.json').read_text())
    control = next(n for n in graph['nodes'] if n['type'] == 'H3SwapDiskChain')
    values = control['widgets_values_named']
    values.update(run_name='native_simple_test10', duration_seconds=10.0, prompt=SIMPLE_PROMPT,
                  review_mode='Review every chunk', max_retries=0)
    control['widgets_values'] = list(values.values())
    control['title'] = 'ONE REFERENCE + NATIVE SIMPLE PROMPT: 10-second test'
    for node in graph['nodes']:
        if node['type'] == 'Note':
            note = ('Select the SAME source, reference, seed and mask as your good native result. '
                    'No Ollama and no face/body prompt wrapper. Queue once; use /_swap/chain. '
                    'This still uses the disk controller, not the native in-memory loop. '
                    'After reviewing the first 10 seconds start a NEW run for 48 seconds. '
                    'Recovered workflows and the final MP4 keep the recorded settings.')
            node['widgets_values'] = [note]; node['widgets_values_named'] = {'text': note}
    graph['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'H3 simple disk chain 2026-11-01'))
    api = {str(control['id']): {'class_type': 'H3SwapDiskChain', 'inputs': dict(values),
                               '_meta': {'title': control['title']}}}
    validate_graph(graph)
    return graph, api


def native_recoverable(original, original_api):
    """Preserve the non-preview native recipe, but strip cache markers on export."""
    canvas,api=copy.deepcopy(original),copy.deepcopy(original_api)
    node=next(n for n in canvas['nodes'] if n['id']=='save')
    if any(p.get('links') for p in node.get('outputs',[])):
        raise ValueError('Native output has consumers; cannot substitute its saver blindly')
    node.update(type='H3SwapSaveVideo',title='FINAL MP4 + reopenable workflow JSON',outputs=[])
    node['properties']['Node name for S&R']='H3SwapSaveVideo'
    node['widgets_values_named']={'filename_prefix':'h3_character_swap/native_preview/akatz_native','crf':16}
    node['widgets_values']=list(node['widgets_values_named'].values())
    api['save']={'class_type':'H3SwapSaveVideo','inputs':{'video':['video',0],**node['widgets_values_named']}}
    validate_graph(canvas)
    return canvas,api
