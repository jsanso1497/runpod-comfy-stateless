"""Replace the unused padding path with the pinned native ComfyUI resizer.

No new nodes or model files are needed by these workflows on the original image.
Native rounding differs slightly from RMBG's floor-to-multiple rounding.
"""
from __future__ import annotations
import copy
import uuid


def native_resize(graph: dict) -> dict:
    """Return a copy with all resize image connections preserved, without COLORCODE."""
    result = copy.deepcopy(graph)
    removed = {n['id'] for n in result['nodes'] if n['type'] == 'AILab_ColorInput'}
    old_color_links = {link[0] for link in result['links'] if link[1] in removed}
    for node in result['nodes']:
        if node['type'] != 'AILab_ImageResize':
            continue
        image = next(p for p in node['inputs'] if p['name'] == 'image')
        if not image.get('link'):
            raise ValueError('Reference resizer needs an image connection.')
        for port in node.get('outputs', [])[1:]:
            if port.get('links'):
                raise ValueError('Unexpected used mask/size output on source resizer.')
        values = node.get('widgets_values_named', {})
        mp = values.get('megapixels', node['widgets_values'][2])
        if values.get('output_mode', 'stretch') != 'stretch':
            raise ValueError('This migration is for the unpadded developer preset only.')
        image_links = list(node['outputs'][0].get('links') or [])
        # The original image input is port 0, unchanged in the native node.
        node.update(type='ImageScaleToTotalPixels', size=[300,140],
                    inputs=[{'name':'image','type':'IMAGE','link':image['link']}],
                    outputs=[{'name':'IMAGE','type':'IMAGE','links':image_links}],
                    properties={'cnr_id':'comfy-core','ver':'0.43.0',
                                'Node name for S&R':'ImageScaleToTotalPixels'},
                    widgets_values=['lanczos',mp,32],
                    widgets_values_named={'upscale_method':'lanczos','megapixels':mp,'resolution_steps':32},
                    title=f'Reference resize: {mp:g} MP / 32-pixel grid (no padding)')
    result['nodes'] = [n for n in result['nodes'] if n['id'] not in removed]
    result['links'] = [l for l in result['links'] if l[0] not in old_color_links]
    # Color nodes from an earlier patch must not be connected to any other type.
    if any(l[1] in removed or l[3] in removed for l in result['links']):
        raise ValueError('Unexpected remaining color-node connection.')
    result.setdefault('extra',{})['h3_core_resize'] = {
        'version':'2026-11-06.1','padding':False,'extra_nodes_required':False,
        'resolution_note':'Native nearest multiple of 32; not pixel-identical to RMBG floor rounding.'}
    return result


def add_second_reference(graph: dict) -> dict:
    result=copy.deepcopy(graph)
    loads=[n for n in result['nodes'] if n['type']=='LoadImage']
    resizers=[n for n in result['nodes'] if n['type']=='ImageScaleToTotalPixels']
    refs=[n for n in result['nodes'] if n['type']=='MiniMaxH3ReferenceToVideo']
    if len(loads)!=1 or len(resizers)!=1 or len(refs)!=1:
        raise ValueError('Two-photo preset must start from a single-photo editing graph.')
    ref=refs[0];slot=next(i for i,p in enumerate(ref['inputs']) if p['name']=='ref_images.ref_image_1')
    if ref['inputs'][slot].get('link') is not None:
        raise ValueError('Second reference already connected.')
    node_id=max(n['id'] for n in result['nodes'])+1
    link_id=max(l[0] for l in result['links'])+1
    load=copy.deepcopy(loads[0]);resize=copy.deepcopy(resizers[0])
    load['id']=node_id;resize['id']=node_id+1
    load['title']='Upload Picture 2: supporting reference'
    load['pos']=[loads[0]['pos'][0],loads[0]['pos'][1]+420]
    load['widgets_values']=['upload-second-photo.png','image']
    load['widgets_values_named']={'image':'upload-second-photo.png','upload':'image'}
    for port in load['outputs']:port['links']=None
    load['outputs'][0]['links']=[link_id]
    resize['pos']=[resizers[0]['pos'][0],resizers[0]['pos'][1]+420]
    resize['inputs'][0]['link']=link_id;resize['outputs'][0]['links']=[link_id+1]
    ref['inputs'][slot]['link']=link_id+1
    result['nodes'].extend([load,resize])
    result['links'].extend([[link_id,node_id,0,node_id+1,0,'IMAGE'],
                            [link_id+1,node_id+1,0,ref['id'],slot,'IMAGE']])
    prompt=('Use <Picture 1> as the base photograph. <Picture 2> shows the same person; '
            'use it only to clarify facial identity. Preserve the pose, body proportions, '
            'clothing, lighting and background of <Picture 1>.')
    ref['widgets_values'][0]=prompt;ref['widgets_values_named']['prompt']=prompt
    loads[0]['title']='Upload Picture 1: base photograph'
    return result


def finalize(graph: dict, name: str) -> dict:
    result=copy.deepcopy(graph)
    result['id']=str(uuid.uuid5(uuid.NAMESPACE_URL,'h3-still-core-resize-2026-11-06.1/'+name))
    result['last_node_id']=max(n['id'] for n in result['nodes'])
    result['last_link_id']=max(l[0] for l in result['links'])
    # Recompute a topological order for duplicated reference inputs.
    incoming={n['id']:set() for n in result['nodes']}
    for _,a,_,b,_,_ in result['links']:incoming[b].add(a)
    done=set();order=0
    while len(done)<len(incoming):
        ready=[n for n in result['nodes'] if n['id'] not in done and incoming[n['id']]<=done]
        if not ready:raise ValueError('Workflow contains a cycle.')
        for n in ready:n['order']=order;order+=1;done.add(n['id'])
    return result
