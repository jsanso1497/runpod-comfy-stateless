"""Build QKEEP1 around the existing pinned native Qwen edit template.

Image weights, encoder, VAE, optional stock prompt enhancer, and sampling
settings stay unchanged. An internal text-preview pass-through is removed
so mask review has only one output sink and cannot schedule prompt inference.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

from .qwen_people_catalog import WORKFLOW
from .qwen_stock_catalog import STOCK_PATH
from .qwen_stock_workflows import native_node, verify_links

SUBGRAPH_ID = 'fb62f19e-b1df-4ec8-aa4a-8e1e3d6d518f'
GRAPH_ID = '897d0c45-9058-4e78-a642-c8d82eb63817'


def _node(ident, kind, title, pos, size, inputs, outputs, widgets):
    return {'id': ident, 'type': kind, 'title': title, 'pos': list(pos),
            'size': list(size), 'flags': {}, 'mode': 0, 'order': 0,
            'inputs': [{'name': name, 'type': typ, 'link': None} for name,typ in inputs],
            'outputs': [{'name': name, 'type': typ, 'links': []} for name,typ in outputs],
            'properties': {'Node name for S&R': kind}, 'widgets_values': widgets}


def _remove_text_preview(graph, edit):
    definitions=graph.get('definitions', {}).get('subgraphs', [])
    subgraph=next((s for s in definitions if s['id']==edit['type']), None)
    if subgraph is None:
        raise ValueError('Native edit subgraph definition is missing.')
    previews=[n for n in subgraph['nodes'] if n['type']=='PreviewAny']
    for node in previews:
        incoming=[r for r in subgraph['links'] if r['target_id']==node['id']]
        outgoing=[r for r in subgraph['links'] if r['origin_id']==node['id']]
        if len(incoming)!=1 or any(r['type']!='STRING' for r in incoming+outgoing):
            raise ValueError('Unexpected native preview node. Review upstream before bypassing it.')
        link=incoming[0]
        # This PreviewAny only passes an already-STRING prompt into the encoder.
        # Forward all outgoing links from the prompt switch and remove the UI sink.
        for row in outgoing:
            row['origin_id']=link['origin_id'];row['origin_slot']=link['origin_slot']
        subgraph['links']=[r for r in subgraph['links'] if r['id']!=link['id']]
        subgraph['nodes']=[n for n in subgraph['nodes'] if n['id']!=node['id']]
    # Rebuild inner regular-node ports. The -10/-20 pseudo nodes have separate
    # subgraph interface records; no interface links have changed.
    for node in subgraph['nodes']:
        for slot,port in enumerate(node.get('inputs', [])):
            links=[r['id'] for r in subgraph['links'] if r['target_id']==node['id'] and r['target_slot']==slot]
            if len(links)>1:raise ValueError('Duplicate subgraph input link.')
            port['link']=links[0] if links else None
        for slot,port in enumerate(node.get('outputs', [])):
            port['links']=[r['id'] for r in subgraph['links'] if r['origin_id']==node['id'] and r['origin_slot']==slot]
    # New subgraph ID avoids mutating/colliding with QNATIVE in the frontend.
    old=subgraph['id'];subgraph['id']=SUBGRAPH_ID
    subgraph['name']='Qwen Image Edit: protected keep-one repair'
    edit['type']=SUBGRAPH_ID
    if edit.get('properties', {}).get('Node name for S&R')==old:
        edit['properties']['Node name for S&R']=SUBGRAPH_ID
    return len(previews)


def make_people(stock):
    graph=copy.deepcopy(stock)
    edit=native_node(graph)
    source_candidates=[n for n in graph['nodes'] if n['type']=='LoadImage']
    if not source_candidates:
        raise ValueError('Native template has no source LoadImage.')
    first=next(i for i,p in enumerate(edit['inputs']) if p.get('label')=='image_1')
    second=next(i for i,p in enumerate(edit['inputs']) if p.get('label')=='image_2')
    prompt_slot=next(i for i,p in enumerate(edit['inputs']) if p['name']=='prompt')
    link=next(r for r in graph['links'] if r[0]==edit['inputs'][first]['link'])
    source=next(n for n in source_candidates if n['id']==link[1])
    graph['nodes']=[source,edit];graph['links']=[];graph['groups']=[]
    for node in graph['nodes']:
        for port in node.get('inputs',[]):port['link']=None
        for port in node.get('outputs',[]):port['links']=[]
    source['title']='1 | ORIGINAL photo - paint MASK to PROTECT pixels'
    source['pos']=[0,0];source['size']=[340,410]
    source['widgets_values']=['','image']
    if 'widgets_values_named' in source:
        source['widgets_values_named']['image']=''
    edit['title']='4 | Qwen repair: BF16 / stock enhancer / same crop size'
    edit['pos']=[1310,0];edit['size']=[460,690]
    # Prompt value is overridden by a connected STRING, but keep both saved
    # frontend representations coherent (no old clothing/watermark prompt).
    text='Use the connected keep-one-person prompt from node 3.'
    edit['widgets_values_named']['prompt']=text
    edit['widgets_values'][1]=text
    removed=_remove_text_preview(graph,edit)

    detect=_node(1201,'WBPeopleDetect','2 | SAM: detect and number up to 5 people',
                 [390,0],[370,240],[('source','IMAGE')],
                 [('detected_people','WB_PEOPLE'),('person_count','INT')],[0.5,2,0])
    # Default instruction copied from the runtime without importing torch at
    # Docker/Pages catalog-build time. Verify synchronization in CPU tests.
    instruction=(
        'Remove all people from <image1> except the single person shown in <image2>. '
        'Use <image2> only to identify which person to retain, not as a replacement '
        'scene or a second copy of the person. Keep the retained person in their '
        'original position with the same face, expression, clothing and pose. '
        'Fill the removed people\'s locations with plausible continuous background '
        'and reconstruct any newly exposed portions of the retained person that '
        'were hidden by them. Repair local edges, contact shadows and nearby scene '
        'detail inside the editable area. Match the original lighting, perspective, '
        'texture, color and focus. Do not add people, duplicate limbs, text, labels '
        'or outlines. Return a complete opaque photographic crop, not a cutout, '
        'transparent hole, collage, mask or isolated person.'
    )
    prepare=_node(1202,'WBPeoplePrepare','3 | Keep person 1-5 / ORIGINAL mask = protection',
                  [810,0],[450,670],
                  [('detections','WB_PEOPLE'),('manual_protection','MASK'),('extra_repair_mask','MASK')],
                  [('protected_edit_package','WB_PEOPLE_EDIT'),('image1_working_scene','IMAGE'),
                   ('image2_person_to_keep','IMAGE'),('edit_prompt','STRING')],
                  [1,2048,128,24,2,0,instruction])
    prepare['inputs'][1]['shape']=7;prepare['inputs'][2]['shape']=7
    export=_node(1203,'WBPeopleExport','5 | Preview first, then EDIT: full photo + RGBA overlay',
                 [1840,0],[640,790],[('package','WB_PEOPLE_EDIT'),('generated_crop','IMAGE')],[],
                 [False,'Qwen21_Keep_One/Result',12,False])
    export['inputs'][1]['shape']=7
    note=_node(1204,'Note','QKEEP1: read this once',[390,820],[1360,350],[],[],[
        '1. Load the COMPLETE original sRGB photograph. Optional: paint a MASK on this loader '
        'to protect those original pixels. This is NOT the removal mask.\n'
        '2. First Queue with run_edit OFF: preview numbered people. SAM runs automatically. '
        'IDs are left to right by box center; choose keep_person 1-5 in node 3. '
        'expected_people=0 is automatic; set 1-5 to catch a missing/wrong detection count. '
        'A sixth detected person stops rather than silently ignoring anyone.\n'
        '3. Green is the person kept, red is removal/repair, blue is painted protection. '
        'Protected pixels ALWAYS win. Painting over a removed person leaves those fragments intact.\n'
        '4. Turn run_edit ON to repair. Original visible keeper pixels are locked; occluded '
        'parts/background may be reconstructed only in the editable region. repair_grow_pixels '
        'adds edge context. For distant shadows/reflections, connect a separate full-size '
        'extra_repair_mask to node 3. Never use the protected source mask for this.\n'
        '5. Full-photo RGB composite is previewed first. A separate original-size straight RGBA '
        'overlay, edit mask and audit are saved under Qwen21_Keep_One. show_overlay_preview '
        'adds the overlay to the preview; it is saved either way.\n'
        'Keep the native editor at resolution 0 and custom_size OFF. No new models are needed. '
        'The new WBPeople nodes MUST be installed before this JSON can run. PNG export is '
        '8-bit sRGB. SAM selection and inferred detail require a real-image review.'
    ])
    graph['nodes']=[source,detect,prepare,edit,export,note]
    def connect(a,slot,b,dest,typ):
        ident=1300+len(graph['links'])
        graph['links'].append([ident,a['id'],slot,b['id'],dest,typ])
        a['outputs'][slot]['links'].append(ident);b['inputs'][dest]['link']=ident
    connect(source,0,detect,0,'IMAGE')
    connect(detect,0,prepare,0,'WB_PEOPLE')
    connect(source,1,prepare,1,'MASK')
    connect(prepare,1,edit,first,'IMAGE')
    connect(prepare,2,edit,second,'IMAGE')
    connect(prepare,3,edit,prompt_slot,'STRING')
    connect(prepare,0,export,0,'WB_PEOPLE_EDIT')
    connect(edit,0,export,1,'IMAGE')
    for order,node in enumerate(graph['nodes']):node['order']=order
    graph['id']=GRAPH_ID;graph['revision']=0
    graph['last_node_id']=1204;graph['last_link_id']=graph['links'][-1][0]
    graph['config']={}
    graph['extra']={'ds':{'scale':0.58,'offset':[80,80]}}
    graph['extra']['qkeep1']={'version':'1.0.0','source':'existing pinned QNATIVE BF16 template',
        'changes':'SAM keep-one selection; painted protection; one bounded edit crop; RGB composite and RGBA edit layer',
        'text_preview_sinks_removed':removed,'stock_enhancer_settings_changed':False,'gpu_tested':False}
    verify_links(graph)
    return graph


def build(root):
    root=Path(root)
    stock=json.loads((root/STOCK_PATH).read_text(encoding='utf-8'))
    graph=make_people(stock)
    path=root/WORKFLOW;path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(graph,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
    path.write_bytes(data)
    return {'file':WORKFLOW,'sha256':hashlib.sha256(data).hexdigest(),'gpu_tested':False}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    print(json.dumps(build(parser.parse_args().root),indent=2))
