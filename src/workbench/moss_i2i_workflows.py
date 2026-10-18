"""Standard latent img2img using the pinned MOSS T2I model and native nodes.

This is a Workbench adaptation, not an official instruction-edit checkpoint.
The source goes through VAEEncode. It is NOT a vision reference to TextGenerate.
No new weights or custom ComfyUI nodes are needed.
"""
from __future__ import annotations
import copy
import hashlib
import json

from .moss_catalog import I2I_TASK, I2I_WORKFLOW

VERSION = 'moss-native-img2img-1.0.0'
DENOISE = 0.35
MEGAPIXELS = 1.0
RESOLUTION_STEPS = 16
PROMPT = ('A realistic photograph with natural lighting, balanced exposure, lifelike colors, '
          'coherent depth, and fine photographic texture.')
NOTE = (
    'MOSS IMAGE-TO-IMAGE | Workbench adaptation, not an official edit model.\n\n'
    '1. Upload one complete image in SOURCE IMAGE.\n'
    '2. Replace the positive prompt with a description of the complete desired image, '
    'including its subject, setting and style. This is not Qwen-style instruction editing.\n'
    '3. Denoise starts at 0.35. Lower values tend to stay closer to the source; higher '
    'values permit larger changes. 1.0 is effectively text-to-image.\n'
    '4. Processing size starts at 1 MP, following the input aspect ratio and rounding '
    'dimensions to multiples of 16. This resizes the whole image; output is NOT '
    'automatically restored to the original full resolution.\n'
    '5. The stock prompt upsampler is ON. It receives TEXT ONLY, not your photo. '
    'Switch it OFF to use your exact caption and reduce invented scene details.\n'
    '6. Uses the same MOSS SFT, text encoder and VAE as QMOSS. No new model downloads.\n'
    '7. Save Image writes the complete generated image under MOSS_Image/I2I. '
    'No masks, transparent layers or protected-pixel compositing are added. '
    'Faces, geometry and text may change, even with low denoise.\n\n'
    'Keep-one-person editing and watermark repair remain separate Qwen tasks. '
    'Static/CPU tests do not certify model quality or live GPU compatibility.'
)


def _set_widget(node: dict, name: str, value) -> None:
    """Keep both ComfyUI serialization formats in agreement."""
    named = node.get('widgets_values_named', {})
    values = node.get('widgets_values', [])
    keys = list(named)
    if name not in named or len(values) != len(keys) or values != list(named.values()):
        raise ValueError('MOSS source widget contract changed: ' + node['type'] + '/' + name)
    values[keys.index(name)] = value
    named[name] = value


def _node(ident, kind, title, pos, inputs, outputs, named=None, size=(350, 180)):
    named = {} if named is None else named
    ports = []
    for name, typ, widget in inputs:
        port = {'name': name, 'type': typ, 'link': None}
        if widget:
            port['widget'] = {'name': name}
        ports.append(port)
    return {'id': ident, 'type': kind, 'title': title, 'pos': list(pos),
            'size': list(size), 'flags': {}, 'order': 0, 'mode': 0,
            'inputs': ports,
            'outputs': [{'name': name, 'type': typ, 'links': []} for name, typ in outputs],
            'properties': {'Node name for S&R': kind},
            'widgets_values': list(named.values()), 'widgets_values_named': dict(named)}


def _link_row(row):
    if isinstance(row, dict):
        return tuple(row[k] for k in ('id', 'origin_id', 'origin_slot', 'target_id', 'target_slot', 'type'))
    if isinstance(row, list) and len(row) == 6:
        return tuple(row)
    raise ValueError('Unexpected MOSS link representation.')


def validate_graph(graph: dict) -> None:
    nodes = {n['id']: n for n in graph['nodes']}
    links = {r[0]: r for r in graph['links']}
    if len(nodes) != len(graph['nodes']) or len(links) != len(graph['links']):
        raise ValueError('Duplicate img2img node or link ID.')
    for lid, src, out, dst, inp, kind in graph['links']:
        if src not in nodes or dst not in nodes:
            raise ValueError('Dangling img2img link.')
        output = nodes[src]['outputs'][out]
        target = nodes[dst]['inputs'][inp]
        if lid not in output['links'] or target['link'] != lid:
            raise ValueError('Img2img socket/link mismatch.')
        if output['type'] not in (kind, '*') or target['type'] not in (kind, '*'):
            raise ValueError('Img2img link type mismatch.')
    for n in nodes.values():
        for i, port in enumerate(n.get('inputs', [])):
            lid = port.get('link')
            if lid is not None and (lid not in links or links[lid][3:5] != [n['id'], i]):
                raise ValueError('Missing or misdirected img2img input link.')
        for i, port in enumerate(n.get('outputs', [])):
            for lid in port.get('links') or []:
                if lid not in links or links[lid][1:3] != [n['id'], i]:
                    raise ValueError('Missing or misdirected img2img output link.')
    # All image outputs must descend from the encoded source, never empty noise.
    samplers = [n for n in nodes.values() if n['type'] == 'KSampler']
    if len(samplers) != 1:
        raise ValueError('Expected one img2img sampler.')
    sampler = samplers[0]
    port = next(p for p in sampler['inputs'] if p['name'] == 'latent_image')
    if nodes[links[port['link']][1]]['type'] != 'VAEEncode':
        raise ValueError('Img2img sampler must use the encoded input image.')
    if any(n['type'].startswith('Empty') and 'Latent' in n['type'] for n in nodes.values()):
        raise ValueError('Empty latent must not remain in img2img.')
    if graph.get('definitions'):
        raise ValueError('Img2img is a flat native-node workflow.')


def make_img2img(stock: dict) -> dict:
    """Flatten one verified stock subgraph, replacing only the source-latent path.

    The caller verifies the immutable upstream Git blob before invoking this.
    Source values exposed on the outer subgraph override internal widget defaults,
    notably the negative caption and seed. Incoming widget sockets are restored
    to widgets instead of leaving unconnected converted inputs behind.
    """
    roots = [n for n in stock['nodes'] if 'unet_name' in n.get('widgets_values_named', {})]
    if len(roots) != 1:
        raise ValueError('Expected one MOSS source subgraph.')
    root = roots[0]
    matches = [g for g in stock.get('definitions', {}).get('subgraphs', []) if g['id'] == root['type']]
    if len(matches) != 1:
        raise ValueError('MOSS subgraph definition is missing.')
    sg = matches[0]
    original = {n['id']: n for n in sg['nodes']}
    expected = {1:'UNETLoader', 2:'CLIPLoader', 3:'VAELoader', 4:'CLIPTextEncode',
                5:'CLIPTextEncode', 6:'EmptyFlux2LatentImage', 7:'KSampler', 8:'VAEDecode',
                11:'TextGenerate', 13:'PreviewAny', 14:'PrimitiveStringMultiline',
                15:'SkipLayerGuidanceDiTSimple', 16:'ComfySwitchNode',
                17:'PrimitiveStringMultiline', 24:'ComfySwitchNode'}
    if {k:n['type'] for k,n in original.items()} != expected:
        raise ValueError('Pinned MOSS node layout changed; review img2img adapter.')
    in_id, out_id = sg['inputNode']['id'], sg['outputNode']['id']
    links = [_link_row(row) for row in sg['links']]
    adapted = {ident: copy.deepcopy(node) for ident, node in original.items() if ident != 6}
    external_ports = {}
    for lid, src, src_slot, dst, dst_slot, kind in links:
        if src != in_id or dst not in adapted:
            continue
        outer_name = sg['inputs'][src_slot]['name']
        if outer_name not in root['widgets_values_named']:
            raise ValueError('Missing exposed MOSS value: ' + outer_name)
        inner_name = original[dst]['inputs'][dst_slot]['name']
        _set_widget(adapted[dst], inner_name, root['widgets_values_named'][outer_name])
        external_ports.setdefault(dst, set()).add(dst_slot)
    for ident, node in adapted.items():
        node['inputs'] = [p for index,p in enumerate(node.get('inputs', []))
                          if index not in external_ports.get(ident, set())]
        for port in node.get('inputs', []):
            port['link'] = None
        for port in node.get('outputs', []):
            port['links'] = []
        node['flags'] = {}
        node['mode'] = 0
        node['properties'] = {'Node name for S&R': node['type']}
        # Remove stale sample-image, canvas-color and promoted-widget metadata.
        for key in ('color', 'bgcolor', 'showAdvanced', 'collapsed'):
            node.pop(key, None)
    _set_widget(adapted[7], 'denoise', DENOISE)
    _set_widget(adapted[17], 'value', PROMPT)
    _set_widget(adapted[4], 'text', PROMPT)

    layout = {
        1:('MOSS SFT MODEL - same as QMOSS',(400,0),(360,110)),
        2:('MOSS TEXT ENCODER - shared by conditioning and upsampler',(400,340),(360,130)),
        3:('MOSS VAE - same encoder and decoder',(400,180),(360,100)),
        4:('POSITIVE CONDITIONING',(1710,0),(340,240)),
        5:('NEGATIVE PROMPT - stock MOSS default',(1310,180),(350,190)),
        7:('IMAGE-TO-IMAGE - adjust DENOISE here',(2110,0),(370,330)),
        8:('DECODE THE GENERATED IMAGE',(2110,420),(300,100)),
        11:('STOCK TEXT PROMPT UPSAMPLER - no image input',(1310,700),(370,540)),
        13:('FINAL CAPTION USED BY MOSS',(1730,700),(370,500)),
        14:('STOCK UPSAMPLER SYSTEM PROMPT - unchanged',(810,790),(440,440)),
        15:('OPTIONAL SLG - stock settings',(810,420),(350,220)),
        16:('PROMPT UPSAMPLER ON / OFF',(1310,0),(350,110)),
        17:('DESCRIBE THE COMPLETE DESIRED IMAGE',(810,0),(440,320)),
        24:('SLG ON / OFF - stock default OFF',(1310,420),(350,120)),
    }
    for ident,(title,pos,size) in layout.items():
        adapted[ident]['title'] = title
        adapted[ident]['pos'], adapted[ident]['size'] = list(pos), list(size)
    for ident, node in list(adapted.items()):
        inputs = [{k: copy.deepcopy(p[k]) for k in ('name','type','link','shape','widget') if k in p}
                  for p in node.get('inputs', [])]
        outputs = [{k: copy.deepcopy(p[k]) for k in ('name','type','links')} for p in node['outputs']]
        adapted[ident] = {k: copy.deepcopy(node[k]) for k in
            ('id','type','title','pos','size','flags','mode','order','properties')}
        adapted[ident].update(inputs=inputs, outputs=outputs,
            widgets_values=copy.deepcopy(node.get('widgets_values', [])),
            widgets_values_named=copy.deepcopy(node.get('widgets_values_named', {})))
    source = _node(25,'LoadImage','SOURCE IMAGE - upload your photo',(0,0),[],
                   [('IMAGE','IMAGE'),('MASK','MASK')],{'image':'example.png','upload':'image'},(340,450))
    resize = _node(26,'ImageScaleToTotalPixels','WORKING SIZE - source ratio / 16px alignment',(0,510),
                   [('image','IMAGE',False)],[('IMAGE','IMAGE')],
                   {'upscale_method':'lanczos','megapixels':MEGAPIXELS,
                    'resolution_steps':RESOLUTION_STEPS},(340,180))
    encode = _node(27,'VAEEncode','ENCODE SOURCE - not an empty latent',(400,530),
                   [('pixels','IMAGE',False),('vae','VAE',False)],[('LATENT','LATENT')],size=(350,105))
    save = _node(28,'SaveImage','FINAL IMAGE - complete generated frame',(2540,0),
                 [('images','IMAGE',False)],[],{'filename_prefix':'MOSS_Image/I2I'},(560,780))
    note = _node(29,'Note','MOSS img2img usage and limits',(0,810),[],[],{'text':NOTE},(720,440))
    nodes = list(adapted.values()) + [source,resize,encode,save,note]
    by_id = {n['id']:n for n in nodes}
    graph = {'id':'1b163d91-ea0e-48f7-9c01-ed5cc2c4358f', 'revision':0,
             'last_node_id':29,'last_link_id':0,'nodes':nodes,'links':[],
             'groups':[],'config':{},'extra':{'frontendVersion':'1.55.17',
             'ds':{'scale':0.46,'offset':[100,100]},'workbench_source':{
                 'task':I2I_TASK,'version':VERSION,'derived_from':'QMOSS pinned ComfyUI T2I template',
                 'official_img2img_workflow':False,'reference_conditioned_edit_model':False,
                 'changes':'Flat native graph; source image -> area resize -> VAEEncode -> KSampler; denoise 0.35.',
                 'source_to_prompt_upsampler':False,'gpu_tested':False}},'version':0.4}

    def add(src, src_slot, dst, port_name, kind):
        target = by_id[dst]
        idx = next(i for i,p in enumerate(target['inputs']) if p['name'] == port_name)
        if target['inputs'][idx]['link'] is not None:
            raise ValueError('Two links target one input: ' + port_name)
        lid = len(graph['links']) + 1
        graph['links'].append([lid,src,src_slot,dst,idx,kind])
        target['inputs'][idx]['link'] = lid
        by_id[src]['outputs'][src_slot]['links'].append(lid)
    for lid,src,src_slot,dst,dst_slot,kind in links:
        if src in adapted and dst in adapted:
            add(src,src_slot,dst,original[dst]['inputs'][dst_slot]['name'],kind)
    outgoing = [r for r in links if r[3] == out_id]
    if len(outgoing) != 1 or outgoing[0][1:3] != (8,0):
        raise ValueError('Expected MOSS decoded-image output.')
    add(25,0,26,'image','IMAGE')
    add(26,0,27,'pixels','IMAGE')
    add(3,0,27,'vae','VAE')
    add(27,0,7,'latent_image','LATENT')
    add(8,0,28,'images','IMAGE')
    # Stable topological order and cycle check, independent of upstream UI order.
    pending = set(by_id)
    emitted = set()
    ordered = []
    while pending:
        ready = sorted(n for n in pending if all(r[1] in emitted for r in graph['links'] if r[3] == n))
        if not ready:
            raise ValueError('Cycle in img2img graph.')
        for ident in ready:
            by_id[ident]['order'] = len(ordered)
            ordered.append(by_id[ident]); emitted.add(ident); pending.remove(ident)
    graph['nodes'] = ordered
    graph['last_link_id'] = len(graph['links'])
    validate_graph(graph)
    return graph


def encode_graph(graph: dict) -> bytes:
    return (json.dumps(graph, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def provenance(graph: dict) -> dict:
    return {'task':I2I_TASK,'workflow':I2I_WORKFLOW,'adapter_version':VERSION,
            'sha256':hashlib.sha256(encode_graph(graph)).hexdigest(),
            'official_template':False,'conditioning':'source VAE latent plus text',
            'denoise':DENOISE,'megapixels':MEGAPIXELS,'resolution_steps':RESOLUTION_STEPS,
            'additional_models':[],'additional_custom_nodes':[], 'gpu_tested':False}
