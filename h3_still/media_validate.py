"""Validate all v2 graph links, required inputs, core contracts and installed schemas."""
from pathlib import Path
import json
from build_media_workflows import build, FILES

PACKAGE = Path(__file__).resolve().parent


def validate_media_graph(graph, profile, info=None):
    if graph != build(profile):
        raise ValueError('Media graph differs from its checked deterministic generator.')
    nodes = {n['id']: n for n in graph['nodes']}
    links = {e[0]: e for e in graph['links']}
    if len(nodes) != len(graph['nodes']) or len(links) != len(graph['links']):
        raise ValueError('Duplicate identifiers.')
    for lid, source, slot, target, target_slot, kind in links.values():
        a, b = nodes[source]['outputs'][slot], nodes[target]['inputs'][target_slot]
        if b['link'] != lid or lid not in (a['links'] or []) or a['type'] != kind or b['type'] != kind:
            raise ValueError('Broken graph connection.')
    for node in nodes.values():
        for index, port in enumerate(node['inputs']):
            if links[port['link']][3:5] != [node['id'], index]:
                raise ValueError('Incorrect reciprocal input connection.')
        for index, port in enumerate(node['outputs']):
            if any(links[lid][1:3] != [node['id'], index] for lid in port['links'] or []):
                raise ValueError('Incorrect reciprocal output connection.')
        if node['type'].startswith('AILab'):
            raise ValueError('AI Lab dependency returned.')
    remaining, done = set(nodes), set()
    while remaining:
        ready={i for i in remaining if all(links[p['link']][1] in done for p in nodes[i]['inputs'])}
        if not ready:
            raise ValueError('Graph cycle.')
        done.update(ready);remaining-=ready
    def origin(id, name):
        edge=links[next(p['link'] for p in nodes[id]['inputs'] if p['name']==name)]
        return edge[1],edge[2]
    assert origin(22,'latent_image')==(17,0)
    assert nodes[17]['type']=='FizgigH3StillLatent'
    assert origin(24,'images')==(28,0) and origin(28,'images')==(23,0)
    assert nodes[23]['type']=='FizgigH3StillDecode'
    assert origin(21,'model')==(19,0) and origin(22,'sigmas')==(19,2)
    assert origin(16,'plan')==(10,0) and origin(10,'video')==(30,0)
    assert origin(17,'width')==(10,1) and origin(17,'height')==(10,2)
    if profile==2:
        assert nodes[1]['type']=='LoadImage' and origin(31,'mask')==(1,1)
        assert origin(10,'reference_1')==(31,0)
        assert nodes[10]['widgets_values_named']['protect_background'] is True
    assert all(not nodes[i]['widgets_values_named']['enabled'] for i in (12,13))
    if info is not None:
        for node in nodes.values():
            kind=node['type']
            if kind=='Note':continue
            if kind not in info:raise ValueError('Missing node: '+kind)
            spec=info[kind];required=spec.get('input',{}).get('required',{})
            fields=dict(required,**spec.get('input',{}).get('optional',{}))
            connected={p['name'] for p in node['inputs']};named=node['widgets_values_named']
            for key in required:
                if key not in connected and key not in named:raise ValueError('Missing required input '+kind+'.'+key)
            for name,value in named.items():
                if name in ('upload','control_after_generate'):continue
                if name not in fields:raise ValueError('Unknown widget '+kind+'.'+name)
                declared=fields[name][0]
                if kind=='LoadImage' and name=='image':continue  # User supplies target photograph.
                if isinstance(declared,list) and value not in declared:raise ValueError('Invalid combo '+kind+'.'+name)
                if isinstance(declared, str) and declared in ('INT', 'FLOAT'):
                    limits=fields[name][1] if len(fields[name])>1 else {}
                    if not limits.get('min',-float('inf'))<=value<=limits.get('max',float('inf')):
                        raise ValueError('Out of range '+kind+'.'+name)
            for port in node['inputs']:
                field=fields.get(port['name'])
                if not field or field[0]!=port['type']:raise ValueError('Invalid port '+kind+'.'+port['name'])
            for i,port in enumerate(node['outputs']):
                if spec['output'][i]!=port['type']:raise ValueError('Invalid output '+kind)
    return True


def validate_all(info=None):
    for i,name in enumerate(FILES):
        validate_media_graph(json.loads((PACKAGE/'media_workflows'/name).read_text()),i,info)
    return True

if __name__=='__main__':
    validate_all();print('Three v2 graph contracts passed. No inference was run.')
