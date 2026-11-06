"""Strict structural checks, plus optional checks against the installed schema."""
import json
from pathlib import Path
from build_studio_workflow import build, FILENAME

PACKAGE = Path(__file__).resolve().parent
FRONTEND = {'Note', 'MarkdownNote'}


def validate_studio(info=None, graph=None):
    graph = graph or json.loads((PACKAGE/'studio_workflows'/FILENAME).read_text())
    if graph != build():
        raise ValueError('Committed Studio workflow differs from its deterministic generator.')
    nodes = {n['id']: n for n in graph['nodes']}
    links = {e[0]: e for e in graph['links']}
    if len(nodes) != len(graph['nodes']) or len(links) != len(graph['links']):
        raise ValueError('Duplicate graph identifiers.')
    for lid, source, slot, target, target_slot, kind in links.values():
        a,b = nodes[source]['outputs'][slot], nodes[target]['inputs'][target_slot]
        if b['link'] != lid or lid not in a['links'] or a['type'] != kind or b['type'] != kind:
            raise ValueError('Broken or mismatched graph connection.')
    for node in nodes.values():
        for i, port in enumerate(node.get('inputs', [])):
            if port['link'] not in links or links[port['link']][3:5] != [node['id'], i]:
                raise ValueError('Input has an inconsistent reciprocal link.')
        for i, port in enumerate(node.get('outputs', [])):
            for lid in port.get('links') or []:
                if lid not in links or links[lid][1:3] != [node['id'], i]:
                    raise ValueError('Output has an inconsistent reciprocal link.')
        if node['type'].startswith('AILab'):
            raise ValueError('Studio must not depend on AI Lab/RMBG.')
    unseen=set(nodes);done=set()
    while unseen:
        ready={i for i in unseen if all(links[p['link']][1] in done for p in nodes[i]['inputs'])}
        if not ready:raise ValueError('Workflow contains a cycle.')
        done.update(ready);unseen-=ready
    optional=[n for n in nodes.values() if n['type']=='H3StudioOptionalImage']
    assert len(optional)==9
    assert all(n['widgets_values_named']['image']=='(none)' for n in optional)
    assert nodes[10]['widgets_values_named']['mode']=='Auto'
    assert nodes[10]['widgets_values_named']['preset']=='Quality - 50 steps, no Turbo'
    assert all(not nodes[i]['widgets_values_named']['enabled'] for i in (12,13))
    def source(id, field):
        edge=links[next(p['link'] for p in nodes[id]['inputs'] if p['name']==field)]
        return (edge[1],edge[2])
    assert source(22,'latent_image')==(17,0) and nodes[17]['type']=='FizgigH3StillLatent'
    assert source(24,'images')==(23,0) and nodes[23]['type']=='FizgigH3StillDecode'
    assert source(17,'width')==(10,1) and source(17,'height')==(10,2)
    assert source(16,'plan')==(10,0) and source(11,'plan')==(10,0)
    if info is not None:
        for node in nodes.values():
            if node['type'] in FRONTEND:continue
            spec=info.get(node['type'])
            if not spec:raise ValueError('Missing installed node: '+node['type'])
            required=spec.get('input',{}).get('required',{})
            fields=dict(required,**spec.get('input',{}).get('optional',{}))
            connected={port['name'] for port in node['inputs']}
            named=node.get('widgets_values_named',{})
            for name in required:
                if name not in connected and name not in named:
                    raise ValueError('Missing required input: '+node['type']+'.'+name)
            for name,value in named.items():
                if name in ('upload','control_after_generate'):continue
                if name not in fields:raise ValueError('Unrecognized widget: '+node['type']+'.'+name)
                kind=fields[name][0]
                if isinstance(kind,list) and value not in kind:
                    raise ValueError('Unsupported value: '+node['type']+'.'+name)
            for i, port in enumerate(node['outputs']):
                actual=spec.get('output',[])
                if i>=len(actual) or actual[i]!=port['type']:
                    raise ValueError('Installed output differs: '+node['type'])
            for port in node['inputs']:
                if port['name'] not in fields:
                    raise ValueError('Installed input is absent: '+node['type']+'.'+port['name'])
                actual=fields[port['name']][0]
                if isinstance(actual,str) and actual!=port['type']:
                    raise ValueError('Installed input type differs: '+node['type']+'.'+port['name'])
        from importlib.util import spec_from_file_location, module_from_spec
        spec=spec_from_file_location('studio_rules_validate', PACKAGE/'nodes/studio/logic.py')
        module=module_from_spec(spec);spec.loader.exec_module(module)
        names=info['UNETLoader']['input']['required']['unet_name'][0]
        if module.FL2VA not in names or module.REF2VA not in names:
            raise ValueError('The FL2VA or REF2VA INT8 model is not installed.')
    return True

if __name__=='__main__':
    validate_studio();print('Studio graph checks passed; no GPU inference was performed.')
