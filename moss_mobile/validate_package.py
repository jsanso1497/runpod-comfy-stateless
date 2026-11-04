#!/usr/bin/env python3
"""Offline graph, App Mode, precision, model-lock and safety consistency checks."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent

def validate(root=ROOT):
    count = 0
    rows = json.loads((root/'models.lock.json').read_text())['models']
    assert len(rows) == 4, 'Unexpected model download added.'
    available = {Path(x['destination']).name for x in rows}
    for row in rows:
        assert re.fullmatch('[0-9a-f]{64}', row['sha256'])
        assert re.fullmatch('[0-9a-f]{40}', row['revision'])
        assert not Path(row['destination']).is_absolute() and '..' not in Path(row['destination']).parts
    for path in sorted((root/'workflows').glob('*.app.json')):
        graph = json.loads(path.read_text())
        api = json.loads((root/'api'/path.name.replace('.app.', '.api.')).read_text())
        nodes = {n['id']: n for n in graph['nodes']}
        assert len(nodes) == len(graph['nodes'])
        assert graph['extra']['linearMode'] is True
        assert graph['extra']['linearData']['inputs'] and graph['extra']['linearData']['outputs']
        for nid, widget, *_ in graph['extra']['linearData']['inputs']:
            assert widget in nodes[nid]['widgets_values_named'], (path.name, nid, widget)
        for nid in graph['extra']['linearData']['outputs']:
            assert nodes[int(nid)]['type'] == 'SaveImage'
        links = {link[0]:link for link in graph['links']}
        assert len(links) == len(graph['links'])
        for lid,src,slot,dst,input_slot,dtype in links.values():
            assert src in nodes and dst in nodes
            sn,dn = nodes[src],nodes[dst]
            assert lid in sn['outputs'][slot]['links']
            assert dn['inputs'][input_slot]['link'] == lid
            assert sn['outputs'][slot]['type'] == dtype
            assert api[str(dst)]['inputs'][dn['inputs'][input_slot]['name']] == [str(src),slot]
        for node in graph['nodes']:
            for i, inp in enumerate(node.get('inputs', [])):
                if inp.get('link') is not None:
                    assert links[inp['link']][3:5] == [node['id'], i]
            for slot,out in enumerate(node.get('outputs', [])):
                for lid in out.get('links') or []:
                    assert links[lid][1:3] == [node['id'], slot]
        # Verify DAG plus every model filename, and that all editable workflows have API counterparts.
        done=set()
        def visit(nid, active):
            assert nid not in active, 'Cycle in graph'
            if nid in done:return
            active=active | {nid}
            assert nid in api, 'Dangling API node reference'
            for value in api[nid]['inputs'].values():
                if isinstance(value,list):visit(value[0],active)
            done.add(nid)
        for nid,node in api.items():
            visit(nid,set())
            for key in ('unet_name','clip_name','vae_name','lora_name'):
                if key in node['inputs']:assert node['inputs'][key] in available
            if node['class_type']=='KSampler':
                steps,cfg,sampler=(25,2.5,'er_sde') if 'Generate' in path.name else (50,3.0,'euler')
                assert node['inputs']['steps']==steps and node['inputs']['cfg']==cfg
                assert node['inputs']['sampler_name']==sampler and node['inputs']['denoise']==1.0
            if node['class_type']=='TextEncodeMOSSImageEdit':
                assert node['inputs']['edit_format']=='ai-toolkit'
                assert 'images.image_0' in node['inputs'] and 'vae' in node['inputs']
            if 'Edit' in path.name:assert node['class_type']!='TextGenerate'
        slg=[n for n in api.values() if n['class_type']=='SkipLayerGuidanceDiTSimple']
        if 'Generate' in path.name:
            assert not slg, 'SFT stock recipe has SLG off.'
        else:
            assert len(slg)==1 and slg[0]['inputs']['single_layers']=='22'
        enc=[n['inputs'] for n in api.values() if n['class_type']=='TextEncodeMOSSImageEdit']
        if enc:
            assert len(enc)==2
            for key in enc[0]:
                if key.startswith('images.') or key in ('edit_format','vae','clip'):
                    assert enc[0][key]==enc[1][key]
        if 'Protected' in path.name:assert any(n['class_type']=='MossMobileComposite' for n in api.values())
        assert len(done)==len(api)
        count += 1
    assert count==4
    print(f'PASS: {count} native App Mode graphs, {count} API graphs, four pinned model files, no edit prompt enhancer.')
    return count

if __name__=='__main__':
    validate()
