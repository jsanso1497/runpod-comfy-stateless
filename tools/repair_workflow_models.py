#!/usr/bin/env python3
"""Repair a single-family saved ComfyUI UI graph into a NEW file.

Inference uses diffusion selections and real VAE links, never node titles.
This changes model-loader fields only. Prompts, media, seed, sampler and wiring
stay unchanged. Mixed-family graphs require manual review and are rejected.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('_repair_policy', REPO/'model_safety/node/policy.py')
policy = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(policy)


def values(node):
    native = policy.SPECS.get(node['type'], (node['type'],))[0]
    fields = policy.NATIVE_FIELDS.get(native, ())
    return native, dict(zip(fields, node.get('widgets_values', [])))


def repair(graph, protected=False):
    if not isinstance(graph, dict) or not isinstance(graph.get('nodes'), list):
        raise ValueError('Export the ComfyUI workflow JSON with a nodes array, not API format or the error report.')
    out = copy.deepcopy(graph)
    ns = {n['id']: n for n in out['nodes']}
    selected = []
    for n in ns.values():
        native, v = values(n)
        if native == 'UNETLoader':
            name = v.get('unet_name')
            if name in policy.KREA_DIFFUSION: selected.append(('krea', name))
            elif name in policy.H3_DIFFUSION: selected.append(('h3', name))
            else: raise ValueError('Unrecognized diffusion model. This repair tool does not substitute arbitrary architectures.')
    if not selected or len({f for f, _ in selected}) != 1 or len({n for _, n in selected}) != 1:
        raise ValueError('Expected one unambiguous model family/checkpoint. Mixed H3/KREA graphs are not auto-repaired.')
    family, diffusion = selected[0]
    index = policy.H3_DIFFUSION.index(diffusion) if family == 'h3' else 0
    changes = []
    for n in ns.values():
        native, v = values(n)
        if native not in policy.NATIVE_FIELDS: continue
        updated = dict(v)
        if native == 'CLIPLoader':
            updated['clip_name'] = policy.KREA_ENCODER if family == 'krea' else policy.H3_ENCODERS[index]
            updated['type'] = 'krea2' if family == 'krea' else 'minimax'
            updated.setdefault('device', 'default')
        elif native == 'VAELoader':
            if family == 'krea': updated['vae_name'] = policy.KREA_VAE
            else:
                roles = set()
                for link in out.get('links', []):
                    if link[1] != n['id']: continue
                    dest = ns[link[3]]
                    slot = dest.get('inputs', [])[link[4]]['name']
                    if slot == 'audio_vae' or dest['type'] in ('VAEDecodeAudio', 'VAEEncodeAudio'):
                        roles.add('audio')
                    elif slot == 'vae': roles.add('video')
                if len(roles) != 1:
                    raise ValueError(f'Cannot prove video/audio VAE role from the connections of node {n["id"]}. No output written.')
                updated['vae_name'] = policy.H3_AUDIO_VAE if roles == {'audio'} else policy.H3_VIDEO_VAE
        for key, value in updated.items():
            if v.get(key) != value: changes.append(f'Node {n["id"]}: {key} -> {value}')
        n['type'] = native
        n['widgets_values'] = [updated[k] for k in policy.NATIVE_FIELDS[native]]
        n['widgets_values_named'] = dict(zip(policy.NATIVE_FIELDS[native], n['widgets_values']))
        n.setdefault('properties', {})['Node name for S&R'] = native
        n['properties'].pop('models', None)  # Remove stale Z-Image/H3 auto-download metadata.
    out['revision'] = int(out.get('revision', 0)) + 1
    if protected: out = policy.protect_graph(out)
    return out, changes


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--protected', action='store_true', help='Requires the 1.5.3 Model Safety nodes.')
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve(): raise SystemExit('Use a NEW output path; original workflows are never overwritten.')
    result, changes = repair(json.loads(a.input.read_text(encoding='utf-8-sig')), a.protected)
    with a.output.open('x', encoding='utf-8') as f: f.write(json.dumps(result, indent=2)+'\n')
    print('\n'.join(changes) if changes else 'Model-family settings were already correct.')
    print('Saved new workflow: '+str(a.output))
