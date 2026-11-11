#!/usr/bin/env python3
"""Add community sampling experiments without altering the native six-step baselines.

These graphs are reconstructions of settings in user-supplied screenshots.
The uploaded JPEG contained no recoverable ComfyUI graph or prompt metadata.
"""
import copy
import importlib.util
import json
from pathlib import Path
import uuid

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h3_manual_baseline_builder', PACKAGE / 'build_workflows.py')
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
VERSION = '2026-11-10-community.1'
PRUNA_VAE = 'p-h3-image_vae_int8_convrot.safetensors'
# Six steps and the simple scheduler for LCM are controlled test choices, not settings
# recoverable from the LCM screenshot. ER-SDE four/six steps come from the second screenshot.
PROFILES = [
    ('10_Community_T2I_Euler_16MP_Control', 0, 'euler', 6, False),
    ('11_Community_T2I_ER_SDE_6_16MP', 0, 'er_sde', 6, False),
    ('12_Community_T2I_ER_SDE_4_16MP', 0, 'er_sde', 4, False),
    ('13_Community_T2I_LCM_6_16MP', 0, 'lcm', 6, False),
    ('14_Community_Replace_ER_SDE_6_16MP', 3, 'er_sde', 6, False),
    ('15_Community_Replace_LCM_6_16MP', 3, 'lcm', 6, False),
    ('16_Community_Two_People_ER_SDE_6_16MP', 4, 'er_sde', 6, False),
    ('17_Community_Two_People_LCM_6_16MP', 4, 'lcm', 6, False),
    ('18_Community_Pruna_Decode_AB_16MP', 0, 'euler', 6, True),
]


def reindex(graph):
    nodes = {n['id']: n for n in graph['nodes']}
    for n in nodes.values():
        for port in n['outputs']:
            port['links'] = []
    for edge in graph['links']:
        ident, source, out, dest, inp, kind = edge
        nodes[source]['outputs'][out]['links'].append(ident)
        nodes[dest]['inputs'][inp]['link'] = ident
    for n in nodes.values():
        for port in n['outputs']:
            port['links'] = port['links'] or None
    # A topological order keeps legacy canvases and native schemas in agreement.
    incoming = {ident: 0 for ident in nodes}
    children = {ident: [] for ident in nodes}
    for _, source, _, dest, _, _ in graph['links']:
        incoming[dest] += 1
        children[source].append(dest)
    ready = sorted(n for n, degree in incoming.items() if not degree)
    order = []
    while ready:
        ident = ready.pop(0)
        order.append(ident)
        for child in children[ident]:
            incoming[child] -= 1
            if not incoming[child]:
                ready.append(child)
                ready.sort()
    if len(order) != len(nodes):
        raise ValueError('Cycle in workflow')
    for index, ident in enumerate(order):
        nodes[ident]['order'] = index
    graph['last_node_id'] = max(nodes)
    graph['last_link_id'] = max((v[0] for v in graph['links']), default=0)
    return graph


def api_from_graph(graph):
    nodes = {n['id']: n for n in graph['nodes']}
    links = {v[0]: v for v in graph['links']}
    api = {}
    for ident, n in nodes.items():
        if n['type'] in ('Note', 'MarkdownNote'):
            continue
        inputs = {k: v for k, v in n.get('widgets_values_named', {}).items()
                  if k not in ('control_after_generate', 'upload')}
        for port in n['inputs']:
            if port.get('link') is not None:
                edge = links[port['link']]
                inputs[port['name']] = [str(edge[1]), edge[2]]
        api[str(ident)] = {'class_type': n['type'], 'inputs': inputs,
                          '_meta': {'title': n.get('title', n['type'])}}
    return api


def build(profile):
    name, base_index, sampler, steps, vae_ab = profile
    graph, _ = BASE.build(BASE.PROFILES[base_index])
    graph = copy.deepcopy(graph)
    nodes = {n['id']: n for n in graph['nodes']}
    graph['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, VERSION + '/' + name))
    graph['extra']['h3_image'].update(steps=steps, community_version=VERSION,
        source='User screenshots; NOT an extracted original workflow',
        sampler=sampler, scheduler='simple', output_megapixels=16.0,
        experimental=(sampler != 'euler' or vae_ab), decode_only_vae_comparison=vae_ab)
    nodes[40]['widgets_values_named']['output_megapixels'] = 16.0
    nodes[40]['widgets_values'] = list(nodes[40]['widgets_values_named'].values())
    nodes[40]['title'] = 'YOUR MANUAL PROMPT + 16 MP (4 MP only for a memory smoke test)'
    nodes[44]['widgets_values_named']['run_label'] = name
    nodes[44]['widgets_values'] = [name]
    for n in nodes.values():
        if n['type'] == 'SaveImage':
            suffix = 'comparison' if n['id'] == 50 else 'final'
            prefix = 'H3_Image/Community_Tests/' + name + '/' + suffix
            n['widgets_values_named']['filename_prefix'] = prefix
            n['widgets_values'] = [prefix]

    # Replace only the scheduler in these NEW graphs. Original workflow files and
    # the verified Euler helper remain unchanged.
    n = nodes[41]
    n['type'] = 'BasicScheduler'
    n['title'] = 'SIMPLE SCHEDULE: %s steps, denoise 1.0' % steps
    n['properties'] = {'Node name for S&R': 'BasicScheduler', 'cnr_id': 'comfy-core'}
    n['outputs'] = [{'name': 'SIGMAS', 'type': 'SIGMAS', 'links': []}]
    n['widgets_values_named'] = {'scheduler': 'simple', 'steps': steps, 'denoise': 1.0}
    n['widgets_values'] = ['simple', steps, 1.0]
    n['pos'] = [2760, 170]
    n['size'] = [365, 150]
    for edge in graph['links']:
        if edge[1] == 41:
            if edge[5] == 'SAMPLER':
                edge[1], edge[2] = 51, 0
            else:
                edge[2] = 0

    def add(ident, kind, title, xy, out=(), widgets=None, size=(365, 160)):
        widgets = widgets or {}
        item = {'id': ident, 'type': kind, 'title': title, 'pos': list(xy),
                'size': list(size), 'flags': {}, 'order': len(graph['nodes']), 'mode': 0,
                'inputs': [], 'outputs': [{'name': a, 'type': b, 'links': []} for a, b in out],
                'properties': {'Node name for S&R': kind},
                'widgets_values': list(widgets.values()), 'widgets_values_named': widgets}
        if not kind.startswith('H3Image'):
            item['properties']['cnr_id'] = 'comfy-core'
        nodes[ident] = item
        graph['nodes'].append(item)
        return item

    def link(source, port, target, label, kind):
        ident = max((v[0] for v in graph['links']), default=0) + 1
        slot = len(nodes[target]['inputs'])
        nodes[target]['inputs'].append({'name': label, 'type': kind, 'link': ident})
        graph['links'].append([ident, source, port, target, slot, kind])

    if sampler == 'lcm':
        add(51, 'SamplerLCM', 'LCM NOISE: set BOTH fields together', (2760, 540),
            [('SAMPLER', 'SAMPLER')], {'s_noise': 1.0, 's_noise_end': 1.0, 'noise_clip_std': 0.0}, size=(365, 160))
    else:
        add(51, 'KSamplerSelect', 'SAMPLER: ' + sampler, (2760, 540),
            [('SAMPLER', 'SAMPLER')], {'sampler_name': sampler})

    if vae_ab:
        nodes[45]['title'] = 'A: Native image VAE, same generated latent'
        nodes[46]['title'] = 'A: NATIVE FINAL PNG'
        nodes[46]['widgets_values_named']['filename_prefix'] = 'H3_Image/Community_Tests/' + name + '/A_native'
        nodes[46]['widgets_values'] = list(nodes[46]['widgets_values_named'].values())
        add(52, 'VAELoader', 'OPTIONAL Pruna INT8 VAE: DECODE ONLY', (1480, 550),
            [('VAE', 'VAE')], {'vae_name': PRUNA_VAE}, size=(470, 110))
        add(53, 'H3ImageTimedDecode', 'B: Pruna decode of EXACT SAME latent', (3200, 650),
            [('image', 'IMAGE'), ('complete_report', 'STRING')])
        link(44, 0, 53, 'samples', 'LATENT')
        link(52, 0, 53, 'vae', 'VAE')
        link(44, 1, 53, 'sampling_report', 'STRING')
        add(54, 'SaveImage', 'B: PRUNA FINAL PNG (experimental)', (3640, 720),
            widgets={'filename_prefix': 'H3_Image/Community_Tests/' + name + '/B_pruna'}, size=(620, 600))
        link(53, 0, 54, 'images', 'IMAGE')
        add(55, 'H3ImageReport', 'B: PRUNA DECODE TIME', (2020, 1120),
            widgets={'save_report': True}, size=(660, 360))
        link(53, 1, 55, 'report', 'STRING')
        # Smaller side-by-side preview only; full-res originals above are unaffected.
        for ident, src, label, y in [(56,45,'Native preview',890),(57,53,'Pruna preview',1110)]:
            add(ident,'ImageScaleToMaxDimension',label,(2760,y),[('IMAGE','IMAGE')],
                {'upscale_method':'area','largest_size':1024},size=(365,140))
            link(src,0,ident,'image','IMAGE')
        add(58,'ImageStitch','A native LEFT / B Pruna RIGHT',(3200,1150),[('IMAGE','IMAGE')],
            {'direction':'right','match_image_size':True,'spacing_width':8,'spacing_color':'black'},size=(365,180))
        link(56,0,58,'image1','IMAGE')
        link(57,0,58,'image2','IMAGE')
        add(59,'SaveImage','COMPARISON PREVIEW ONLY',(3640,1430),
            widgets={'filename_prefix':'H3_Image/Community_Tests/'+name+'/compare_native_left'},size=(620,400))
        link(58,0,59,'images','IMAGE')

    notes = [
        'NATIVE H3 IMAGE COMMUNITY EXPERIMENT. Not the old video H3, Eros or P-H3 workflow.',
        'Recreated from visible screenshots. The uploaded JPEG had no embedded workflow, seed or prompt.',
        'Your prompt is blank and passed verbatim. Same INT8 checkpoint, NVFP4/AWQ encoder and fixed seed as the native workspace.',
        '16 MP is the target in the reports. Begin at 4 MP only to check memory, then test 16 MP when feasible. A40 fit is unverified.',
        'For sampler comparisons use IDENTICAL prompt, references, dimensions, precision and seed. A fixed seed does not lock the image across samplers.',
        'Compare 10 (Euler 6) with 11 (ER-SDE 6) before 12 (ER-SDE 4). Four steps changes the trained six-step recipe and is experimental.',
        'LCM: six steps/simple is our controlled starting recipe; the screenshot does NOT specify its steps or schedule. Set s_noise and s_noise_end BOTH to the same value, e.g. 0.95, 1.00, 1.10 or 1.20. noise_clip_std stays 0.',
        'No extra LCM LoRA, model-sampling patch, prompt enhancer, Turbo, Akatz or accelerator is required/added.',
        'The old 00-05 workflows remain untouched. No simultaneous seed sweep or queued batch is included.',
        'Reports contain actual step count/timing. Inspect PNG workflow metadata for the sampler/noise/VAE values; run_label alone is not proof of settings.',
    ]
    if base_index == 3:
        notes += ['Picture 1: scene. Picture 2: head/face. Picture 3: body. Output follows scene aspect.',
                  'Specify identity, body, expression and gaze roles yourself. This is whole-image generation, not SAM/inpainting.']
    elif base_index == 4:
        notes += ['Picture 1: A head. Picture 2: A body. Picture 3: B head. Picture 4: B body.',
                  'Write the scene yourself. Labels do not impose identity binding. Select the desired output aspect ratio.']
    if vae_ab:
        notes += ['OPTIONAL: run community/manage.py download-vae first. This selects Pruna INT8 because the screenshot did not name a precision/file.',
                  'Exactly ONE sampling pass feeds BOTH decoders. Native reference conditioning is NOT replaced. The optional decoder is not evidence of an officially supported cross-model VAE.',
                  'This controlled decoder-only comparison is deliberately narrower than the ambiguous VAE swap in the screenshot. Different results are not automatically better.']
    nodes[48]['widgets_values_named'] = {'text':'\n\n'.join(notes)}
    nodes[48]['widgets_values'] = [nodes[48]['widgets_values_named']['text']]
    nodes[48]['size'] = [1390, 700]
    nodes[48]['title'] = 'READ FIRST: reconstructed community test, not a recovered graph'
    if vae_ab:
        nodes[48]['pos'] = [0,750]
    reindex(graph)
    api = api_from_graph(graph)
    validate(graph, api)
    return graph, api


def validate(graph, api):
    BASE.validate(graph)
    nodes = {n['id']:n for n in graph['nodes']}
    assert graph['extra']['h3_image']['manual_prompt_only'] is True
    assert nodes[40]['widgets_values_named']['prompt'] == ''
    assert nodes[43]['widgets_values_named']['control_after_generate'] == 'fixed'
    assert nodes[30]['widgets_values_named']['unet_name'] == BASE.MODEL
    assert nodes[31]['widgets_values_named']['clip_name'] == BASE.ENCODER
    assert nodes[32]['widgets_values_named']['vae_name'] == BASE.VAE
    assert nodes[41]['type'] == 'BasicScheduler'
    assert nodes[41]['widgets_values_named']['scheduler'] == 'simple'
    assert api == api_from_graph(graph)
    for ident, n in nodes.items():
        if n['type'] in ('Note','MarkdownNote'): continue
        assert n['widgets_values'] == list(n['widgets_values_named'].values())
        assert str(ident) in api
    return True


def main():
    for name in ('workflows','api_templates'):
        (PACKAGE/name).mkdir(parents=True,exist_ok=True)
    rows=[]
    for profile in PROFILES:
        graph,api=build(profile)
        for folder,data in [('workflows',graph),('api_templates',api)]:
            (PACKAGE/folder/(profile[0]+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        rows.append({'file':profile[0]+'.json','sampler':profile[2],'steps':profile[3],
                     'output_megapixels':16.,'sampler_node':'SamplerLCM' if profile[2]=='lcm' else 'KSamplerSelect',
                     'optional_pruna_vae':profile[4],'source_is_reconstruction':True})
    (PACKAGE/'community/manifest.json').write_text(json.dumps({'version':VERSION,'workflows':rows},indent=2)+'\n')
    print('Generated %d additive community workflow/API pairs.'%len(rows))


if __name__=='__main__':main()
