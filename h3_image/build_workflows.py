#!/usr/bin/env python3
"""Build simple graphs that use native H3 Image, not the old video/Fizgig graph."""
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
MODEL = 'minimax_h3_image_pruned_int8_convrot.safetensors'
ENCODER = 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
VAE = 'minimax_h3_image_vae_fp16.safetensors'
PROFILES = [
    ('00_Generate_No_References', 0, 0, False, '1:1'),
    ('01_Edit_One_Image', 1, 1, True, '1:1'),
    ('02_Generate_From_References', 16, 1, False, '2:3'),
    ('03_Replace_Person_Manual', 3, 3, True, '2:3'),
    ('04_Two_People_Heads_Bodies_Manual', 4, 4, False, '3:2'),
    ('05_Replace_Body_Face_Scene_Last', 3, 3, True, '2:3'),
]


class Graph:
    def __init__(self, name):
        self.data = {'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'h3-image-int8-encoder-v3/' + name)),
                     'revision': 0, 'last_node_id': 0, 'last_link_id': 0, 'version': 0.4,
                     'nodes': [], 'links': [], 'groups': [], 'config': {},
                     'extra': {'ds': {'scale': 0.55, 'offset': [30, 80]},
                               'h3_image': {'manual_prompt_only': True, 'gpu_tested': False,
                                            'native_model': 'MiniMax-H3-Image', 'steps': 6}}}
        self.nodes = {}
        self.api = {}

    def add(self, ident, kind, title, x, y, outputs=(), widgets=None, size=(340, 150), collapsed=False):
        values = widgets or {}
        node = {'id': ident, 'type': kind, 'title': title, 'pos': [x, y], 'size': list(size),
                'flags': {'collapsed': True} if collapsed else {}, 'order': len(self.nodes), 'mode': 0,
                'inputs': [], 'outputs': [{'name': n, 'type': t, 'links': []} for n, t in outputs],
                'properties': {'Node name for S&R': kind}, 'widgets_values': list(values.values()),
                'widgets_values_named': values}
        self.nodes[ident] = node
        self.data['nodes'].append(node)
        if kind != 'Note':
            self.api[str(ident)] = {'class_type': kind,
                'inputs': {k: v for k, v in values.items() if k not in ('control_after_generate', 'upload')},
                '_meta': {'title': title}}
        return node

    def link(self, origin, port, target, name, type_name):
        ident = len(self.data['links']) + 1
        inputs = self.nodes[target]['inputs']
        target_port = len(inputs)
        inputs.append({'name': name, 'type': type_name, 'link': ident})
        self.nodes[origin]['outputs'][port]['links'].append(ident)
        self.data['links'].append([ident, origin, port, target, target_port, type_name])
        self.api[str(target)]['inputs'][name] = [str(origin), port]

    def finish(self):
        for n in self.nodes.values():
            for o in n['outputs']:
                if not o['links']:
                    o['links'] = None
        self.data['last_node_id'] = max(self.nodes)
        self.data['last_link_id'] = len(self.data['links'])
        return self.data, self.api


def build(profile):
    name, count, minimum, follow, aspect = profile
    g = Graph(name)
    for i in range(1, count + 1):
        role = 'optional reference'
        if name.startswith('03_'):
            role = ['target scene', 'face / head', 'body'][i - 1]
        elif name.startswith('04_'):
            role = ['Person A - head', 'Person A - body', 'Person B - head', 'Person B - body'][i - 1]
        elif name.startswith('05_'):
            role = ['body', 'face / head', 'target scene'][i - 1]
        elif name.startswith('01_'):
            role = 'source photo to edit'
        g.add(i, 'H3ImageOptionalReference', 'Picture %s: %s' % (i, role), ((i - 1) % 4) * 350,
              ((i - 1) // 4) * 350, [('reference', 'H3_IMAGE_REFERENCE')],
              {'image': '(none)', 'enabled': True, 'upload': 'image'}, size=(325, 300), collapsed=i > 4)
    g.add(30, 'UNETLoader', 'NEW H3 Image INT8 - not video H3 or Eros', 1480, 0, [('MODEL', 'MODEL')],
          {'unet_name': MODEL, 'weight_dtype': 'default'}, size=(470, 110))
    g.add(31, 'CLIPLoader', 'H3 Qwen3-VL-32B INT8 encoder - not a prompt enhancer', 1480, 170, [('CLIP', 'CLIP')],
          {'clip_name': ENCODER, 'type': 'minimax', 'device': 'default'}, size=(470, 145))
    g.add(32, 'VAELoader', 'NEW image VAE - no Fizgig video decode', 1480, 370, [('VAE', 'VAE')],
          {'vae_name': VAE}, size=(470, 100))
    g.add(40, 'H3ImageManualConditioning', 'WRITE YOUR FULL PROMPT HERE - nothing is added', 2020, 0,
          [('conditioning', 'CONDITIONING'), ('image_latent', 'LATENT'), ('reference_report', 'STRING')],
          {'prompt': '', 'aspect_ratio': aspect, 'output_megapixels': 4.,
           'follow_image_1_aspect': follow, 'minimum_references': minimum, 'aspect_source_slot': 3 if name.startswith('05_') else 0}, size=(660, 500))
    g.link(31, 0, 40, 'clip', 'CLIP')
    g.link(32, 0, 40, 'vae', 'VAE')
    for i in range(1, count + 1):
        g.link(i, 0, 40, 'reference_%s' % i, 'H3_IMAGE_REFERENCE')
    g.add(41, 'H3ImageSixStepSchedule', 'Fixed six-step PDD schedule / Euler', 2760, 160,
          [('euler_sampler', 'SAMPLER'), ('six_step_sigmas', 'SIGMAS')], size=(365, 115))
    g.link(30, 0, 41, 'model', 'MODEL')
    g.add(42, 'BasicGuider', 'No classifier-free guidance', 2760, 340, [('GUIDER', 'GUIDER')], size=(365, 110))
    g.link(30, 0, 42, 'model', 'MODEL')
    g.link(40, 0, 42, 'conditioning', 'CONDITIONING')
    g.add(43, 'RandomNoise', 'Seed - fixed while comparing', 2760, 0, [('NOISE', 'NOISE')],
          {'noise_seed': 18071042, 'control_after_generate': 'fixed'}, size=(365, 110))
    g.add(44, 'H3ImageTimedSampler', 'Generate one image + timing', 3200, 0,
          [('samples', 'LATENT'), ('sampling_report', 'STRING')], widgets={'run_label': name}, size=(350, 300))
    for a, b, n, t in [(43, 0, 'noise', 'NOISE'), (42, 0, 'guider', 'GUIDER'),
                        (41, 0, 'sampler', 'SAMPLER'), (41, 1, 'sigmas', 'SIGMAS'),
                        (40, 1, 'latent_image', 'LATENT')]:
        g.link(a, b, 44, n, t)
    g.link(40, 2, 44, 'reference_report', 'STRING')
    g.add(45, 'H3ImageTimedDecode', 'Native H3 IMAGE decode + timing', 3200, 390, [('image', 'IMAGE'), ('complete_report', 'STRING')], size=(350, 130))
    g.link(44, 1, 45, 'sampling_report', 'STRING')
    g.link(44, 0, 45, 'samples', 'LATENT')
    g.link(32, 0, 45, 'vae', 'VAE')
    g.add(46, 'SaveImage', 'FINAL PNG', 3640, 0, widgets={'filename_prefix': 'H3_Image/' + name}, size=(620, 600))
    g.link(45, 0, 46, 'images', 'IMAGE')
    g.add(47, 'H3ImageReport', 'REFERENCE SIZES, SAMPLING / DECODE TIME', 2020, 580, widgets={'save_report': True}, size=(660, 450))
    g.link(45, 1, 47, 'report', 'STRING')
    if name.startswith(('01_', '03_', '05_')):
        source_slot = 3 if name.startswith('05_') else 1
        g.add(49, 'H3ImageCompare', 'Source LEFT / result RIGHT', 3200, 610, [('comparison', 'IMAGE')], size=(350, 120))
        g.link(source_slot, 0, 49, 'reference', 'H3_IMAGE_REFERENCE')
        g.link(45, 0, 49, 'generated', 'IMAGE')
        g.add(50, 'SaveImage', 'COMPARISON ONLY - final PNG is above', 3640, 720, widgets={'filename_prefix': 'H3_Image/Comparisons/' + name}, size=(620, 450))
        g.link(49, 0, 50, 'images', 'IMAGE')
    note = ('This requires the NEW H3 Image workspace. It is NOT a workflow for the existing H3 Lab.\n\n'
            'Prompt is passed verbatim. No prompt builder, enhancer, hidden prefixes, negative prompt, Turbo or Akatz.\n'
            'Native MiniMax-H3-Image INT8 model, matching IMAGE VAE, six PDD steps, Euler.\n'
            'Start at 4 MP with few references; 16 MP is not an A40 memory guarantee.\n'
            'Aspect source slot: 0 = first active picture; 3 = original upload slot 3. Only used when follow_image_1_aspect is ON.\n'
            'Picture tags follow ACTIVE slots in order. Disabled/empty slots are omitted. Check the report.\n'
            'Labels are reminders only. State head/body/scene roles yourself in your prompt.\n'
            'Native reference resizing can increase small references; lowering output MP alone does not cap all reference cost.\n'
            'This is whole-image generation/editing, not exact background preservation or inpainting.\n'
            'The original Lab and files under /workspace/h3-still are not modified.\n'
            'Reports include actual vision/VAE reference sizes and sampling/decode time. Comparisons are resized previews, not the final image.\n'
            'Your results save under /workspace/h3-image/output/. No GPU inference has been validated by this package author.')
    g.add(48, 'Note', 'READ FIRST', 0, 1520 if count > 4 else 440,
          widgets={'text': note}, size=(1390, 360))
    return g.finish()


def validate(graph):
    nodes = {n['id']: n for n in graph['nodes']}
    if len(nodes) != len(graph['nodes']):
        raise ValueError('Duplicate node ID')
    links = {}
    for ident, source, out, target, inp, kind in graph['links']:
        if ident in links:
            raise ValueError('Duplicate link ID')
        links[ident] = (source, out, target, inp, kind)
        a, b = nodes[source]['outputs'][out], nodes[target]['inputs'][inp]
        assert a['type'] == b['type'] == kind
        assert ident in a['links'] and b['link'] == ident
    for n in nodes.values():
        for inp in n['inputs']:
            assert inp['link'] in links
        for out in n['outputs']:
            assert all(v in links for v in (out['links'] or []))
    assert graph['last_node_id'] == max(nodes)
    assert graph['last_link_id'] == max(links, default=0)
    assert nodes[40]['widgets_values_named']['prompt'] == ''
    assert not any(n['type'] in ['FizgigH3StillLatent', 'FizgigH3StillDecode', 'PH3ImageConditioning', 'LoraLoaderModelOnly'] for n in nodes.values())
    return True


def main():
    for folder in ('workflows', 'api_templates'):
        (ROOT / folder).mkdir(exist_ok=True)
    for profile in PROFILES:
        graph, api = build(profile)
        validate(graph)
        (ROOT / 'workflows' / (profile[0] + '.json')).write_text(json.dumps(graph, indent=2) + '\n')
        (ROOT / 'api_templates' / (profile[0] + '.json')).write_text(json.dumps(api, indent=2) + '\n')
    print('Built and validated %s workflow graphs.' % len(PROFILES))


if __name__ == '__main__':
    main()
