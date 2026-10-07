"""Deterministic UI workflows and API graphs; explicit widget order, no missing LoRA."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent


class Graph:
    def __init__(self, name):
        self.nodes, self.links, self.groups = [], [], []
        self.name = name

    def add(self, kind, title, pos, size, ins=(), outs=(), widgets=(), names=()):
        nid = len(self.nodes) + 1
        self.nodes.append({'id': nid, 'type': kind, 'pos': list(pos), 'size': list(size),
                           'flags': {}, 'order': nid - 1, 'mode': 0,
                           'inputs': [{'name': n, 'type': t, 'link': None} for n, t in ins],
                           'outputs': [{'name': n, 'type': t, 'links': [], 'slot_index': i}
                                       for i, (n, t) in enumerate(outs)],
                           'title': title, 'properties': {'Node name for S&R': kind},
                           'widgets_values': list(widgets),
                           'widgets_values_named': dict(names)})
        return nid

    def link(self, src, slot, dst, name):
        a, b = self.nodes[src - 1], self.nodes[dst - 1]
        target_slot = next(i for i, v in enumerate(b['inputs']) if v['name'] == name)
        output, inp = a['outputs'][slot], b['inputs'][target_slot]
        assert output['type'] == inp['type'], (output, inp)
        lid = len(self.links) + 1
        output['links'].append(lid); inp['link'] = lid
        self.links.append([lid, src, slot, dst, target_slot, output['type']])

    def result(self):
        return {'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'flux-photo-1.0.0/' + self.name)),
                'revision': 0, 'last_node_id': len(self.nodes), 'last_link_id': len(self.links),
                'nodes': self.nodes, 'links': self.links, 'groups': self.groups,
                'config': {}, 'extra': {'ds': {'scale': 0.58, 'offset': [60, 70]},
                                       'flux_photo_version': '1.0.0'}, 'version': 0.4}


def api_graph(graph):
    links = {v[0]: v for v in graph['links']}
    result = {}
    for n in graph['nodes']:
        if n['type'] == 'Note':
            continue
        inputs = copy.deepcopy(n['widgets_values_named'])
        for inp in n['inputs']:
            if inp['link'] is not None:
                link = links[inp['link']]
                inputs[inp['name']] = [str(link[1]), link[2]]
        result[str(n['id'])] = {'class_type': n['type'], 'inputs': inputs, '_meta': {'title': n['title']}}
    return result


def build(native=False):
    name = 'FLUX_Photo_02_Native_3072_v1_0' if native else 'FLUX_Photo_01_Quality_Fit_2048_v1_0'
    g = Graph(name)
    source = g.add('LoadImage', '1 | SOURCE PHOTO - paint mask here', (0, 200), (350, 440),
                   outs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')], widgets=['', 'image'], names=[('image', '')])
    ref = g.add('LoadImage', '2 | REPLACEMENT REFERENCE - no mask needed', (0, 720), (350, 440),
                outs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')], widgets=['', 'image'], names=[('image', '')])
    model = g.add('FluxPhotoModels', '9B DISTILLED BF16 | fixed matching models', (440, 200), (380, 130),
                  outs=[('MODEL', 'MODEL'), ('CLIP', 'CLIP'), ('VAE', 'VAE')],
                  widgets=['cpu'], names=[('text_encoder_device', 'cpu')])
    previous = model
    for i in range(3):
        lora = g.add('FluxPhotoOptionalLoRA', f'Optional LoRA {i + 1} | None = off', (440, 390 + i * 180), (380, 125),
                     ins=[('model', 'MODEL')], outs=[('MODEL', 'MODEL')], widgets=['None', 0.75],
                     names=[('lora_name', 'None'), ('strength', 0.75)])
        g.link(previous, 0, lora, 'model'); previous = lora
    text = ('Replace the [describe the target] in image 1 with the [person or object] from image 2. '
            'Preserve the reference identity, facial features, proportions, distinctive details and colors. '
            'Use image 1 for pose, camera angle, perspective and scene lighting. '
            'Keep the surrounding photo unchanged.\n\nReplace this example with your own instruction.')
    prompt = g.add('CLIPTextEncode', '3 | YOUR EDIT PROMPT - image 1 = crop; image 2 = reference',
                   (890, 200), (530, 290), ins=[('clip', 'CLIP')], outs=[('CONDITIONING', 'CONDITIONING')],
                   widgets=[text], names=[('text', text)])
    g.link(model, 1, prompt, 'clip')
    mode, cap = ('native', 3072) if native else ('fit', 2048)
    crop = g.add('FluxPhotoCrop', '4 | INPAINT CROP - native is experimental' if native else '4 | INPAINT CROP - only this area is resized',
                 (890, 590), (530, 250), ins=[('source', 'IMAGE'), ('mask', 'MASK')],
                 outs=[('stitch_data', 'FLUX_PHOTO_CROP'), ('cropped_image', 'IMAGE'), ('sampling_mask', 'MASK')],
                 widgets=[mode, cap, 128, 16], names=[('processing', mode), ('max_crop_side', cap),
                                                    ('context_pixels', 128), ('sampling_mask_grow', 16)])
    g.link(source, 0, crop, 'source'); g.link(source, 1, crop, 'mask')
    prep = g.add('FluxPhotoReferenceConditioning', '5 | TWO IMAGE REFERENCES + MASKED LATENT', (1490, 200), (450, 320),
                 ins=[('conditioning', 'CONDITIONING'), ('vae', 'VAE'), ('cropped_image', 'IMAGE'),
                      ('sampling_mask', 'MASK'), ('replacement_reference', 'IMAGE')],
                 outs=[('two_image_conditioning', 'CONDITIONING'), ('masked_source_latent', 'LATENT')],
                 widgets=[1024, 2048, 1024, 128], names=[('scene_reference_long_side', 1024),
                      ('replacement_reference_long_side', 2048), ('vae_tile_size', 1024), ('vae_overlap', 128)])
    for a, slot, field in [(prompt,0,'conditioning'),(model,2,'vae'),(crop,1,'cropped_image'),
                            (crop,2,'sampling_mask'),(ref,0,'replacement_reference')]:
        g.link(a, slot, prep, field)
    sampler = g.add('FluxPhotoSampler', '6 | DISTILLED - Euler / CFG 1 / start at 4 steps', (1490, 620), (450, 200),
                    ins=[('model','MODEL'),('conditioning','CONDITIONING'),('latent','LATENT')], outs=[('LATENT','LATENT')],
                    widgets=[12345,'fixed',4], names=[('seed',12345),('steps',4)])
    g.link(previous,0,sampler,'model');g.link(prep,0,sampler,'conditioning');g.link(prep,1,sampler,'latent')
    decode = g.add('VAEDecodeTiled', '7 | TILED VAE DECODE - not tiled diffusion', (2010, 200), (380, 220),
                   ins=[('samples','LATENT'),('vae','VAE')], outs=[('IMAGE','IMAGE')],
                   widgets=[1024,128,64,8], names=[('tile_size',1024),('overlap',128),('temporal_size',64),('temporal_overlap',8)])
    g.link(sampler,0,decode,'samples');g.link(model,2,decode,'vae')
    stitch = g.add('FluxPhotoStitch', '8 | PROTECTED STITCH - original photo dimensions', (2010, 540), (380, 210),
                   ins=[('stitch_data','FLUX_PHOTO_CROP'),('generated_crop','IMAGE')],
                   outs=[('original_size_result','IMAGE'),('verification_report','STRING')], widgets=[16,0.0],
                   names=[('feather_pixels',16),('boundary_color_strength',0.0)])
    g.link(crop,0,stitch,'stitch_data');g.link(decode,0,stitch,'generated_crop')
    compare = g.add('FluxPhotoCompare', '9 | ORIGINAL / OUTPUT - preview only', (2470,200), (700,600),
                    ins=[('original','IMAGE'),('edited','IMAGE'),('report','STRING')], widgets=[2048], names=[('preview_long_side',2048)])
    g.link(source,0,compare,'original');g.link(stitch,0,compare,'edited');g.link(stitch,1,compare,'report')
    save = g.add('FluxPhotoSave', '10 | FULL-SIZE LOSSLESS PNG + VERIFICATION JSON', (2470,920), (550,260),
                 ins=[('images','IMAGE'),('report','STRING')], widgets=['FLUX_Photo/Edited'], names=[('filename_prefix','FLUX_Photo/Edited')])
    g.link(stitch,0,save,'images');g.link(stitch,1,save,'report')
    note = ('USE sRGB 8-bit photos. Source and output dimensions are identical; no full-image diffusion or upscaler.\n'
            'Right-click SOURCE > Open in MaskEditor > paint > Save. White mask is editable.\n'
            'Image 1 in the prompt is the cropped scene. Image 2 is your replacement reference.\n'
            'LoRAs must be FLUX.2 klein 9B compatible. None works without installing a LoRA.\n'
            'FIT 2048: only the crop may be resized, then fitted back. NATIVE 3072: no crop resampling; experimental above ~4MP.\n'
            '2500px mask + 128px context on each side gives ~2768px padded crop. Native mode errors rather than silently shrinking.\n'
            'Tiled VAE saves VAE memory only. Final identity, color and synthesized detail still need visual review.')
    g.add('Note','READ FIRST', (0,-190), (1420,250), widgets=[note])
    g.groups = [{'title':'INPUTS | SOURCE + REFERENCE','bounding':[-30,140,410,1090],'color':'#3f5159','font_size':24},
                {'title':'FULL-PRECISION MODELS + OPTIONAL LoRAs','bounding':[410,140,440,950],'color':'#495346','font_size':24},
                {'title':'YOUR PROMPT + CROP CONTROLS','bounding':[860,140,590,950],'color':'#555047','font_size':24},
                {'title':'REFERENCE-GUIDED GENERATION','bounding':[1460,140,510,950],'color':'#4a4d5b','font_size':24},
                {'title':'RESTORE ORIGINAL CANVAS','bounding':[1980,140,440,950],'color':'#47594c','font_size':24},
                {'title':'COMPARE + EXPORT','bounding':[2440,140,770,1130],'color':'#574b50','font_size':24}]
    return name, g.result()


def main():
    out = HERE/'workflows'; out.mkdir(exist_ok=True)
    for native in (False,True):
        name,graph = build(native)
        (out/(name+'.json')).write_text(json.dumps(graph,indent=2)+'\n')
        (HERE/'config'/(name+'.api.json')).write_text(json.dumps(api_graph(graph),indent=2)+'\n')
        print(name, len(graph['nodes']), 'nodes', len(graph['links']), 'links')

if __name__=='__main__': main()
