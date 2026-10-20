#!/usr/bin/env python3
"""Rebuild the explicit native H3 first/last-frame graphs. No downloads."""
from __future__ import annotations
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
CORE = 'b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c'
RELEASE = '2026-10-19.1'
FOLDER = 'workflows/h3/HFL_First_Last_Identity_HQ'
FINISH = 'workflows/h3/HFL4K_Identity_Finish'
PROMPT = '''<Picture 1> is the authoritative appearance and identity reference for the single subject, and shows the opening composition. <Picture 2> shows the desired ending pose and composition of that same subject. Keep the subject recognizably the same person throughout: the same facial structure, eyes, nose, mouth, hair, skin texture and body proportions. Keep the outfit, lighting and setting consistent with the endpoint images.

One continuous shot. The subject makes a small, natural movement from the opening pose to the ending pose, with subtle breathing and blinking. The face remains clearly visible. The camera is steady. Preserve natural photographic detail and coherent motion. No cuts, morphing, face replacement or sudden changes in appearance. Natural ambient sound; no dialogue or music.'''


class Graph:
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.links = []
        self.api = {}

    def add(self, ident, kind, title, pos, inputs=(), outputs=(), widgets=None, size=(320, 180), virtual=False):
        widgets = widgets or {}
        ports = []
        for value in inputs:
            if len(value) == 2:
                name, typ = value
                port = {'name': name, 'type': typ, 'link': None}
            else:
                name, typ, widget = value
                port = {'name': name, 'type': typ, 'link': None, 'widget': {'name': widget}}
            ports.append(port)
        node = {'id': ident, 'type': kind, 'pos': list(pos), 'size': list(size), 'flags': {},
                'order': len(self.nodes), 'mode': 0, 'inputs': ports,
                'outputs': [{'name': n, 'type': t, 'links': []} for n, t in outputs],
                'title': title, 'properties': {'Node name for S&R': kind},
                'widgets_values': list(widgets.values()), 'widgets_values_named': dict(widgets)}
        if not kind.startswith('WB') and not virtual:
            node['properties'].update(cnr_id='comfy-core', ver=CORE)
        self.nodes.append(node)
        if not virtual:
            self.api[str(ident)] = {'class_type': kind, 'inputs': {
                k: v for k, v in widgets.items() if k not in ('control_after_generate', 'upload')},
                '_meta': {'title': title}}
        return ident

    def node(self, ident):
        return next(n for n in self.nodes if n['id'] == ident)

    def connect(self, source, slot, target, name):
        src, dst = self.node(source), self.node(target)
        target_slot = next(i for i, p in enumerate(dst['inputs']) if p['name'] == name)
        port = dst['inputs'][target_slot]
        if port['link'] is not None:
            raise ValueError('Duplicate incoming link.')
        typ = src['outputs'][slot]['type']
        if typ != port['type']:
            raise ValueError(f'Type mismatch {typ} -> {port["type"]}')
        ident = len(self.links) + 1
        self.links.append([ident, source, slot, target, target_slot, typ])
        src['outputs'][slot]['links'].append(ident)
        port['link'] = ident
        self.api[str(target)]['inputs'][name] = [str(source), slot]

    def note(self, ident, title, text, pos, size):
        return self.add(ident, 'MarkdownNote', title, pos, widgets={'text': text}, size=size, virtual=True)

    def export(self, filename, api_filename):
        from graphlib import TopologicalSorter
        deps = {n['id']: set() for n in self.nodes}
        for _, src, _, dst, _, _ in self.links:
            deps[dst].add(src)
        for order, ident in enumerate(TopologicalSorter(deps).static_order()):
            self.node(ident)['order'] = order
        result = {'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'h3-hq/' + self.name + '/' + RELEASE)),
                  'revision': 0, 'last_node_id': max(n['id'] for n in self.nodes),
                  'last_link_id': len(self.links), 'nodes': self.nodes, 'links': self.links,
                  'groups': [], 'config': {}, 'extra': {
                      'ds': {'scale': 0.55, 'offset': [1940, 130]},
                      'workbench': {'task': self.name, 'release': RELEASE,
                                    'source_validation': 'static + local media round-trip only',
                                    'gpu_tested': False, 'comfy_commit': CORE}}, 'version': 0.4}
        for file, content in ((filename, result), (api_filename, self.api)):
            path = ROOT / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')
        return result


def generation(width, height, label, filename):
    g = Graph('HFL_' + label)
    g.add(1, 'UNETLoader', 'Full BF16 Ref2VA - no turbo or quantization', (-1800, 1080),
          outputs=[('MODEL', 'MODEL')], widgets={'unet_name': 'minimax_h3_ref2va_bf16.safetensors',
                                               'weight_dtype': 'default'}, size=(510, 105))
    g.add(2, 'CLIPLoader', 'H3-specific Qwen3-VL 32B BF16', (-1800, 1240),
          outputs=[('CLIP', 'CLIP')], widgets={'clip_name': 'qwen3vl_32b_minimax_h3_bf16.safetensors',
                                             'type': 'minimax', 'device': 'default'}, size=(510, 130))
    g.add(3, 'VAELoader', 'Native FP16 video VAE', (-1230, 1080), outputs=[('VAE', 'VAE')],
          widgets={'vae_name': 'minimax_h3_video_vae_fp16.safetensors'}, size=(480, 90))
    g.add(4, 'VAELoader', 'Native FP32 audio VAE', (-1230, 1240), outputs=[('VAE', 'VAE')],
          widgets={'vae_name': 'minimax_h3_audio_vae_fp32.safetensors'}, size=(480, 90))
    g.add(5, 'LoadImage', '1. FIRST FRAME - primary subject identity', (-1800, 30),
          outputs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')], widgets={'image': 'first_frame.png', 'upload': 'image'}, size=(420, 430))
    g.add(6, 'LoadImage', '2. LAST FRAME - same person, ending pose', (-1800, 560),
          outputs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')], widgets={'image': 'last_frame.png', 'upload': 'image'}, size=(420, 430))
    g.add(7, 'MiniMaxH3SigmaShift', 'Native H3 flow shifts', (-600, 1080),
          inputs=[('model', 'MODEL')], outputs=[('MODEL', 'MODEL')],
          widgets={'shift_video': 12.0, 'shift_audio': 3.0}, size=(280, 115))
    g.connect(1, 0, 7, 'model')
    g.add(10, 'MiniMaxH3ReferenceToVideo', '3. IDENTITY + MOTION - edit prompt here', (-1240, 30),
          inputs=[('clip', 'CLIP'), ('vae', 'VAE'), ('audio_vae', 'VAE'),
                  ('ref_images.ref_image_0', 'IMAGE'), ('ref_images.ref_image_1', 'IMAGE'),
                  ('ref_images.ref_image_2', 'IMAGE')],
          outputs=[('positive', 'CONDITIONING'), ('LATENT', 'LATENT')],
          widgets={'prompt': PROMPT, 'width': width, 'height': height, 'length': 124,
                   'ref_image_size': 'max'}, size=(580, 660))
    g.connect(2, 0, 10, 'clip')
    g.connect(3, 0, 10, 'vae')
    g.connect(5, 0, 10, 'ref_images.ref_image_0')
    g.connect(6, 0, 10, 'ref_images.ref_image_1')
    g.add(11, 'MiniMaxH3AddGuide', '4. Anchor FIRST frame at index 0', (-570, 45),
          inputs=[('positive', 'CONDITIONING'), ('vae', 'VAE'), ('audio_vae', 'VAE'),
                  ('latent', 'LATENT'), ('image', 'IMAGE'), ('audio', 'AUDIO')],
          outputs=[('positive', 'CONDITIONING')], widgets={'frame_idx': 0}, size=(360, 220))
    g.add(12, 'MiniMaxH3AddGuide', '5. Anchor LAST frame at index -1', (-570, 355),
          inputs=[('positive', 'CONDITIONING'), ('vae', 'VAE'), ('audio_vae', 'VAE'),
                  ('latent', 'LATENT'), ('image', 'IMAGE'), ('audio', 'AUDIO')],
          outputs=[('positive', 'CONDITIONING')], widgets={'frame_idx': -1}, size=(360, 220))
    for target, image in ((11, 5), (12, 6)):
        g.connect(3, 0, target, 'vae')
        g.connect(10, 1, target, 'latent')
        g.connect(image, 0, target, 'image')
    g.connect(10, 0, 11, 'positive')
    g.connect(11, 0, 12, 'positive')
    g.add(13, 'BasicGuider', 'Native CFG-distilled guidance - no extra CFG', (-65, 30),
          inputs=[('model', 'MODEL'), ('conditioning', 'CONDITIONING')], outputs=[('GUIDER', 'GUIDER')], size=(345, 90))
    g.connect(7, 0, 13, 'model')
    g.connect(12, 0, 13, 'conditioning')
    g.add(14, 'RandomNoise', 'Fixed seed for controlled comparisons', (-65, 200), outputs=[('NOISE', 'NOISE')],
          widgets={'noise_seed': 20261019, 'control_after_generate': 'fixed'}, size=(340, 110))
    g.add(15, 'KSamplerSelect', 'Full-step sampler', (-65, 370), outputs=[('SAMPLER', 'SAMPLER')],
          widgets={'sampler_name': 'res_multistep'}, size=(340, 85))
    g.add(16, 'BasicScheduler', '30-step quality baseline - not turbo', (-65, 505), inputs=[('model', 'MODEL')],
          outputs=[('SIGMAS', 'SIGMAS')], widgets={'scheduler': 'simple', 'steps': 30, 'denoise': 1.0}, size=(340, 135))
    g.connect(7, 0, 16, 'model')
    g.add(17, 'SamplerCustomAdvanced', '6. Generate native video + audio', (420, 60),
          inputs=[('noise', 'NOISE'), ('guider', 'GUIDER'), ('sampler', 'SAMPLER'),
                  ('sigmas', 'SIGMAS'), ('latent_image', 'LATENT')],
          outputs=[('output', 'LATENT'), ('denoised_output', 'LATENT')], size=(350, 210))
    for source, name in ((14, 'noise'), (13, 'guider'), (15, 'sampler'), (16, 'sigmas')):
        g.connect(source, 0, 17, name)
    g.connect(10, 1, 17, 'latent_image')
    g.add(18, 'VAEDecode', 'Decode video from joint AV latent', (420, 370),
          inputs=[('samples', 'LATENT'), ('vae', 'VAE')], outputs=[('IMAGE', 'IMAGE')], size=(350, 90))
    g.add(19, 'VAEDecodeAudio', 'Decode original generated audio', (420, 550),
          inputs=[('samples', 'LATENT'), ('vae', 'VAE')], outputs=[('AUDIO', 'AUDIO')], size=(350, 90))
    g.connect(17, 0, 18, 'samples')
    g.connect(17, 0, 19, 'samples')
    g.connect(3, 0, 18, 'vae')
    g.connect(4, 0, 19, 'vae')
    g.add(20, 'WBH3LosslessMaster', '7. SAVE RGB16 MASTER - source for 4K finish', (915, 50),
          inputs=[('images', 'IMAGE'), ('audio', 'AUDIO')],
          outputs=[('images', 'IMAGE'), ('master_file', 'STRING')],
          widgets={'fps': 24.0, 'filename_prefix': 'video/H3_First_Last/native/H3_' + label}, size=(450, 160))
    g.connect(18, 0, 20, 'images')
    g.connect(19, 0, 20, 'audio')
    g.add(21, 'CreateVideo', '8. Create browser preview at native 24 fps', (915, 320),
          inputs=[('images', 'IMAGE'), ('audio', 'AUDIO')], outputs=[('VIDEO', 'VIDEO')],
          widgets={'fps': 24.0, 'bit_depth': 8, 'color_space': 'sRGB', 'codec': 'none'}, size=(420, 190))
    g.connect(20, 0, 21, 'images')
    g.connect(19, 0, 21, 'audio')
    g.add(22, 'SaveVideo', 'MP4 PREVIEW - keep the MKV master for finishing', (1450, 60),
          inputs=[('video', 'VIDEO')], outputs=[('video', 'VIDEO')],
          widgets={'filename_prefix': 'video/H3_First_Last/previews/H3_' + label,
                   'format': 'mp4', 'format.codec': 'auto'}, size=(670, 640))
    g.connect(21, 0, 22, 'video')
    g.note(90, 'READ FIRST - native first/last + identity references',
           '## First / last frame + identity\n1. Upload the first and last images. Use the same person, compatible clothes, lighting and framing.\n2. Edit the motion paragraph in node 10. Start with a small movement and a steady camera.\n3. Keep `max` references, 30 steps and 124 frames for the first run.\n4. Queue. The workflow saves an RGB16 FFV1 MKV master plus an MP4 preview.\n5. Review the face throughout, not only the endpoints. Open HFL4K only after approving the native result.\n\n**This is Ref2VA + two native Add Guide nodes.** The first photo enters both the reference path and frame 0. The last photo enters both the reference path and frame -1. These are generation constraints, not exact-pixel guarantees. No source frames are pasted over the generated endpoints.\n\n**Optional face reference:** add a Load Image node and connect its IMAGE output to `ref_images.ref_image_2`. Append: `<Picture 3> shows the same subject; use it only to reinforce facial identity, not pose, outfit, lighting or background.` No third upload is required by default.',
           (-2370, 35), (500, 790))
    g.note(91, 'Quality / resolution / validation',
           f'## Quality baseline\nNative canvas: **{width} x {height}**. 124 frames at 24 fps is **5.167 seconds**, not exactly 5 seconds. Other valid lengths: 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345, 362.\n\nReference `max` keeps up to a 2048-pixel short edge and does not upscale small references. It is slower and uses more memory. The native 768 x 1344 canvas is approximately 1 MP; arbitrary 2K/4K diffusion is not a validated quality upgrade. Add Guide cover-crops to the canvas, so preview endpoint framing before an expensive run.\n\nFull BF16 weights, native VAEs, no added turbo LoRA, no sparse attention, no face restoration. H3 launch uses dense PyTorch attention.\n\n30 steps is a starting point, not a measured optimum. Compare 40 or 50 with the SAME seed and inputs only after the baseline looks right. `beta` / `normal` are optional scheduler A/B tests, not guarantees of improvement.\n\nSource-pinned build. Offline graph and media checks only; Docker, real ComfyUI import and GPU quality still require live validation. Only your SAM local-edit workflow retains approved status.',
           (-2370, 890), (500, 720))
    return g.export(FOLDER + '/' + filename, 'automation/HFL/' + filename.replace('.json', '.api.json'))


def finishing():
    g = Graph('HFL4K')
    g.add(1, 'WBH3LoadNativeMaster', '1. Select APPROVED NATIVE MKV - not MP4 / 4K', (-1360, 80),
          outputs=[('images', 'IMAGE'), ('audio', 'AUDIO'), ('fps', 'FLOAT')],
          widgets={'master_file': 'Select a native H3 MKV master after generation'}, size=(650, 100))
    g.add(2, 'WBH3ReleaseMemory', '2. Offload H3 before loading SeedVR2', (-1360, 310),
          inputs=[('images', 'IMAGE')], outputs=[('IMAGE', 'IMAGE')], size=(400, 85))
    g.connect(1, 0, 2, 'images')
    g.add(3, 'SeedVR2LoadDiTModel', 'Full 7B FP16 - no sharp / FP8 model', (-1360, 525),
          inputs=[('torch_compile_args', 'TORCH_COMPILE_ARGS')], outputs=[('SEEDVR2_DIT', 'SEEDVR2_DIT')],
          widgets={'model': 'seedvr2_ema_7b_fp16.safetensors', 'device': 'cuda:0', 'blocks_to_swap': 0,
                   'swap_io_components': False, 'offload_device': 'cpu', 'cache_model': False,
                   'attention_mode': 'sdpa'}, size=(490, 285))
    g.add(4, 'SeedVR2LoadVAEModel', 'Full FP16 VAE - tiled decode', (-1360, 920),
          inputs=[('torch_compile_args', 'TORCH_COMPILE_ARGS')], outputs=[('SEEDVR2_VAE', 'SEEDVR2_VAE')],
          widgets={'model': 'ema_vae_fp16.safetensors', 'device': 'cuda:0', 'encode_tiled': True,
                   'encode_tile_size': 1024, 'encode_tile_overlap': 128, 'decode_tiled': True,
                   'decode_tile_size': 1024, 'decode_tile_overlap': 128, 'tile_debug': 'false',
                   'offload_device': 'cpu', 'cache_model': False}, size=(490, 375))
    for ident in (3, 4):
        g.node(ident)['properties'] = {'Node name for S&R': g.node(ident)['type']}
    g.add(5, 'SeedVR2VideoUpscaler', '3. 4K-CLASS FINISH - zero added noise', (-660, 100),
          inputs=[('image', 'IMAGE'), ('dit', 'SEEDVR2_DIT'), ('vae', 'SEEDVR2_VAE')],
          outputs=[('IMAGE', 'IMAGE')],
          widgets={'seed': 42, 'control_after_generate': 'fixed', 'resolution': 2160,
                   'max_resolution': 3840, 'batch_size': 9, 'uniform_batch_size': True,
                   'temporal_overlap': 4, 'prepend_frames': 4, 'color_correction': 'lab',
                   'input_noise_scale': 0.0, 'latent_noise_scale': 0.0,
                   'offload_device': 'cpu', 'enable_debug': False}, size=(470, 490))
    g.node(5)['properties'] = {'Node name for S&R': 'SeedVR2VideoUpscaler'}
    g.connect(2, 0, 5, 'image')
    g.connect(3, 0, 5, 'dit')
    g.connect(4, 0, 5, 'vae')
    g.add(6, 'WBH3LosslessMaster', '4. Save separate RGB16 upscaled master', (15, 100),
          inputs=[('images', 'IMAGE'), ('audio', 'AUDIO'), ('fps', 'FLOAT', 'fps')],
          outputs=[('images', 'IMAGE'), ('master_file', 'STRING')],
          widgets={'fps': 24.0, 'filename_prefix': 'video/H3_First_Last/4k/H3_4K_finish'}, size=(500, 180))
    g.connect(5, 0, 6, 'images')
    g.connect(1, 1, 6, 'audio')
    g.connect(1, 2, 6, 'fps')
    g.add(7, 'CreateVideo', 'Keep original audio and frame rate', (15, 430),
          inputs=[('images', 'IMAGE'), ('audio', 'AUDIO'), ('fps', 'FLOAT', 'fps')], outputs=[('VIDEO', 'VIDEO')],
          widgets={'fps': 24.0, 'bit_depth': 8, 'color_space': 'sRGB', 'codec': 'none'}, size=(440, 190))
    g.connect(6, 0, 7, 'images')
    g.connect(1, 1, 7, 'audio')
    g.connect(1, 2, 7, 'fps')
    g.add(8, 'SaveVideo', '4K-class MP4 preview - master is saved separately', (710, 100),
          inputs=[('video', 'VIDEO')], outputs=[('video', 'VIDEO')],
          widgets={'filename_prefix': 'video/H3_First_Last/previews/H3_4K_finish', 'format': 'mp4', 'format.codec': 'auto'},
          size=(660, 710))
    g.connect(7, 0, 8, 'video')
    g.note(90, 'Finishing, not native 4K generation',
           '## Run only after approving the native clip\nRefresh ComfyUI after generating HFL, then choose its native MKV in node 1. The dropdown scans input and output storage. It does not require a re-upload.\n\n2160-pixel short edge, capped at 3840 pixels on the long edge, with the native aspect ratio preserved. For 1344 x 768 / 768 x 1344 masters, expect approximately **3780 x 2160 / 2160 x 3780**. This is 4K-class, not an exact 16:9 UHD crop. No stretch, synthetic frame-rate interpolation or source-frame insertion.\n\nThe upscaler uses nine-frame temporal batches, four-frame overlap, uniform last-batch padding, LAB color correction and zero additional input/latent noise. Keep the native master: generative upscaling can alter fine facial features even with noise controls at zero.\n\nCompare native and finished faces at the beginning, middle and end, and through motion. Reject the finish if identity is worse. Increase batch size to 13 or 17 only after a successful GPU test with enough headroom.\n\nNative MKV masters are FFV1 RGB16 with the generated audio retained; they may not play in a browser. The MP4 is the browser preview.\n\nPlanning target: H200-class 141 GB VRAM, 256 GB system RAM and 300 GB volume. Not a measured hardware minimum. No GPU inference has been certified.',
           (-2140, 80), (620, 940))
    return g.export(FINISH + '/01_SeedVR2_7B_Identity_4K_Finish.json',
                    'automation/HFL4K/01_SeedVR2_7B_Identity_4K_Finish.api.json')


def main():
    generation(768, 1344, 'portrait', '01_H3_First_Last_Identity_HQ_Portrait.json')
    generation(1344, 768, 'landscape', '02_H3_First_Last_Identity_HQ_Landscape.json')
    finishing()
    print('Built two native first/last-frame graphs and one independent 4K-class finishing graph.')


if __name__ == '__main__':
    main()
