#!/usr/bin/env python3
"""Generate editable H3 true-still graphs and matching API exports, offline."""
from pathlib import Path
import json
import build_h3_first_last_hq as base

base.RELEASE = '2026-10-20.1'
ROOT = Path(__file__).resolve().parents[1]
FIZGIG = '10d5171f6e45b0757a03df977f53a660e060c33d'
REF_PROMPT = '''<Picture 1> is the authoritative reference for the person and the photograph. Create one photographic still of that same person. Preserve their facial structure, apparent age, eye shape, nose, mouth, hairstyle, skin tone and body proportions. Keep the pose, framing, clothing, expression, background, camera perspective, lighting and color mood consistent with <Picture 1>. Render coherent photographic detail across the entire image, with natural skin texture, fabric and hair. Do not beautify, change the face, or substitute a different person. One image, not a collage or split screen.

Requested change: improve overall photographic clarity without changing the scene or the person's identity.'''
TEXT_PROMPT = '''A realistic editorial photograph of a person seated beside a large window in a quiet modern studio, natural side lighting, coherent anatomy, expressive eyes, realistic skin texture, detailed hair and woven fabric, restrained color grading, balanced composition, subtle depth of field. One photographic still. No collage, captions or graphic overlays.'''


def build(label, megapixels, text_only=False):
    g = base.Graph('HSTXT' if text_only else 'HSTILL_' + label)
    g.add(1, 'UNETLoader', 'Full BF16 FL2VA' if text_only else 'Full BF16 Ref2VA - your existing reference model', (-1500, 890),
          outputs=[('MODEL', 'MODEL')], widgets={'unet_name': 'minimax_h3_fl2va_bf16.safetensors' if text_only else 'minimax_h3_ref2va_bf16.safetensors', 'weight_dtype': 'default'}, size=(520, 100))
    g.add(2, 'CLIPLoader', 'H3-specific Qwen3-VL 32B BF16', (-1500, 1050), outputs=[('CLIP', 'CLIP')],
          widgets={'clip_name': 'qwen3vl_32b_minimax_h3_bf16.safetensors', 'type': 'minimax', 'device': 'default'}, size=(520, 135))
    g.add(3, 'VAELoader', 'Original FP16 video VAE - not a finetuned T1 decoder', (-910, 1050), outputs=[('VAE', 'VAE')],
          widgets={'vae_name': 'minimax_h3_video_vae_fp16.safetensors'}, size=(485, 100))
    if not text_only:
        g.add(4, 'LoadImage', '1. ONE authoritative reference image', (-1500, 25), outputs=[('IMAGE', 'IMAGE'), ('MASK', 'MASK')],
              widgets={'image': 'identity_reference.png', 'upload': 'image'}, size=(450, 510))
    g.add(5, 'WBH3StillSize', '2. Output resolution - shared with BOTH conditioning and latent', (-1500, 620),
          inputs=[('reference', 'IMAGE')], outputs=[('width', 'INT'), ('height', 'INT'), ('size_report', 'STRING')],
          widgets={'target_megapixels': megapixels, 'aspect_ratio': '16:9' if text_only else 'source'}, size=(460, 190))
    if not text_only:
        g.connect(4, 0, 5, 'reference')
    inputs = [('clip', 'CLIP'), ('vae', 'VAE')]
    if text_only:
        inputs += [('first_frame', 'IMAGE'), ('last_frame', 'IMAGE')]
    else:
        inputs += [('audio_vae', 'VAE'), ('ref_images.ref_image_0', 'IMAGE')]
    inputs += [('width', 'INT', 'width'), ('height', 'INT', 'height')]
    widgets = {'prompt': TEXT_PROMPT if text_only else REF_PROMPT, 'width': 1344, 'height': 768, 'length': 5}
    if not text_only:
        widgets['ref_image_size'] = 'max'
    g.add(6, 'MiniMaxH3ImageToVideo' if text_only else 'MiniMaxH3ReferenceToVideo',
          '3. Edit prompt - CONDITIONING only, leave its LATENT unused', (-910, 25), inputs=inputs,
          outputs=[('positive', 'CONDITIONING'), ('LATENT', 'LATENT')], widgets=widgets, size=(550, 650))
    g.connect(2, 0, 6, 'clip'); g.connect(3, 0, 6, 'vae')
    if not text_only:
        g.connect(4, 0, 6, 'ref_images.ref_image_0')
    g.add(7, 'FizgigH3StillLatent', '4. TRUE SINGLE FRAME - batch 1', (-910, 760),
          inputs=[('width', 'INT', 'width'), ('height', 'INT', 'height')], outputs=[('LATENT', 'LATENT')],
          widgets={'width': 1344, 'height': 768, 'batch_size': 1}, size=(425, 190))
    for target in (6, 7):
        g.connect(5, 0, target, 'width'); g.connect(5, 1, target, 'height')
    g.add(8, 'MiniMaxH3SigmaShift', 'Native shifts - 12 video / 3 audio', (-290, 910),
          inputs=[('model', 'MODEL')], outputs=[('MODEL', 'MODEL')], widgets={'shift_video': 12.0, 'shift_audio': 3.0}, size=(335, 135))
    g.connect(1, 0, 8, 'model')
    g.add(9, 'BasicGuider', 'Native guidance - no extra CFG', (-270, 25),
          inputs=[('model', 'MODEL'), ('conditioning', 'CONDITIONING')], outputs=[('GUIDER', 'GUIDER')], size=(330, 110))
    g.connect(8, 0, 9, 'model'); g.connect(6, 0, 9, 'conditioning')
    g.add(10, 'RandomNoise', 'Fixed seed for repeatable A/B settings comparisons', (-270, 200),
          outputs=[('NOISE', 'NOISE')], widgets={'noise_seed': 42, 'control_after_generate': 'fixed'}, size=(335, 115))
    g.add(11, 'KSamplerSelect', 'Maintainer still example: er_sde', (-270, 395), outputs=[('SAMPLER', 'SAMPLER')],
          widgets={'sampler_name': 'er_sde'}, size=(335, 90))
    g.add(12, 'BasicScheduler', '50 steps - simple - denoise 1 - NO Turbo LoRA', (-270, 590), inputs=[('model', 'MODEL')],
          outputs=[('SIGMAS', 'SIGMAS')], widgets={'scheduler': 'simple', 'steps': 50, 'denoise': 1.0}, size=(335, 155))
    g.connect(8, 0, 12, 'model')
    g.add(13, 'SamplerCustomAdvanced', '5. Generate one still, not a video clip', (160, 25),
          inputs=[('noise', 'NOISE'), ('guider', 'GUIDER'), ('sampler', 'SAMPLER'), ('sigmas', 'SIGMAS'), ('latent_image', 'LATENT')],
          outputs=[('output', 'LATENT'), ('denoised_output', 'LATENT')], size=(350, 260))
    for source, name in ((10, 'noise'), (9, 'guider'), (11, 'sampler'), (12, 'sigmas'), (7, 'latent_image')):
        g.connect(source, 0, 13, name)
    g.add(14, 'VAEDecode', '6. CURRENT native decode - five-copy T=1 fix is built into this core', (160, 430),
          inputs=[('samples', 'LATENT'), ('vae', 'VAE')], outputs=[('IMAGE', 'IMAGE')], size=(405, 110))
    g.connect(13, 0, 14, 'samples'); g.connect(3, 0, 14, 'vae')
    g.add(15, 'WBH3SaveStillMaster', '7. Lossless RGB16 PNG master', (670, 25), inputs=[('images', 'IMAGE')], outputs=[('images', 'IMAGE')],
          widgets={'filename_prefix': 'h3_still/' + ('Text_8MP' if text_only else 'Reference_' + label)}, size=(530, 580))
    g.connect(14, 0, 15, 'images')
    g.add(16, 'SaveImage', '8. Standard 8-bit PNG copy for compatibility', (1275, 25), inputs=[('images', 'IMAGE')], outputs=[],
          widgets={'filename_prefix': 'h3_still/preview/' + ('Text_8MP' if text_only else 'Reference_' + label)}, size=(530, 580))
    g.connect(15, 0, 16, 'images')
    g.note(20, 'READ BEFORE RUNNING',
           ('TRUE H3 STILL: one sampled temporal latent frame. The native conditioning node still shows length=5; its LATENT output is intentionally unused. No audio decoder or frame extraction is used.\n\n'
            'IMPORTANT CURRENT-CORE UPDATE: the pinned native H3 VAE already repeats a T=1 latent into five copies and keeps pixel frame 3. Use native VAEDecode here. The old Fizgig Still Decode is redundant on this pin.\n\nThe current best-documented high-resolution example uses FL2VA, 50 steps, er_sde/simple, and no Turbo. This graph keeps full BF16 weights and the FP16 video VAE. These precision changes are not benchmark-proven improvements.\n\n'
            + ('Text-only uses the separate FL2VA checkpoint. Prepare HSTXT assets before running. ' if text_only else
               'Reference stills use your existing Ref2VA checkpoint and one authoritative reference. 2.5 MP is the edit baseline; 8.4 MP is a higher-resolution adaptation, not a verified identity upgrade. Multiple-reference still edits have an unresolved report dated 2026-10-20. ')
            + '\n\nSource size preserves framing approximately after 32-pixel rounding. Up to 4096 per side is accepted by the node, not a demonstrated quality ceiling. No generative upscale is applied.\n\n'
            'The reference is conditioning, not a pixel lock. Keep denoise=1: lowering denoise on an empty latent does not preserve the original photograph.\n\n'
            'All new workflows are source-reviewed and offline-tested only. Docker/GPU inference and face fidelity still need live validation. No promise of exact identity.'),
           (670, 735), (1140, 570))
    for ident in (7,):
        g.node(ident)['properties'].update(cnr_id='comfyui-fizgig-h3-still', aux_id='shootthesound/ComfyUI-Fizgig-H3-Still', ver=FIZGIG)
    folder = 'HSTXT_Text_Still_HQ' if text_only else 'HSTILL_Reference_Still_HQ'
    filename = '01_H3_Text_Still_Native_8MP.json' if text_only else ('01_H3_Reference_Still_Identity_2p5MP.json' if megapixels == 2.5 else '02_H3_Reference_Still_Native_8MP.json')
    path = f'workflows/h3/{folder}/{filename}'
    api = f'automation/{"HSTXT" if text_only else "HSTILL"}/{filename.replace(".json", ".api.json")}'
    result = g.export(path, api)
    result['extra']['workbench'].update(source_validation='source-reviewed; graph, helper and mocked-interface tests only', true_single_frame=True,
                                        upstream_still_commit=FIZGIG, reference_count=0 if text_only else 1,
                                        max_resolution_evidence='8 MP FL2VA maintainer example; Ref2VA 8 MP is an adaptation')
    (ROOT / path).write_text(json.dumps(result, indent=2) + '\n')
    return path


if __name__ == '__main__':
    for label, mp, txt in [('2p5MP', 2.5, False), ('8MP', 8.4, False), ('8MP', 8.4, True)]:
        print(build(label, mp, txt))
