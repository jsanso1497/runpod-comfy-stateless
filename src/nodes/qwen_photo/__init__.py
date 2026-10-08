"""Qwen Image 2.1 protected photo editing with optional native SAM 3.1 masks.

This extension delegates model inference to ComfyUI core. It adds deterministic
geometry, ordered reference inputs, lazy mask review, and protected PNG export.
"""
from __future__ import annotations
from workbench.registry import choices as model_choices, check_selection, lora_choices

import hashlib
import json
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image, ImageCms
from PIL.PngImagePlugin import PngInfo
from scipy.ndimage import maximum_filter
import torch

from .geometry import (crop_photo, fit_dimensions, image_check, mask_check,
                       prepare_reference, resize_rgb, stitch_photo, to_uint8)

VERSION = '2.0.0'
CATEGORY = 'Qwen 2.1 Photo Edit'
SAM_FILE = 'sam3.1_multiplex_fp16.safetensors'
PRESETS = {
    'BF16 quality': ('qwen_image_2.1_bf16.safetensors', 'qwen3vl_8b_bf16.safetensors'),
}
VAE_FILE = 'qwen_image_2.1_vae_bf16.safetensors'


def _tuple(output):
    if hasattr(output, 'result'):
        return output.result
    if isinstance(output, dict):
        return output['result']
    return output


def _manual(mask, height, width, required=False):
    if mask is None or not torch.any(mask > 0):
        if required:
            raise ValueError('Paint and SAVE a mask on SOURCE Load Image first. '
                             'An RGB image without a saved mask has a blank placeholder mask.')
        return None
    return mask_check(mask, height, width)


def choose_instances(masks, index):
    if not torch.is_tensor(masks) or masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError('SAM found no matching object. Revise the short mask prompt, '
                         'lower the threshold, use a painted search ROI, or use a manual mask.')
    if index < -1 or index >= masks.shape[0]:
        raise ValueError(f'SAM returned {masks.shape[0]} instance(s); choose -1 for all '
                         f'or an index from 0 to {masks.shape[0] - 1}.')
    out = masks.amax(dim=0, keepdim=True) if index == -1 else masks[index:index + 1]
    if not torch.any(out > 0):
        raise ValueError('SAM returned an empty mask. No full-image fallback is permitted.')
    return out.detach().float().cpu()


def combine_masks(selected, manual, operation, grow):
    if operation not in ('ignore', 'restrict to painted area', 'add painted area', 'subtract painted area'):
        raise ValueError('Unknown manual correction operation.')
    if operation != 'ignore' and manual is None:
        raise ValueError('The selected SAM correction requires a saved manual source mask.')
    if operation == 'add painted area':
        selected = torch.maximum(selected, manual)
    if grow:
        if grow < 0 or grow > 128:
            raise ValueError('Mask expansion must be 0 to 128 source pixels.')
        selected = torch.from_numpy(maximum_filter(selected[0].numpy(),
                     size=2 * grow + 1, mode='constant').copy()).unsqueeze(0)
    # Apply limits AFTER growth, so protected pixels cannot be grown back in.
    if operation == 'restrict to painted area':
        selected = selected * manual
    elif operation == 'subtract painted area':
        selected = selected * (1.0 - manual)
    return selected


class Q21PhotoMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source': ('IMAGE',),
            'auto_mask': ('BOOLEAN', {'default': False}),
            'sam_prompt': ('STRING', {'default': 'shirt sleeves:2', 'multiline': True}),
            'threshold': ('FLOAT', {'default': 0.5, 'min': 0.05, 'max': 0.95, 'step': 0.05}),
            'refine_iterations': ('INT', {'default': 2, 'min': 0, 'max': 5}),
            'instance_index': ('INT', {'default': -1, 'min': -1, 'max': 63}),
            'search_region': (['full photo', 'painted ROI'], {'default': 'full photo'}),
            'manual_correction': (['ignore', 'restrict to painted area', 'add painted area', 'subtract painted area'],),
            'expand_pixels': ('INT', {'default': 0, 'min': 0, 'max': 128}),
        }, 'optional': {'manual_mask': ('MASK',)}}
    RETURN_TYPES = ('MASK', 'STRING')
    RETURN_NAMES = ('effective_edit_mask', 'mask_report')
    FUNCTION = 'make_mask'
    CATEGORY = CATEGORY
    DESCRIPTION = ('OFF uses the saved source mask and never loads SAM. ON runs core SAM 3.1. '
                   'A painted ROI focuses small-object detection. Expansion changes the final edit boundary.')

    @torch.inference_mode()
    def make_mask(self, source, auto_mask=False, sam_prompt='shirt sleeves:2', threshold=0.5,
                  refine_iterations=2, instance_index=-1, search_region='full photo',
                  manual_correction='ignore', expand_pixels=0, manual_mask=None):
        image_check(source, 'source')
        height, width = source.shape[1:3]
        manual = _manual(manual_mask, height, width, required=not auto_mask)
        info = {'auto_mask': auto_mask, 'expand_pixels': expand_pixels}
        if not auto_mask:
            selected = combine_masks(manual, None, 'ignore', expand_pixels)
        else:
            if not sam_prompt.strip():
                raise ValueError('Auto mask needs a short SAM object prompt, not the edit instruction.')
            import folder_paths
            import nodes
            try:
                from comfy_extras.nodes_sam3 import SAM3_Detect
            except ImportError as exc:
                raise RuntimeError('Native SAM3_Detect is missing. Update/rebuild ComfyUI with '
                                   'SAM 3.1 support; manual masking still works without SAM.') from exc
            if folder_paths.get_full_path('checkpoints', SAM_FILE) is None:
                raise FileNotFoundError(f'Missing models/checkpoints/{SAM_FILE}. '
                                        'Run this kit install.sh --models bf16 --sam.')
            x, y, x2, y2 = 0, 0, width, height
            if search_region == 'painted ROI':
                if manual is None:
                    raise ValueError('Paint a rough source ROI and save it before selecting painted ROI.')
                active = manual[0] > 0
                ys = torch.where(active.any(dim=1))[0]
                xs = torch.where(active.any(dim=0))[0]
                x, y = max(0, int(xs[0]) - 128), max(0, int(ys[0]) - 128)
                x2, y2 = min(width, int(xs[-1]) + 129), min(height, int(ys[-1]) + 129)
            elif search_region != 'full photo':
                raise ValueError('Unknown SAM search region.')
            model, clip, _ = nodes.CheckpointLoaderSimple().load_checkpoint(SAM_FILE)
            cond = nodes.CLIPTextEncode().encode(clip, sam_prompt)[0]
            try:
                result = _tuple(SAM3_Detect.execute(
                    model=model, image=source[:, y:y2, x:x2, :3].clone(), conditioning=cond,
                    threshold=threshold, refine_iterations=refine_iterations, individual_masks=True))
                masks = result[0]
                selected_roi = choose_instances(masks, instance_index)
                if tuple(selected_roi.shape[1:]) != (y2 - y, x2 - x):
                    raise ValueError('SAM mask is not in source coordinates; refusing to resize it silently.')
                selected = torch.zeros((1, height, width), dtype=torch.float32)
                selected[:, y:y2, x:x2] = selected_roi
                info.update(sam_prompt=sam_prompt, threshold=threshold,
                            refine_iterations=refine_iterations, instance_count=masks.shape[0],
                            instance_index=instance_index, search_xyxy=[x, y, x2, y2],
                            manual_correction=manual_correction)
            finally:
                # SAM is not a graph output. Let core offload all loaded models after
                # detection, before the downstream Qwen loader/encoder is requested.
                import comfy.model_management as mm
                mm.unload_all_models()
            selected = combine_masks(selected, manual, manual_correction, expand_pixels)
        selected = mask_check(selected, height, width)
        info.update(editable_pixels=int(torch.count_nonzero(selected)), source_size=[width, height])
        return selected, json.dumps(info, indent=2)


class Q21PhotoCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source': ('IMAGE',), 'mask': ('MASK',),
            'mask_report': ('STRING', {'forceInput': True}),
            'resolution': ('INT', {'default': 2048, 'min': 256, 'max': 2048, 'step': 32}),
            'max_long_side': ('INT', {'default': 3072, 'min': 256, 'max': 4096, 'step': 32}),
            'context_pixels': ('INT', {'default': 128, 'min': 0, 'max': 2048, 'step': 16}),
            'upscale_small_crops': ('BOOLEAN', {'default': False}),
        }}
    RETURN_TYPES = ('Q21_PHOTO_CROP', 'IMAGE')
    RETURN_NAMES = ('crop_data', 'working_crop')
    FUNCTION = 'crop'
    CATEGORY = CATEGORY
    DESCRIPTION = ('Resolution 2048 is an area ceiling of 2048 squared, not a forced square. '
                   'No source-canvas resizing. 32px alignment is padding, removed before stitching.')

    def crop(self, source, mask, mask_report, resolution=2048, max_long_side=3072,
             context_pixels=128, upscale_small_crops=False):
        data, work, _ = crop_photo(source, mask, context_pixels, resolution,
                                   max_long_side, upscale_small_crops)
        data['mask_report'] = json.loads(mask_report)
        return data, work


class Q21PhotoModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'crop_data': ('Q21_PHOTO_CROP',),
            'precision': (list(PRESETS), {'default': 'BF16 quality'}),
            'text_encoder_device': (['default', 'cpu'], {'default': 'default'}),
            'cache_device': (['auto', 'cpu', 'gpu', 'off'], {'default': 'auto'}),
        },'optional':{'checkpoint':(model_choices('qwen','diffusion_models'),)}}
    RETURN_TYPES = ('MODEL', 'CLIP', 'VAE')
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Matching Qwen 2.1 models. Cache remains unquantized. Mask preview never requests this node.'

    def load(self, crop_data, precision='BF16 quality', text_encoder_device='default', cache_device='auto', checkpoint='qwen_image_2.1_bf16.safetensors'):
        import folder_paths
        import nodes
        from comfy_extras.nodes_qwen import QwenImage21Cache
        unet, encoder = PRESETS[precision]
        unet=check_selection('qwen','diffusion_models',checkpoint)
        for category, name in [('diffusion_models', unet), ('text_encoders', encoder), ('vae', VAE_FILE)]:
            if folder_paths.get_full_path(category, name) is None:
                raise FileNotFoundError(f'Missing models/{category}/{name}. Run the kit model installer.')
        model = nodes.UNETLoader().load_unet(unet, 'default')[0]
        model = _tuple(QwenImage21Cache.execute(model, cache_device, 'default'))[0]
        clip = nodes.CLIPLoader().load_clip(encoder, 'qwen_image', device=text_encoder_device)[0]
        vae = nodes.VAELoader().load_vae(VAE_FILE)[0]
        return model, clip, vae


class Q21PhotoLoRAs:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        required = {'model': ('MODEL',)}
        choices = ['None'] + lora_choices('qwen')
        for i in range(1, 4):
            required[f'lora_{i}'] = (choices, {'default': 'None'})
            required[f'strength_{i}'] = ('FLOAT', {'default': 0.6, 'min': -2., 'max': 2., 'step': .05})
        return {'required': required}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY
    DESCRIPTION = ('Three model-only LoRA slots, all off by default. Use Qwen-Image-2.1 adapters only, '
                   'not FLUX, Qwen Edit 2509/2511, or Lightning. Duplicate node for additional slots.')

    def apply(self, model, **kwargs):
        import nodes
        for i in range(1, 4):
            name, strength = kwargs.get(f'lora_{i}', 'None'), kwargs.get(f'strength_{i}', 0.6)
            if name != 'None' and strength:
                if name not in lora_choices('qwen'):raise ValueError('LoRA is not registered for Qwen Image 2.1.')
                if not name.lower().endswith('.safetensors'):
                    raise ValueError('Only safetensors LoRAs are accepted.')
                model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, name, strength)[0]
        return (model,)


class Q21PhotoReference:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        paths = Path(folder_paths.get_input_directory())
        # The standard image_upload flag creates ComfyUI's upload widget.
        files = sorted(str(p.relative_to(paths)).replace('\\', '/') for p in paths.rglob('*')
                       if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'))
        return {'required': {
            'image': (['None'] + files, {'image_upload': True}),
            'role': ('STRING', {'default': '', 'multiline': True}),
            'resolution': ('INT', {'default': 1536, 'min': 256, 'max': 2048, 'step': 32}),
        }}
    RETURN_TYPES = ('Q21_PHOTO_REF',)
    RETURN_NAMES = ('reference',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    DESCRIPTION = ('None disables this reference without requiring a file. Role text is appended verbatim '
                   'with its image tag. References get their own area budget, never enlarge by default.')

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        if image == 'None':
            return 'None'
        import folder_paths
        path = Path(folder_paths.get_annotated_filepath(image))
        if not path.is_file():
            return 'missing'
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def load(self, image='None', role='', resolution=1536):
        if image == 'None':
            return ({'enabled': False},)
        import nodes
        value = nodes.LoadImage().load_image(image)[0]
        value = prepare_reference(value[..., :3], resolution, 3072)
        return ({'enabled': True, 'image': value, 'filename': image, 'role': role.strip()},)


def ordered_references(references):
    result = []
    disabled_seen = False
    for i in range(1, 10):
        ref = references.get(f'reference_{i}')
        if not ref or not ref.get('enabled'):
            disabled_seen = True
            continue
        if disabled_seen:
            raise ValueError('Fill reference slots consecutively, starting at reference 1. '
                             'Gaps would change image-tag numbering; move the reference to the first empty slot.')
        result.append(ref)
    return result


class Q21PhotoEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'clip': ('CLIP',), 'vae': ('VAE',), 'crop_data': ('Q21_PHOTO_CROP',),
            'prompt': ('STRING', {'default': 'Edit <image1> using the supplied reference roles. Describe the exact change here.', 'multiline': True}),
        }, 'optional': {**{f'reference_{i}': ('Q21_PHOTO_REF',) for i in range(1, 10)},
                         'prompt_enhancer': ('WB_QWEN_PROMPT',)}}
    RETURN_TYPES = ('CONDITIONING', 'CONDITIONING', 'LATENT', 'STRING')
    RETURN_NAMES = ('positive', 'negative', 'latent', 'reference_report')
    FUNCTION = 'encode'
    CATEGORY = CATEGORY
    DESCRIPTION = ('Official Qwen 2.1 conditioning with resolution=0 after independent, 32-aligned preprocessing. '
                   'One source crop plus up to nine references. Optional official I2I prompt enhancer; no image resizing a second time.')

    def encode(self, clip, vae, crop_data, prompt, prompt_enhancer=None, **references):
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        refs = ordered_references(references)
        images = {'image_1': crop_data['work_image']}
        roles = []
        info = [{'tag': '<image1>', 'role': 'source crop / edit target',
                 'size': [images['image_1'].shape[2], images['image_1'].shape[1]]}]
        for index, ref in enumerate(refs, 2):
            images[f'image_{index}'] = ref['image']
            if ref['role']:
                roles.append(f'<image{index}>: {ref["role"]}')
            info.append({'tag': f'<image{index}>', 'filename': ref['filename'],
                         'role': ref['role'], 'size': [ref['image'].shape[2], ref['image'].shape[1]]})
        user_tags = [int(n) for n in re.findall(r'<image(\d+)>', prompt)]
        if any(n < 1 or n > len(images) for n in user_tags):
            raise ValueError(f'Prompt names an unavailable image. This run has <image1> through <image{len(images)}>.')
        full_prompt = prompt.strip()
        if roles:
            full_prompt += '\n\nReference roles specified by the user:\n' + '\n'.join(roles)
        from workbench.qwen_prompt import enhance_edit_prompt, ui_result
        effective, enhancement_report = enhance_edit_prompt(full_prompt, images, prompt_enhancer)
        out = _tuple(TextEncodeQwenImage21.execute(
            clip=clip, prompt=effective, negative_prompt='', vae=vae, resolution=0, images=images))
        positive, negative, latent = out
        plan = crop_data['plan']
        if tuple(latent['samples'].shape) != (1, 64, plan.work_height // 16, plan.work_width // 16):
            raise ValueError('Unexpected Qwen 2.1 latent layout or crop size. Check matching Qwen 2.1 VAE and core.')
        report = {'references': info, 'effective_prompt': effective}
        if prompt_enhancer is not None:
            report['prompt_enhancement'] = enhancement_report
        outputs = (positive, negative, latent, json.dumps(report, indent=2))
        return outputs if prompt_enhancer is None else ui_result(outputs, enhancement_report)


class Q21PhotoStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'crop_data': ('Q21_PHOTO_CROP',), 'generated_crop': ('IMAGE',),
            'reference_report': ('STRING', {'forceInput': True}),
            'feather_pixels': ('INT', {'default': 12, 'min': 0, 'max': 128}),
            'boundary_color_strength': ('FLOAT', {'default': 0.0, 'min': 0., 'max': 1., 'step': .05}),
        }}
    RETURN_TYPES = ('Q21_PHOTO_RESULT',)
    RETURN_NAMES = ('edited_result',)
    FUNCTION = 'stitch'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Replace only the effective mask, preserve its holes, feather inward, retain exact source dimensions.'

    def stitch(self, crop_data, generated_crop, reference_report, feather_pixels=12, boundary_color_strength=0.):
        rgba = generated_crop.shape[-1] == 4
        if rgba:
            if generated_crop.shape[:3] != crop_data['work_image'].shape[:3]:
                raise ValueError('RGBA crop dimensions changed; refusing a misaligned stitch.')
            alpha = generated_crop[..., 3:4].detach().cpu().clamp(0, 1)
            generated_crop = generated_crop[..., :3].detach().cpu() * alpha + crop_data['work_image'] * (1 - alpha)
        output, report = stitch_photo(crop_data, generated_crop, feather_pixels, boundary_color_strength)
        report = json.loads(report)
        report.update(version=VERSION, mask=json.loads(json.dumps(crop_data['mask_report'])),
                      references=json.loads(reference_report), rgba_composited=rgba)
        return ({'image': output, 'report': report},)


def preview_overlay(data, crop=False):
    source, mask = data['source'], data['effective_mask']
    if crop:
        p = data['plan']
        source = source[:, p.y:p.y+p.crop_height, p.x:p.x+p.crop_width]
        mask = mask[:, p.y:p.y+p.crop_height, p.x:p.x+p.crop_width]
    w, h = fit_dimensions(source.shape[2], source.shape[1], 1536)
    image = resize_rgb(source, w, h)
    m = torch.nn.functional.interpolate(mask[:, None], (h, w), mode='nearest')[:, 0, :, :, None]
    color = torch.tensor([1., 0.15, 0.1]).view(1, 1, 1, 3)
    return image * (1. - .4 * m) + color * (.4 * m)


class Q21PhotoReviewSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'crop_data': ('Q21_PHOTO_CROP',),
            'run_edit': ('BOOLEAN', {'default': False}),
            'filename_prefix': ('STRING', {'default': 'Qwen21_Photo/Edited'}),
        }, 'optional': {'edited_result': ('Q21_PHOTO_RESULT', {'lazy': True})},
            'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'review'
    CATEGORY = CATEGORY
    DESCRIPTION = ('OFF previews the effective mask without loading Qwen, LoRAs, or references. '
                   'ON runs the edit, compares before/after, and saves full-size PNG + mask + verification JSON.')

    def check_lazy_status(self, crop_data, run_edit=False, filename_prefix='Qwen21_Photo/Edited', edited_result=None, **kwargs):
        return ['edited_result'] if run_edit and edited_result is None else []

    def review(self, crop_data, run_edit=False, filename_prefix='Qwen21_Photo/Edited',
               edited_result=None, prompt=None, extra_pnginfo=None):
        import folder_paths
        tmp = Path(folder_paths.get_temp_directory())
        tmp.mkdir(parents=True, exist_ok=True)
        def preview(image, label):
            w, h = fit_dimensions(image.shape[2], image.shape[1], 1536)
            name = f'q21_{uuid.uuid4().hex}_{label}.png'
            Image.fromarray(to_uint8(resize_rgb(image, w, h))).save(tmp / name)
            return {'filename': name, 'subfolder': '', 'type': 'temp', 'label': label}
        overlay = preview_overlay(crop_data)
        source = crop_data['source']
        original_preview = preview(source, 'original')
        mask_preview = preview(overlay, 'mask')
        if not run_edit:
            p = crop_data['plan']
            report = {'mode': 'MASK PREVIEW ONLY: no Qwen inference and no edited photo saved',
                      'next': 'Review the mask. Turn run_edit ON to generate.',
                      'source_size': [p.source_width, p.source_height],
                      'model_processing_size': [p.work_width, p.work_height],
                      'crop_resampled': p.resampled, 'mask': crop_data['mask_report']}
            return {'ui': {'images': [mask_preview], 'q21_full': [original_preview, mask_preview],
                           'q21_crop': [preview(source[:, p.y:p.y+p.crop_height, p.x:p.x+p.crop_width], 'crop_original'),
                                        preview(preview_overlay(crop_data, crop=True), 'crop_mask')],
                           'q21_report': [json.dumps(report, indent=2)]}}
        if edited_result is None:
            raise ValueError('run_edit is on, but no edited_result is connected.')
        image, report = edited_result['image'], edited_result['report']
        if tuple(image.shape) != tuple(source.shape) or report.get('outside_mask_changed_channels') != 0:
            raise ValueError('Source protection verification failed. Refusing to save.')
        out_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), image.shape[2], image.shape[1])
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f'{filename}_{counter:05}_{uuid.uuid4().hex[:8]}'
        import os
        if os.environ.get('WB_SAVE_METADATA','0')!='1':prompt=None;extra_pnginfo=None
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text('prompt', json.dumps(prompt))
        for key, value in (extra_pnginfo or {}).items():
            metadata.add_text(key, json.dumps(value))
        metadata.add_text('qwen21_photo_verification', json.dumps(report))
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        Image.fromarray(to_uint8(image)).save(out_dir / (name + '.png'), pnginfo=metadata,
                                             icc_profile=profile, compress_level=4)
        mask_array = np.rint(crop_data['effective_mask'][0].numpy().clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(mask_array).save(out_dir / (name + '.mask.png'))
        (out_dir / (name + '.verification.json')).write_text(json.dumps(report, indent=2) + '\n')
        p = crop_data['plan']
        crop_a = source[:, p.y:p.y+p.crop_height, p.x:p.x+p.crop_width]
        crop_b = image[:, p.y:p.y+p.crop_height, p.x:p.x+p.crop_width]
        return {'ui': {
            'images': [{'filename': name + '.png', 'subfolder': subfolder, 'type': 'output'}],
            'q21_full': [original_preview, preview(image, 'edited')],
            'q21_crop': [preview(crop_a, 'crop_original'), preview(crop_b, 'crop_edited')],
            'q21_report': [json.dumps(report, indent=2)],
        }}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in [Q21PhotoMask, Q21PhotoCrop, Q21PhotoModels,
    Q21PhotoLoRAs, Q21PhotoReference, Q21PhotoEncode, Q21PhotoStitch, Q21PhotoReviewSave]}
NODE_DISPLAY_NAME_MAPPINGS = {
    'Q21PhotoMask': 'Qwen Photo | Manual / SAM 3.1 Mask',
    'Q21PhotoCrop': 'Qwen Photo | Inpaint Crop 2K',
    'Q21PhotoModels': 'Qwen Photo | Matching Models + Lossless Cache',
    'Q21PhotoLoRAs': 'Qwen Photo | Optional LoRAs',
    'Q21PhotoReference': 'Qwen Photo | Optional Reference + Role',
    'Q21PhotoEncode': 'Qwen Photo | Multi-reference Conditioning',
    'Q21PhotoStitch': 'Qwen Photo | Protected Inpaint Stitch',
    'Q21PhotoReviewSave': 'Qwen Photo | Preview / Run / Compare / Save',
}
WEB_DIRECTORY = './web'
