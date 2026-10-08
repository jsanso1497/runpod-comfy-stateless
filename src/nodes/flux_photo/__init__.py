"""FLUX.2 klein 9B distilled: explicit reference inpainting with a protected canvas."""
from __future__ import annotations
from workbench.registry import choices, check_selection, lora_choices

import json
import logging
from pathlib import Path
import uuid

import torch
from PIL import Image, ImageCms
from PIL.PngImagePlugin import PngInfo

from .photo_ops import (crop_photo, fit_dimensions, image_check, prepare_reference,
                        resize_rgb, stitch_photo, to_uint8)

CATEGORY = 'FLUX Photo / Protected Inpaint'
MODEL = 'flux-2-klein-9b.safetensors'
ENCODER = 'qwen_3_8b.safetensors'
VAE = 'flux2-vae.safetensors'


class FluxPhotoModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text_encoder_device': (['cpu', 'default'], {'default': 'cpu'})},'optional':{'checkpoint':(choices('flux','diffusion_models'),)}}
    RETURN_TYPES = ('MODEL', 'CLIP', 'VAE')
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Full-precision 9B DISTILLED only. Fixed matching Qwen3 8B and FLUX2 VAE; no FP8 or GGUF substitution.'

    def load(self, text_encoder_device='cpu', checkpoint=MODEL):
        checkpoint=check_selection('flux','diffusion_models',checkpoint)
        import folder_paths
        import nodes
        for category, filename in [('diffusion_models', checkpoint), ('text_encoders', ENCODER), ('vae', VAE)]:
            if folder_paths.get_full_path(category, filename) is None:
                raise FileNotFoundError(f'Missing {filename}. Check FLUX Photo startup download logs. '
                                        'The BFL model requires an HF_TOKEN with accepted model access.')
        model = nodes.UNETLoader().load_unet(checkpoint, 'default')[0]
        clip = nodes.CLIPLoader().load_clip(ENCODER, 'flux2', device=text_encoder_device)[0]
        vae = nodes.VAELoader().load_vae(VAE)[0]
        return model, clip, vae


class FluxPhotoOptionalLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required': {
            'model': ('MODEL',),
            'lora_name': (['None'] + lora_choices('flux'),),
            'strength': ('FLOAT', {'default': 0.75, 'min': -2.0, 'max': 2.0, 'step': 0.05}),
        }}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Optional model-only LoRA. None is valid with an empty LoRA directory. Use FLUX.2 klein 9B-compatible adapters, not FLUX.1 or 4B.'

    def apply(self, model, lora_name='None', strength=0.75):
        if lora_name == 'None' or strength == 0:
            return (model,)
        if lora_name not in lora_choices('flux'):raise ValueError('LoRA is not registered for FLUX.2 klein 9B.')
        if Path(lora_name).suffix.lower() != '.safetensors':
            raise ValueError('Use a .safetensors LoRA. Other checkpoint formats are not loaded here.')
        import nodes
        return nodes.LoraLoaderModelOnly().load_lora_model_only(model, lora_name, strength)


class FluxPhotoCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source': ('IMAGE',), 'mask': ('MASK',),
            'processing': (['fit', 'native'], {'default': 'fit'}),
            'max_crop_side': ('INT', {'default': 2048, 'min': 256, 'max': 8192, 'step': 16}),
            'context_pixels': ('INT', {'default': 128, 'min': 0, 'max': 2048, 'step': 16}),
            'sampling_mask_grow': ('INT', {'default': 16, 'min': 0, 'max': 256, 'step': 1}),
        }}
    RETURN_TYPES = ('FLUX_PHOTO_CROP', 'IMAGE', 'MASK')
    RETURN_NAMES = ('stitch_data', 'cropped_image', 'sampling_mask')
    FUNCTION = 'crop'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Crops on CPU before resizing. Native means no resampling and errors above the size limit. Fit resizes only the crop. Mask grow never expands the final paste boundary.'

    def crop(self, source, mask, processing, max_crop_side, context_pixels, sampling_mask_grow):
        return crop_photo(source, mask, context_pixels, processing, max_crop_side, sampling_mask_grow)


class FluxPhotoReferenceConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'conditioning': ('CONDITIONING',), 'vae': ('VAE',),
            'cropped_image': ('IMAGE',), 'sampling_mask': ('MASK',),
            'replacement_reference': ('IMAGE',),
            'scene_reference_long_side': ('INT', {'default': 1024, 'min': 256, 'max': 2048, 'step': 16}),
            'replacement_reference_long_side': ('INT', {'default': 2048, 'min': 256, 'max': 4096, 'step': 16}),
            'vae_tile_size': ('INT', {'default': 1024, 'min': 256, 'max': 2048, 'step': 64}),
            'vae_overlap': ('INT', {'default': 128, 'min': 32, 'max': 256, 'step': 32}),
        }}
    RETURN_TYPES = ('CONDITIONING', 'LATENT')
    RETURN_NAMES = ('two_image_conditioning', 'masked_source_latent')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Image 1 is the cropped source scene. Image 2 is the replacement reference. Your prompt is not rewritten. Tiled VAE encoding is not tiled diffusion.'

    def prepare(self, conditioning, vae, cropped_image, sampling_mask, replacement_reference,
                scene_reference_long_side=1024, replacement_reference_long_side=2048,
                vae_tile_size=1024, vae_overlap=128):
        import nodes
        image_check(cropped_image)
        if tuple(sampling_mask.shape) != tuple(cropped_image.shape[:3]):
            raise ValueError('Sampling mask must match the crop exactly.')
        if vae_overlap >= vae_tile_size // 2:
            raise ValueError('VAE overlap must be less than half the tile size.')
        encoder = nodes.VAEEncodeTiled()
        def encode(image):
            out = encoder.encode(vae, image, vae_tile_size, vae_overlap)[0]
            if out['samples'].ndim != 4 or out['samples'].shape[1] != 128:
                raise ValueError('Expected a FLUX2 128-channel image latent. Do not substitute KREA/H3/FLUX1 VAEs.')
            return out
        latent = encode(cropped_image)
        scene = prepare_reference(cropped_image, scene_reference_long_side)
        if tuple(scene.shape) == tuple(cropped_image.shape):
            scene_latent = latent['samples']
        else:
            scene_latent = encode(scene)['samples']
        reference = prepare_reference(replacement_reference, replacement_reference_long_side)
        reference_latent = encode(reference)['samples']
        result = []
        for tensor, metadata in conditioning:
            if metadata.get('reference_latents'):
                raise ValueError('Input conditioning already contains references. This node owns the image 1 / image 2 order.')
            metadata = dict(metadata)
            metadata['reference_latents'] = [scene_latent, reference_latent]
            result.append([tensor, metadata])
        latent = dict(latent)
        latent['noise_mask'] = sampling_mask
        latent['flux_photo_dimensions'] = [cropped_image.shape[2], cropped_image.shape[1]]
        return result, latent


class FluxPhotoSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'conditioning': ('CONDITIONING',), 'latent': ('LATENT',),
            'seed': ('INT', {'default': 12345, 'min': 0, 'max': 0xffffffffffffffff,
                             'control_after_generate': True}),
            'steps': ('INT', {'default': 4, 'min': 1, 'max': 32}),
        }}
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'sample'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Euler, CFG 1.0, full masked replacement, native resolution-aware FLUX2 schedule. Start with 4 distilled steps. More steps are not a quality guarantee.'

    def sample(self, model, conditioning, latent, seed=12345, steps=4):
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import latent_preview
        from comfy_extras.nodes_flux import get_schedule
        width, height = latent['flux_photo_dimensions']
        sigmas = get_schedule(steps, (width * height) // 256)
        noise = comfy.sample.prepare_noise(latent['samples'], seed, latent.get('batch_index'))
        callback = latent_preview.prepare_callback(model, steps)
        if steps != 4:
            logging.warning('FLUX Photo: distilled checkpoint is designed for 4 steps; current steps=%s.', steps)
        try:
            result = comfy.sample.sample_custom(
                model, noise, 1.0, comfy.samplers.sampler_object('euler'), sigmas,
                conditioning, conditioning, latent['samples'], noise_mask=latent['noise_mask'],
                callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError('FLUX Photo ran out of GPU memory. No automatic precision or resolution '
                               'downgrade was made. Use the 2048 fit workflow, lower replacement reference '
                               'size to 1536/1024, reduce crop context, or choose more VRAM. Tiled VAE does '
                               'not tile diffusion attention.') from exc
        out = dict(latent)
        out['samples'] = result
        return (out,)


class FluxPhotoStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'stitch_data': ('FLUX_PHOTO_CROP',), 'generated_crop': ('IMAGE',),
            'feather_pixels': ('INT', {'default': 16, 'min': 0, 'max': 256}),
            'boundary_color_strength': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
        }}
    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('original_size_result', 'verification_report')
    FUNCTION = 'stitch'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Strict original-size stitch. All zero-mask source pixels are copied exactly. Feathering stays inside your mask. Optional boundary color matching is OFF by default.'

    def stitch(self, stitch_data, generated_crop, feather_pixels=16, boundary_color_strength=0.0):
        return stitch_photo(stitch_data, generated_crop, feather_pixels, boundary_color_strength)


class FluxPhotoCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'original': ('IMAGE',), 'edited': ('IMAGE',), 'report': ('STRING', {'forceInput': True}),
            'preview_long_side': ('INT', {'default': 2048, 'min': 512, 'max': 8192, 'step': 16}),
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'compare'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Interactive original/output slider. Preview size has no effect on the saved image. Use a full-size PNG externally to judge 100% pixel detail.'

    def compare(self, original, edited, report, preview_long_side=2048):
        import folder_paths
        if tuple(original.shape) != tuple(edited.shape):
            raise ValueError('Compare images must have identical dimensions.')
        directory = Path(folder_paths.get_temp_directory())
        directory.mkdir(parents=True, exist_ok=True)
        width, height = fit_dimensions(original.shape[2], original.shape[1], preview_long_side)
        images = []
        for label, image in [('Original', original), ('Edited', edited)]:
            name = f'flux_photo_compare_{uuid.uuid4().hex}_{label}.png'
            Image.fromarray(to_uint8(resize_rgb(image, width, height))).save(directory / name)
            images.append({'filename': name, 'subfolder': '', 'type': 'temp', 'label': label})
        return {'ui': {'photo_compare': images, 'photo_report': [report]}}


class FluxPhotoSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'images': ('IMAGE',), 'report': ('STRING', {'forceInput': True}),
            'filename_prefix': ('STRING', {'default': 'FLUX_Photo/Edited'}),
        }, 'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'save'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Lossless RGB 8-bit PNG with sRGB ICC profile, workflow and verification report. Not RAW, 16-bit, layered, or a byte-identical JPEG export.'

    def save(self, images, report, filename_prefix='FLUX_Photo/Edited', prompt=None, extra_pnginfo=None):
        import folder_paths
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1])
        import os
        if os.environ.get('WB_SAVE_METADATA','0')!='1':prompt=None;extra_pnginfo=None
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text('prompt', json.dumps(prompt))
        for key, value in (extra_pnginfo or {}).items():
            metadata.add_text(key, json.dumps(value))
        metadata.add_text('flux_photo_verification', report)
        data = json.loads(report)
        if data.get('outside_mask_changed_channels') != 0:
            raise ValueError('Verification report failed; refusing to save.')
        if data.get('output_size') != [images.shape[2], images.shape[1]]:
            raise ValueError('Verification report does not match output dimensions.')
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        # UUID suffix also protects against a simultaneous save using the same counter.
        name = f'{filename}_{counter:05}_{uuid.uuid4().hex[:8]}.png'
        destination = Path(folder) / name
        Image.fromarray(to_uint8(images)).save(destination, pnginfo=metadata, icc_profile=profile, compress_level=4)
        destination.with_suffix('.verification.json').write_text(report + '\n', encoding='utf-8')
        return {'ui': {'images': [{'filename': name, 'subfolder': subfolder, 'type': 'output'}]}}


NODE_CLASS_MAPPINGS = {
    cls.__name__: cls for cls in [FluxPhotoModels, FluxPhotoOptionalLoRA, FluxPhotoCrop,
                                FluxPhotoReferenceConditioning, FluxPhotoSampler,
                                FluxPhotoStitch, FluxPhotoCompare, FluxPhotoSave]
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'FluxPhotoModels': 'FLUX Photo | 9B Distilled BF16 Models',
    'FluxPhotoOptionalLoRA': 'FLUX Photo | Optional LoRA',
    'FluxPhotoCrop': 'FLUX Photo | Inpaint Crop',
    'FluxPhotoReferenceConditioning': 'FLUX Photo | Source + Replacement Reference',
    'FluxPhotoSampler': 'FLUX Photo | Distilled Masked Sampler',
    'FluxPhotoStitch': 'FLUX Photo | Protected Inpaint Stitch',
    'FluxPhotoCompare': 'FLUX Photo | Original / Output Compare',
    'FluxPhotoSave': 'FLUX Photo | Original-Size PNG Save',
}
WEB_DIRECTORY = './web'
