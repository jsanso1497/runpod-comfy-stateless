"""Optional Krea masking and protected compositing using Qwen's tested geometry.

The source Qwen module is imported privately, not registered as a Qwen model
workspace. Its model-loading methods are never called by these utilities.
"""
from __future__ import annotations
from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import uuid
import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps
from PIL.PngImagePlugin import PngInfo
import folder_paths
import nodes
from workbench.config import ROOT

CATEGORY = 'Workbench/Krea 2/Photo tools'


@lru_cache(maxsize=1)
def _photo():
    path = ROOT / 'src/nodes/qwen_photo/__init__.py'
    name = '_workbench_shared_photo_for_krea'
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    if spec is None or spec.loader is None:
        raise RuntimeError('Shared photo utilities are missing. Rebuild the Krea workspace.')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_srgb(image):
    """Keep native mask/EXIF geometry; convert an explicitly tagged still to sRGB."""
    result = nodes.LoadImage().load_image(image)
    rgb, mask = result[:2]
    if rgb.ndim != 4 or rgb.shape[0] != 1:
        raise ValueError('Use one still source/reference image, not an animation or image batch.')
    path = folder_paths.get_annotated_filepath(image)
    with Image.open(path) as original:
        profile = original.info.get('icc_profile')
        if not profile:
            return rgb, mask  # Untagged inputs are assumed to already be sRGB.
        import io
        photo = ImageOps.exif_transpose(original)
        if photo.mode not in ('RGB', 'CMYK', 'LAB', 'L'):
            photo = photo.convert('RGB')
        try:
            photo = ImageCms.profileToProfile(photo, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                             ImageCms.createProfile('sRGB'), outputMode='RGB')
        except (OSError, ValueError, ImageCms.PyCMSError) as exc:
            raise ValueError('The embedded color profile could not be converted. Export this image as sRGB before loading it.') from exc
        converted = torch.from_numpy(np.asarray(photo).copy()).float().unsqueeze(0) / 255
    if tuple(converted.shape) != tuple(rgb.shape):
        raise ValueError('Color conversion changed image geometry. Re-export as an oriented sRGB still.')
    return converted, mask


class WBKreaLoadImage(nodes.LoadImage):
    CATEGORY = CATEGORY
    DESCRIPTION = 'Native still-image and mask loading, with embedded ICC profiles converted to sRGB. Untagged files are assumed sRGB.'

    def load_image(self, image):
        return load_srgb(image)


class WBKreaMask:
    @classmethod
    def INPUT_TYPES(cls):
        result = _photo().Q21PhotoMask.INPUT_TYPES()
        result['required']['sam_prompt'][1]['default'] = 'person'
        return result
    RETURN_TYPES = ('MASK', 'STRING')
    RETURN_NAMES = ('effective_edit_mask', 'mask_report')
    FUNCTION = 'make_mask'
    CATEGORY = CATEGORY
    DESCRIPTION = 'OFF uses the saved mask on the source. ON uses the same native SAM 3.1 path as Qwen, always on the source image.'

    def make_mask(self, *args, **kwargs):
        return _photo().Q21PhotoMask().make_mask(*args, **kwargs)


class WBKreaCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source': ('IMAGE',), 'mask': ('MASK',),
            'mask_report': ('STRING', {'forceInput': True}),
            'megapixels': ('FLOAT', {'default': 1.9, 'min': .25, 'max': 2., 'step': .05}),
            'max_long_side': ('INT', {'default': 2048, 'min': 256, 'max': 4096, 'step': 32}),
            'context_pixels': ('INT', {'default': 128, 'min': 32, 'max': 2048, 'step': 16}),
            'upscale_small_crops': ('BOOLEAN', {'default': False})},
            'optional': {'protect_mask': ('MASK',)}}
    RETURN_TYPES = ('WB_KREA_CROP', 'IMAGE')
    RETURN_NAMES = ('crop_data', 'working_crop')
    FUNCTION = 'crop'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Original-size crop/stitch geometry, capped below 2 MP for Krea. Source pixels outside the final mask stay untouched.'

    def crop(self, source, mask, mask_report, megapixels=1.9, max_long_side=2048,
             context_pixels=128, upscale_small_crops=False, protect_mask=None):
        if not .25 <= megapixels <= 2. or not 32 <= context_pixels <= 2048:
            raise ValueError('Use 0.25 to 2.0 MP and 32 to 2048 pixels of crop context.')
        if not torch.isfinite(source).all():
            raise ValueError('Source contains invalid pixels.')
        photo = _photo()
        mask = photo.mask_check(mask, source.shape[1], source.shape[2])
        if protect_mask is not None:
            if protect_mask.shape != mask.shape or not torch.isfinite(protect_mask).all() or torch.any((protect_mask < 0) | (protect_mask > 1)):
                raise ValueError('Protection must be a finite 0-to-1 mask at the exact source size.')
            mask = torch.where(protect_mask.detach().cpu() > 0, torch.zeros_like(mask), mask)
        edge = int(math.sqrt(megapixels * 1_000_000)) // 32 * 32
        data, work, _ = photo.crop_photo(source, mask, context_pixels, edge, max_long_side, upscale_small_crops)
        if min(work.shape[1:3]) < 64:
            raise ValueError('The working crop is too narrow. Increase crop context or enable crop upscaling.')
        data['mask_report'] = json.loads(mask_report)
        data['mask_report']['final_editable_pixels'] = int(torch.count_nonzero(mask))
        data['mask_report']['protection_applied'] = protect_mask is not None
        data['krea_megapixels'] = megapixels
        return data, work


class WBKreaStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'crop_data': ('WB_KREA_CROP',), 'generated_crop': ('IMAGE',),
                            'boundary_color_strength': ('FLOAT', {'default': 0., 'min': 0., 'max': 1., 'step': .05})}}
    RETURN_TYPES = ('WB_KREA_RESULT',)
    FUNCTION = 'stitch'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Undo crop padding/resizing, apply the reviewed mask, and verify that unselected source pixels are unchanged. Optional conservative boundary color correction.'

    def stitch(self, crop_data, generated_crop, boundary_color_strength=0.):
        image, raw = _photo().stitch_photo(crop_data, generated_crop, 0, boundary_color_strength)
        report = json.loads(raw)
        report.update(model='Krea 2 Identity Edit', version='krea-qwen-parity-1',
                      feather='explicit source-space outward feather, followed by optional protection',
                      mask=crop_data['mask_report'], generation_cap_mp=crop_data['krea_megapixels'])
        return ({'image': image, 'report': report},)


def _save(image, prefix, report=None, prompt=None, extra_pnginfo=None):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3 or not torch.isfinite(image).all():
        raise ValueError('Save requires one finite RGB image.')
    directory, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), image.shape[2], image.shape[1])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f'{filename}_{counter:05}_{uuid.uuid4().hex[:8]}'
    metadata = PngInfo()
    if os.environ.get('WB_SAVE_METADATA', '0') == '1':
        if prompt is not None:
            metadata.add_text('prompt', json.dumps(prompt))
        for key, value in (extra_pnginfo or {}).items():
            metadata.add_text(key, json.dumps(value))
    if report is not None:
        metadata.add_text('krea_photo_verification', json.dumps(report))
        (directory / (stem + '.verification.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    array = (image[0].detach().float().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(array).save(directory / (stem + '.png'), pnginfo=metadata, icc_profile=profile, compress_level=4)
    return {'filename': stem + '.png', 'subfolder': subfolder, 'type': 'output'}


class WBKreaSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'filename_prefix': ('STRING', {'default': 'Krea/Edited'})},
                'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'save_images'
    CATEGORY = CATEGORY

    def save_images(self, images, filename_prefix='Krea/Edited', prompt=None, extra_pnginfo=None):
        saved = _save(images, filename_prefix, prompt=prompt, extra_pnginfo=extra_pnginfo)
        return {'ui': {'images': [saved]}}


class WBKreaReviewSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'crop_data': ('WB_KREA_CROP',), 'run_edit': ('BOOLEAN', {'default': False}),
                            'filename_prefix': ('STRING', {'default': 'Krea/Masked_Edit'})},
                'optional': {'edited_result': ('WB_KREA_RESULT', {'lazy': True})},
                'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'review'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Preview the final mask first with no Krea or prompt-model inference. Turn run_edit ON to generate, verify, and save a full-size sRGB PNG.'

    def check_lazy_status(self, crop_data, run_edit=False, filename_prefix='Krea/Masked_Edit', edited_result=None, **kwargs):
        return ['edited_result'] if run_edit and edited_result is None else []

    def review(self, crop_data, run_edit=False, filename_prefix='Krea/Masked_Edit',
               edited_result=None, prompt=None, extra_pnginfo=None):
        if not run_edit:
            result = nodes.PreviewImage().save_images(_photo().preview_overlay(crop_data))
            result.setdefault('ui', {})['text'] = ['MASK PREVIEW ONLY. Turn run_edit ON after checking the source selection.']
            return result
        if edited_result is None:
            raise ValueError('Connect the stitched result before enabling run_edit.')
        image, report = edited_result['image'], edited_result['report']
        source, mask = crop_data['source'], crop_data['effective_mask']
        if image.shape != source.shape or report.get('outside_mask_changed_channels') != 0:
            raise ValueError('Source preservation failed; output was not saved.')
        outside = (mask == 0).unsqueeze(-1).expand_as(source)
        if torch.any(image[outside] != source[outside]):
            raise ValueError('Unselected pixels changed; output was not saved.')
        saved = _save(image, filename_prefix, report, prompt, extra_pnginfo)
        return {'ui': {'images': [saved], 'text': [json.dumps(report, indent=2)]}}


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    WBKreaLoadImage, WBKreaMask, WBKreaCrop, WBKreaStitch, WBKreaSaveImage, WBKreaReviewSave)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'WBKreaLoadImage': 'Krea | Load source / reference as sRGB',
    'WBKreaMask': 'Krea | Manual / SAM 3.1 source mask',
    'WBKreaCrop': 'Krea | Protected photo crop (up to 2 MP)',
    'WBKreaStitch': 'Krea | Exact-source masked stitch',
    'WBKreaSaveImage': 'Krea | Save sRGB PNG',
    'WBKreaReviewSave': 'Krea | Preview mask / Run edit / Verify / Save',
}
