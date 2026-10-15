"""Full-resolution, straight-alpha edit overlays. No image-model implementation.

All inference is delegated to the existing mask/crop nodes and the unchanged
native ComfyUI Qwen edit subgraph. PNG export is explicitly 8-bit sRGB.
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms
from PIL.PngImagePlugin import PngInfo
from scipy.ndimage import distance_transform_edt, maximum_filter
import torch

CATEGORY = 'Qwen 2.1 / Full-resolution overlay'


def _array(value):
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError('Non-finite image or mask. Refusing to export.')
    return value


def _mask(value, height, width, label):
    arr = _array(value)
    if arr.shape != (1, height, width) or np.any((arr < 0) | (arr > 1)):
        raise ValueError(f'{label} must be a full-source-size [1,H,W] mask in [0,1].')
    return arr[0]


def _resize(arr, width, height):
    if arr.shape[:2] == (height, width):
        return arr.copy()
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr).resize((width, height), Image.Resampling.LANCZOS)).copy()
    return np.stack([_resize(arr[..., c], width, height) for c in range(arr.shape[-1])], axis=-1)


def _inward(mask, pixels, plan):
    """Feather only inside the allowed area, including around protected holes."""
    if pixels == 0:
        return mask.copy()
    border = pixels + 1
    extended = np.pad(mask > 0, border, mode='edge')
    if plan.x > 0:
        extended[:, :border] = False
    if plan.y > 0:
        extended[:border, :] = False
    if plan.x + plan.crop_width < plan.source_width:
        extended[:, -border:] = False
    if plan.y + plan.crop_height < plan.source_height:
        extended[-border:, :] = False
    if extended.all():
        return mask.copy()
    distance = distance_transform_edt(extended)[border:-border, border:-border]
    t = np.clip(distance / pixels, 0, 1).astype(np.float32)
    return mask * t * t * (3 - 2 * t)


def compose_overlay(package, generated, feather_pixels=12):
    """Return RGB composite, straight RGBA overlay, alpha mask and pixel audit.

    Quantize alpha BEFORE compositing, so the saved overlay reproduces the saved
    composite when placed over the same 8-bit sRGB source with normal blending.
    Untouched/protected pixels are copied, never VAE-decoded or resized.
    """
    if not isinstance(feather_pixels, int) or not 0 <= feather_pixels <= 256:
        raise ValueError('Feather must be an integer from 0 to 256 source pixels.')
    p = package['plan']
    source = _array(package['source'])
    if source.shape != (1, p.source_height, p.source_width, 3):
        raise ValueError('Original image no longer matches the crop plan.')
    image = _array(generated)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError('One RGB or RGBA generated crop is required.')
    if image.shape[1:3] != (p.work_height, p.work_width):
        raise ValueError('Generated crop dimensions changed. Keep native resolution=0 and custom_size OFF.')
    x, y, w, h = p.x, p.y, p.crop_width, p.crop_height
    if x < 0 or y < 0 or x + w > p.source_width or y + h > p.source_height:
        raise ValueError('Invalid crop coordinates.')
    image = image[0, :p.scaled_height, :p.scaled_width]
    patch = np.clip(_resize(image[..., :3], w, h), 0, 1)
    mask = _mask(package['paste_mask'], h, w, 'Edit mask')
    protect = _mask(package['overlay_protection'], p.source_height, p.source_width, 'Protection mask') > 0
    protected_crop = protect[y:y+h, x:x+w]
    allowed = mask * (~protected_crop)
    alpha = _inward(allowed, feather_pixels, p)
    if image.shape[-1] == 4:
        alpha *= np.clip(_resize(image[..., 3], w, h), 0, 1)
    alpha[protected_crop | (mask == 0)] = 0
    alpha8 = np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    original8 = np.rint(np.clip(source[0], 0, 1) * 255).astype(np.uint8)
    rgb8 = np.rint(patch * 255).astype(np.uint8)
    region = original8[y:y+h, x:x+w]
    mixed = np.rint(region.astype(np.float64) +
                    (rgb8.astype(np.float64) - region) * (alpha8[..., None] / 255.0)).astype(np.uint8)
    changed = np.any(mixed != region, axis=-1)
    alpha8[~changed] = 0
    rgba = np.zeros((p.source_height, p.source_width, 4), dtype=np.uint8)
    rgba[y:y+h, x:x+w, :3] = np.where(alpha8[..., None] > 0, rgb8, 0)
    rgba[y:y+h, x:x+w, 3] = alpha8
    composite = original8.copy()
    composite[y:y+h, x:x+w] = mixed
    untouched = rgba[..., 3] == 0
    outside_errors = int(np.count_nonzero(composite[untouched] != original8[untouched]))
    protected_errors = int(np.count_nonzero(composite[protect] != original8[protect]))
    if outside_errors or protected_errors or np.any(rgba[..., 3][protect]):
        raise AssertionError('Pixel protection audit failed. No files were saved.')
    report = {'source_size': [p.source_width, p.source_height],
              'processing_size': [p.work_width, p.work_height],
              'crop_xywh': [x, y, w, h], 'alpha': 'straight, not premultiplied',
              'protected_changed_channels': protected_errors,
              'outside_overlay_changed_channels': outside_errors,
              'changed_pixels': int(np.count_nonzero(rgba[..., 3])),
              'export': '8-bit sRGB PNG', 'gpu_inference_validated': False}
    return composite, rgba, rgba[..., 3].copy(), report


class WBOverlayProtection:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source': ('IMAGE',), 'enabled': ('BOOLEAN', {'default': True}),
            'description': ('STRING', {'default': 'face', 'multiline': True}),
            'threshold': ('FLOAT', {'default': 0.5, 'min': 0.05, 'max': 0.95, 'step': 0.05}),
            'instance_index': ('INT', {'default': -1, 'min': -1, 'max': 63})}}
    RETURN_TYPES = ('MASK',)
    RETURN_NAMES = ('protected_on_original',)
    FUNCTION = 'detect'
    CATEGORY = CATEGORY

    def detect(self, source, enabled=True, description='face', threshold=0.5, instance_index=-1):
        if not enabled:
            return (torch.zeros(tuple(source.shape[:3]), dtype=torch.float32),)
        import nodes
        cls = nodes.NODE_CLASS_MAPPINGS.get('Q21PhotoMask')
        if cls is None:
            raise RuntimeError('Q21PhotoMask is missing. Install the Qwen photo nodes with this workflow.')
        # Detect against ORIGINAL pixels. Empty/failed SAM detection raises, never silently unprotects.
        result = cls().make_mask(source=source, auto_mask=True, sam_prompt=description,
                                threshold=threshold, refine_iterations=2, instance_index=instance_index,
                                search_region='full photo', manual_correction='ignore', expand_pixels=0)
        return (result[0],)


class WBOverlayPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source': ('IMAGE',), 'edit_mask': ('MASK',), 'protection': ('MASK',),
            'mask_report': ('STRING', {'forceInput': True}),
            'resolution': ('INT', {'default': 2048, 'min': 256, 'max': 2048, 'step': 32}),
            'context_pixels': ('INT', {'default': 128, 'min': 0, 'max': 2048, 'step': 16}),
            'protection_grow_pixels': ('INT', {'default': 8, 'min': 0, 'max': 128})}}
    RETURN_TYPES = ('WB_OVERLAY_CROP', 'IMAGE')
    RETURN_NAMES = ('full_resolution_package', 'working_crop')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY

    def prepare(self, source, edit_mask, protection, mask_report, resolution=2048,
                context_pixels=128, protection_grow_pixels=8):
        import nodes
        height, width = source.shape[1:3]
        edit = _mask(edit_mask, height, width, 'Edit mask')
        protected = _mask(protection, height, width, 'Protection mask') > 0
        if not 0 <= protection_grow_pixels <= 128:
            raise ValueError('Protection growth must be 0 to 128 source pixels.')
        if protection_grow_pixels:
            protected = maximum_filter(protected, size=2 * protection_grow_pixels + 1, mode='constant')
        if not np.any((edit > 0) & ~protected):
            raise ValueError('No editable area remains after protection. Review both mask selections.')
        cls = nodes.NODE_CLASS_MAPPINGS.get('Q21PhotoCrop')
        if cls is None:
            raise RuntimeError('Q21PhotoCrop is missing. Install the Qwen photo nodes.')
        data, work = cls().crop(source=source, mask=edit_mask, mask_report=mask_report,
                               resolution=resolution, max_long_side=3072,
                               context_pixels=context_pixels, upscale_small_crops=False)
        data = dict(data)
        data['overlay_protection'] = torch.from_numpy(protected.astype(np.float32)).unsqueeze(0)
        return data, work


class WBOverlayExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'package': ('WB_OVERLAY_CROP',),
            'run_edit': ('BOOLEAN', {'default': False}),
            'filename_prefix': ('STRING', {'default': 'Qwen21_Overlay/Edit'}),
            'feather_pixels': ('INT', {'default': 12, 'min': 0, 'max': 256})},
            'optional': {'generated_crop': ('IMAGE', {'lazy': True})},
            'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ()
    FUNCTION = 'save'
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def check_lazy_status(self, run_edit, generated_crop=None, **kwargs):
        return ['generated_crop'] if run_edit and generated_crop is None else []

    def save(self, package, run_edit=False, filename_prefix='Qwen21_Overlay/Edit',
             feather_pixels=12, generated_crop=None, prompt=None, extra_pnginfo=None):
        import folder_paths
        if not run_edit:
            source = np.clip(_array(package['source'])[0], 0, 1)
            h, w = source.shape[:2]
            edit = _mask(package['effective_mask'], h, w, 'Edit mask')
            protect = _mask(package['overlay_protection'], h, w, 'Protection mask')
            selected = source * (1 - 0.45 * edit[..., None])
            selected[..., 1] += 0.45 * edit
            selected = selected * (1 - 0.65 * protect[..., None])
            selected[..., 0] += 0.65 * protect
            preview = Image.fromarray(np.rint(np.clip(selected, 0, 1) * 255).astype(np.uint8))
            preview.thumbnail((1200, 1200))
            name = 'qwen_overlay_mask_' + uuid.uuid4().hex + '.png'
            preview.save(Path(folder_paths.get_temp_directory()) / name)
            return {'ui': {'images': [{'filename': name, 'subfolder': '', 'type': 'temp'}],
                           'text': ['MASK REVIEW ONLY. Green: edit selection. Red: protected original pixels. '
                                    'Check the face/element coverage, then set run_edit ON.']}}
        if generated_crop is None:
            raise ValueError('Connect the native Qwen image output to generated_crop.')
        composite, rgba, alpha, report = compose_overlay(package, generated_crop, feather_pixels)
        root = folder_paths.get_output_directory()
        folder, stem, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, root, composite.shape[1], composite.shape[0])
        if not Path(folder).resolve().is_relative_to(Path(root).resolve()):
            raise ValueError('Output prefix escapes the ComfyUI output directory.')
        info = PngInfo()
        info.add_text('overlay_audit', json.dumps(report))
        if prompt is not None:
            info.add_text('prompt', json.dumps(prompt))
        for key, value in (extra_pnginfo or {}).items():
            info.add_text(key, json.dumps(value))
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        token = f'{stem}_{counter:05d}_{uuid.uuid4().hex[:8]}'
        results = []
        for suffix, array in [('overlay', rgba), ('composite', composite), ('mask', alpha)]:
            filename = f'{token}_{suffix}.png'
            kwargs = {'icc_profile': profile} if array.ndim == 3 else {}
            Image.fromarray(array).save(Path(folder) / filename, pnginfo=info, compress_level=4, **kwargs)
            results.append({'filename': filename, 'subfolder': subfolder, 'type': 'output'})
        return {'ui': {'images': results, 'text': [json.dumps(report, indent=2)]}}
