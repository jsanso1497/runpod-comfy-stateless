"""Qwen 2.1 photo geometry, adapted from the supplied FLUX Photo 1.0.1 code.
CPU-only protected compositing; model alignment is 32-pixel padding.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CropPlan:
    source_width: int
    source_height: int
    x: int
    y: int
    crop_width: int
    crop_height: int
    scaled_width: int
    scaled_height: int
    work_width: int
    work_height: int
    mode: str

    @property
    def resampled(self) -> bool:
        return (self.scaled_width, self.scaled_height) != (self.crop_width, self.crop_height)


def image_check(image: torch.Tensor, name: str = 'image') -> None:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
        raise ValueError(f'{name}: one RGB photo is required, not a batch/video/RGBA tensor.')
    if image.shape[1] < 1 or image.shape[2] < 1:
        raise ValueError(f'{name}: empty image.')


def mask_check(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.detach().to(device='cpu', dtype=torch.float32)
    if not torch.isfinite(mask).all():
        raise ValueError('Mask contains NaN or infinity.')
    if torch.any((mask < 0) | (mask > 1)):
        raise ValueError('Mask values must be in [0, 1].')
    # ComfyUI LoadImage intentionally emits a blank 64x64 placeholder mask when
    # a normal RGB image has no alpha/mask. Detect that case before the strict
    # geometry check so the error tells the user what is actually missing.
    if not torch.any(mask > 0):
        raise ValueError('Empty mask: right-click the SOURCE Load Image, open MaskEditor, '
                         'paint the edit area, then Save. A normal RGB image without a saved '
                         'mask may arrive from ComfyUI as a blank 64x64 placeholder mask.')
    if tuple(mask.shape) != (1, height, width):
        raise ValueError('Mask dimensions must exactly match the source. Reopen MaskEditor on the '
                         'SOURCE image, repaint/save the mask, and do not resize the source after masking. '
                         'No non-empty mask resizing is done because that could shift the protected boundary.')
    return mask


def fit_dimensions(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Aspect-preserving fit, never upscale. Alignment is padding, never stretching."""
    if min(width, height, long_side) <= 0:
        raise ValueError('Dimensions and size limit must be positive.')
    scale = min(1.0, long_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_rgb(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Float32 Lanczos per channel. Do not quantize the working crop to 8 bit."""
    if tuple(image.shape[1:3]) == (height, width):
        return image.detach().to(device='cpu', dtype=torch.float32)
    arr = image.detach().to(device='cpu', dtype=torch.float32)[0].numpy()
    channels = [np.asarray(Image.fromarray(arr[..., i]).resize(
        (width, height), Image.Resampling.LANCZOS), dtype=np.float32) for i in range(3)]
    return torch.from_numpy(np.stack(channels, axis=-1).copy()).unsqueeze(0).clamp_(0, 1)


def pad_rgb(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    dh, dw = height - image.shape[1], width - image.shape[2]
    if min(dh, dw) < 0:
        raise ValueError('Padding cannot crop an image.')
    if not dh and not dw:
        return image
    return F.pad(image.movedim(-1, 1), (0, dw, 0, dh), mode='replicate').movedim(1, -1)


def budget_dimensions(width: int, height: int, budget_edge: int = 2048,
                      long_side: int = 3072, upscale: bool = False) -> tuple[int, int, int, int]:
    """Fit area <= budget_edge**2 and padded long side <= limit; no stretching.

    Return scaled width/height and 32-aligned padded width/height.
    Padding, not forced aspect-ratio rounding, provides model alignment.
    """
    if min(width, height, budget_edge, long_side) <= 0:
        raise ValueError('Dimensions and limits must be positive.')
    if budget_edge % 32 or long_side % 32:
        raise ValueError('Resolution and long-side limit must be multiples of 32.')
    cap = budget_edge ** 2
    upper = min(math.sqrt(cap / (width * height)), long_side / max(width, height))
    if not upscale:
        upper = min(1.0, upper)
    def dims(scale):
        sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
        return sw, sh, math.ceil(sw / 32) * 32, math.ceil(sh / 32) * 32
    def fits(values):
        return values[2] * values[3] <= cap and max(values[2:]) <= long_side
    candidate = dims(upper)
    if fits(candidate):
        return candidate
    lo, hi = 0.0, upper
    for _ in range(60):
        mid = (lo + hi) / 2
        if fits(dims(mid)):
            lo = mid
        else:
            hi = mid
    return dims(lo)


def prepare_reference(image: torch.Tensor, budget_edge: int = 1536,
                      long_side: int = 2048) -> torch.Tensor:
    image_check(image, 'reference')
    sw, sh, ww, wh = budget_dimensions(image.shape[2], image.shape[1], budget_edge, long_side)
    return pad_rgb(resize_rgb(image, sw, sh), ww, wh)


def crop_photo(source: torch.Tensor, mask: torch.Tensor, context_pixels: int = 128,
               budget_edge: int = 2048, max_crop_side: int = 3072,
               upscale_small: bool = False) -> tuple[dict, torch.Tensor, torch.Tensor]:
    image_check(source, 'source')
    source = source.detach().cpu()
    height, width = source.shape[1:3]
    mask = mask_check(mask, height, width)
    if not 0 <= context_pixels <= 2048:
        raise ValueError('Context must be between 0 and 2048 source pixels.')
    active = mask[0] > 0
    ys = torch.where(active.any(dim=1))[0]
    xs = torch.where(active.any(dim=0))[0]
    x = max(0, int(xs[0]) - context_pixels)
    y = max(0, int(ys[0]) - context_pixels)
    x2 = min(width, int(xs[-1]) + 1 + context_pixels)
    y2 = min(height, int(ys[-1]) + 1 + context_pixels)
    cw, ch = x2 - x, y2 - y
    sw, sh, ww, wh = budget_dimensions(cw, ch, budget_edge, max_crop_side, upscale_small)
    plan = CropPlan(width, height, x, y, cw, ch, sw, sh, ww, wh, 'area_fit_32')
    original_crop = source[:, y:y2, x:x2, :]
    paste_mask = mask[:, y:y2, x:x2].clone()
    work = pad_rgb(resize_rgb(original_crop, sw, sh), ww, wh).clone()
    work_mask = paste_mask[:, None]
    if (sw, sh) != (cw, ch):
        if sw <= cw and sh <= ch:
            work_mask = F.adaptive_max_pool2d(work_mask, (sh, sw))
        else:
            work_mask = F.interpolate(work_mask, (sh, sw), mode='nearest')
    work_mask = F.pad(work_mask[:, 0], (0, ww - sw, 0, wh - sh))
    package = {'plan': plan, 'source': source, 'paste_mask': paste_mask,
               'effective_mask': mask, 'work_image': work}
    return package, work, work_mask


def inward_alpha(mask: torch.Tensor, feather_pixels: int, plan: CropPlan) -> torch.Tensor:
    if feather_pixels < 0:
        raise ValueError('Feather cannot be negative.')
    if feather_pixels == 0:
        return mask.clone()
    support = mask[0].numpy() > 0
    # Extend at actual photo edges. Do not fade edits merely because they touch a frame edge.
    border = feather_pixels + 1
    extended = np.pad(support, border, mode='edge')
    if plan.x > 0:
        extended[:, :border] = False
    if plan.y > 0:
        extended[:border, :] = False
    if plan.x + plan.crop_width < plan.source_width:
        extended[:, -border:] = False
    if plan.y + plan.crop_height < plan.source_height:
        extended[-border:, :] = False
    if extended.all():
        return mask.clone()
    distance = distance_transform_edt(extended)[border:-border, border:-border]
    t = np.clip(distance / feather_pixels, 0.0, 1.0).astype(np.float32)
    t = t * t * (3.0 - 2.0 * t)
    return mask * torch.from_numpy(t).unsqueeze(0)


def stitch_photo(package: dict, generated: torch.Tensor, feather_pixels: int = 16,
                 boundary_color_strength: float = 0.0) -> tuple[torch.Tensor, str]:
    image_check(generated, 'generated crop')
    plan: CropPlan = package['plan']
    if tuple(generated.shape[1:3]) != (plan.work_height, plan.work_width):
        raise ValueError(f'Generated crop must be {plan.work_width}x{plan.work_height}, '
                         'not an arbitrary resize. Refusing a shifted/misaligned stitch.')
    if not 0 <= boundary_color_strength <= 1:
        raise ValueError('Boundary color strength must be in [0, 1].')
    source, mask = package['source'], package['paste_mask']
    patch = generated.detach().to(device='cpu', dtype=torch.float32)
    if not torch.isfinite(patch).all():
        raise ValueError('Generated crop contains NaN/infinity; no output saved.')
    # Remove right/bottom model padding BEFORE undoing the optional crop resize.
    patch = patch[:, :plan.scaled_height, :plan.scaled_width, :]
    patch = resize_rgb(patch, plan.crop_width, plan.crop_height)
    original = source[:, plan.y:plan.y + plan.crop_height,
                      plan.x:plan.x + plan.crop_width, :]
    color_shift = [0.0, 0.0, 0.0]
    if boundary_color_strength:
        support = mask[0].numpy() > 0
        ring = maximum_filter(support, size=65, mode='constant') & ~support
        if int(ring.sum()) >= 64:
            delta = (original[0] - patch[0]).numpy()[ring]
            shift = np.clip(np.median(delta, axis=0), -0.04, 0.04) * boundary_color_strength
            color_shift = shift.tolist()
            patch = (patch + torch.from_numpy(shift)).clamp(0, 1)
    alpha = inward_alpha(mask, feather_pixels, plan).unsqueeze(-1)
    blended = original + alpha * (patch.to(original.dtype) - original)
    # Strict replacement boundary: neither VAE drift nor mask grow can escape this gate.
    blended = torch.where(mask.unsqueeze(-1) > 0, blended, original)
    outside = (mask == 0).unsqueeze(-1).expand_as(original)
    differences = torch.count_nonzero((blended != original) & outside).item()
    if differences:
        raise AssertionError('Protected source pixels changed. Refusing to save.')
    result = source.clone()
    result[:, plan.y:plan.y + plan.crop_height, plan.x:plan.x + plan.crop_width, :] = blended
    report = {
        'source_size': [plan.source_width, plan.source_height],
        'output_size': [result.shape[2], result.shape[1]],
        'source_crop_xywh': [plan.x, plan.y, plan.crop_width, plan.crop_height],
        'model_processing_size': [plan.work_width, plan.work_height],
        'crop_resampled': plan.resampled,
        'processing_mode': plan.mode,
        'outside_mask_changed_channels': int(differences),
        'outside_crop': 'source copied without processing',
        'feather': 'inward only', 'feather_pixels': feather_pixels,
        'boundary_color_strength': boundary_color_strength,
        'boundary_rgb_shift': color_shift,
        'export': 'lossless 8-bit RGB PNG; input must be sRGB',
        'model': 'Qwen-Image-2.1',
        'internal_generation': 'whole working crop; hard mask used at final stitch',
        'identity_or_generated_detail_guaranteed': False,
    }
    return result, json.dumps(report, indent=2)


def to_uint8(image: torch.Tensor) -> np.ndarray:
    image_check(image)
    array = image.detach().to(device='cpu', dtype=torch.float32)[0].numpy()
    if not np.isfinite(array).all():
        raise ValueError('Image contains NaN/infinity.')
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
