"""CPU-only geometry for protected photo editing. No model or global color edits."""
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
    if tuple(mask.shape) != (1, height, width):
        raise ValueError('Mask dimensions must exactly match the source. Right-click the source '
                         'Load Image, open MaskEditor, paint and save the mask. No mask resizing is done.')
    mask = mask.detach().to(device='cpu', dtype=torch.float32)
    if not torch.isfinite(mask).all():
        raise ValueError('Mask contains NaN or infinity.')
    if torch.any((mask < 0) | (mask > 1)):
        raise ValueError('Mask values must be in [0, 1].')
    if not torch.any(mask > 0):
        raise ValueError('Empty mask: paint and save a mask on the SOURCE Load Image node first.')
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


def prepare_reference(image: torch.Tensor, long_side: int) -> torch.Tensor:
    image_check(image, 'reference')
    w, h = fit_dimensions(image.shape[2], image.shape[1], long_side)
    return pad_rgb(resize_rgb(image, w, h), math.ceil(w / 16) * 16, math.ceil(h / 16) * 16)


def crop_photo(source: torch.Tensor, mask: torch.Tensor, context_pixels: int = 128,
               processing: str = 'fit', max_crop_side: int = 2048,
               sampling_mask_grow: int = 16) -> tuple[dict, torch.Tensor, torch.Tensor]:
    image_check(source, 'source')
    if source.device.type != 'cpu':
        # The full canvas belongs in CPU memory, not on the diffusion GPU.
        source = source.detach().cpu()
    height, width = source.shape[1:3]
    mask = mask_check(mask, height, width)
    if processing not in ('fit', 'native'):
        raise ValueError('processing must be fit or native.')
    if not 0 <= context_pixels <= 2048 or not 0 <= sampling_mask_grow <= 256:
        raise ValueError('Context/mask-grow setting is out of range.')
    if max_crop_side < 16 or max_crop_side % 16:
        raise ValueError('max_crop_side must be a positive multiple of 16.')
    # Axis reductions avoid allocating an N-by-2 list for a multi-megapixel mask.
    active = mask[0] > 0
    ys = torch.where(active.any(dim=1))[0]
    xs = torch.where(active.any(dim=0))[0]
    padding = max(context_pixels, sampling_mask_grow)
    x = max(0, int(xs[0]) - padding)
    y = max(0, int(ys[0]) - padding)
    x2 = min(width, int(xs[-1]) + 1 + padding)
    y2 = min(height, int(ys[-1]) + 1 + padding)
    cw, ch = x2 - x, y2 - y
    if processing == 'native':
        sw, sh = cw, ch
    else:
        sw, sh = fit_dimensions(cw, ch, max_crop_side)
    ww, wh = math.ceil(sw / 16) * 16, math.ceil(sh / 16) * 16
    if processing == 'native' and max(ww, wh) > max_crop_side:
        raise ValueError(f'Native crop is {ww}x{wh}, exceeding the {max_crop_side}px limit. '
                         'Reduce context, split the edit, or explicitly select fit mode. '
                         'Native mode never silently downsizes.')
    plan = CropPlan(width, height, x, y, cw, ch, sw, sh, ww, wh, processing)
    original_crop = source[:, y:y2, x:x2, :]
    paste_mask = mask[:, y:y2, x:x2].clone()
    work = pad_rgb(resize_rgb(original_crop, sw, sh), ww, wh)
    noise = paste_mask[0].numpy()
    if sampling_mask_grow:
        noise = maximum_filter(noise, size=2 * sampling_mask_grow + 1, mode='constant')
    noise = torch.from_numpy(np.array(noise, dtype=np.float32, copy=True))[None, None]
    if (sw, sh) != (cw, ch):
        # Max pooling prevents thin brush strokes disappearing during a downscale.
        noise = F.adaptive_max_pool2d(noise, (sh, sw))
    noise = F.pad(noise[:, 0], (0, ww - sw, 0, wh - sh))
    package = {'plan': plan, 'source': source, 'paste_mask': paste_mask}
    return package, work, noise


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
        'identity_or_generated_detail_guaranteed': False,
    }
    return result, json.dumps(report, indent=2)


def to_uint8(image: torch.Tensor) -> np.ndarray:
    image_check(image)
    array = image.detach().to(device='cpu', dtype=torch.float32)[0].numpy()
    if not np.isfinite(array).all():
        raise ValueError('Image contains NaN/infinity.')
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
