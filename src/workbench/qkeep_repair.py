"""Opaque conditioning and honest, protected QKEEP1 repair compositing.

Background prefill is non-generative and can be visibly imperfect. It never
turns absent model RGB into a successful AI reconstruction. Draft fallbacks
are explicitly labeled, audited, and accompanied by diagnostic images.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
import torch
import torch.nn.functional as F

DEFAULT_INSTRUCTIONS = (
    'Refine the rough background-filled patches in this photograph into realistic, '
    'continuous surrounding scenery. Preserve the remaining visible person, their '
    'face, pose, clothing and location. Match the existing light, perspective, '
    'texture, grain and focus. Complete any newly exposed background and local '
    'edges so this looks like a normal photograph taken with only that person '
    'present. Produce one fully opaque photographic frame.'
)


def prefill(work, erase_source, donor_source, sw, sh):
    """Fill only the edit conditioning region, using background-only donors.

    Unlike Telea using the raw photo, this cannot sample the keeper or other
    detected people as fill donors. The final source photo is never changed.
    """
    def reduce_mask(mask, all_covered=False):
        array = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(
            (sw, sh), Image.Resampling.BOX))
        return array == 255 if all_covered else array > 0
    erase = reduce_mask(erase_source)
    donors = reduce_mask(donor_source, all_covered=True) & ~erase
    if not erase.any():
        return work, {'prefill_method': 'none', 'prefill_work_pixels': 0}
    if not donors.any():
        raise ValueError('No visible background remains for a safe prefill. Increase context or use a less crowded crop.')
    raw = work[0, :sh, :sw, :3].detach().float().cpu().numpy()
    indices = distance_transform_edt(~donors, return_distances=False, return_indices=True)
    # Build a background-only plate BEFORE smoothing, so retained clothing,
    # skin and other excluded pixels cannot bleed into the replacement.
    background = raw[indices[0], indices[1]].copy()
    background = gaussian_filter(background, sigma=(1.5, 1.5, 0), mode='nearest')
    filled = raw.copy()
    filled[erase] = background[erase]
    padded = torch.from_numpy(np.clip(filled, 0, 1).copy())[None].movedim(-1, 1)
    padded = F.pad(padded, (0, work.shape[2]-sw, 0, work.shape[1]-sh), mode='replicate').movedim(1, -1)
    return padded, {'prefill_method': 'background-only nearest/smoothed, non-generative',
                    'prefill_work_pixels': int(erase.sum()),
                    'prefill_background_donor_pixels': int(donors.sum())}


def compose(package, generated, feather_pixels=12, transparency_policy='review draft'):
    from .qwen_overlay import compose_overlay
    if transparency_policy not in ('review draft', 'stop on transparency'):
        raise ValueError('Unknown QKEEP1 transparency policy.')
    if package['report']['no_op']:
        generated = package['work_image']
    if generated is None:
        raise ValueError('Connect the Qwen output before enabling Run edit.')
    if (not torch.is_tensor(generated) or generated.ndim != 4 or generated.shape[0] != 1 or
            generated.shape[-1] not in (3, 4)):
        raise ValueError('Qwen must return one RGB/RGBA photographic crop.')
    p = package['plan']
    if tuple(generated.shape[1:3]) != (p.work_height, p.work_width):
        raise ValueError('Qwen output dimensions changed. Keep native resolution=0 and custom_size OFF.')
    raw = generated.detach().float().cpu()
    if not torch.isfinite(raw).all().item():
        raise ValueError('Qwen returned non-finite pixels.')
    if raw.min().item() < -0.0001 or raw.max().item() > 1.0001:
        raise ValueError('Qwen output is not a normalized image.')
    raw = raw.clamp(0, 1)
    fraction = 0.0
    rgb = raw[..., :3]
    if raw.shape[-1] == 4:
        a = raw[..., 3:4]
        # Never reveal hidden purple RGB or the original unwanted people.
        confidence = ((a-0.97)/0.03).clamp(0, 1)
        confidence = confidence*confidence*(3-2*confidence)
        c = confidence[0, :p.scaled_height, :p.scaled_width, 0].numpy()
        c = np.asarray(Image.fromarray(c).resize((p.crop_width, p.crop_height), Image.Resampling.BILINEAR))
        editable = package['paste_mask'][0].numpy() > 0
        if editable.any():
            fraction = float(np.mean(c[editable] < 0.999))
        if fraction > 0.01 and transparency_policy == 'stop on transparency':
            raise ValueError('Qwen still produced transparency. No final photo was saved. '
                             'Use review draft to inspect the coarse fallback and diagnostics.')
        base = package['work_image'].detach().float().cpu()
        if base.shape != rgb.shape:
            raise ValueError('Prefill image does not match the generated crop.')
        rgb = rgb*confidence + base*(1-confidence)
    composite, rgba, alpha, audit = compose_overlay(package, rgb, feather_pixels)
    original = np.rint(package['source'][0].numpy()*255).astype(np.uint8)
    for key in ('manual_protection', 'keep_protection'):
        mask = package[key]
        if np.any(rgba[..., 3][mask]) or not np.array_equal(composite[mask], original[mask]):
            raise AssertionError(key+': original protected pixels changed; no files saved.')
    report = dict(package['report'])
    report.update(audit)
    report.update(generated_channels=int(raw.shape[-1]),
        generated_low_alpha_edit_fraction=round(fraction, 6),
        opaque_background_prefill_used=fraction > 0,
        fallback_review_required=fraction > 0,
        completion_status='DRAFT_FALLBACK_REVIEW_REQUIRED' if fraction > 0 else 'GENERATED_VISUAL_REVIEW_REQUIRED',
        manual_changed_channels=0, kept_person_changed_channels=0,
        primary_output='full-resolution opaque RGB composite',
        separate_overlay='full-resolution straight RGBA; protected pixels transparent',
        semantic_repair_verified=False)
    return composite, rgba, alpha, report
