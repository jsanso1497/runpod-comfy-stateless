"""MiniMax H3 Ref2VA still-image helpers for the HQ portrait workspace.

The still workflow deliberately reuses the same H3 Ref2VA model stack as video.
H3 generates the native minimum five-frame packet; we decode all five frames and
select one stable, sharp frame for delivery. No second image model is downloaded.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

from . import logic
from . import reference_roles as rr

STILL_ASPECTS = ('Match guide/primary', '9:16', '2:3', '4:5', '1:1', '16:9')
STILL_RESOLUTIONS = ('Native detail (~1 MP)', 'High-res (~2 MP, experimental)', 'Preview (~0.5 MP)')
STILL_QUALITIES = ('High fidelity', 'Standard', 'Fast preview')
STILL_SELECTION = ('Best stable quality', 'Middle frame', 'First frame', 'Last frame')
STILL_ROUTING = ('Still safe swap (pose/scene text-only)', 'All images visual (comparison)')
FIXED_RATIOS = {'9:16': 9/16, '2:3': 2/3, '4:5': 4/5, '1:1': 1.0, '16:9': 16/9}
TARGET_MEGAPIXELS = {
    'Native detail (~1 MP)': 0.98,
    'High-res (~2 MP, experimental)': 2.0,
    'Preview (~0.5 MP)': 0.50,
}


def prompt_policy():
    root = Path(__file__).parent
    return {
        'analysis_prompt.txt': (root/'analysis_prompt.txt').read_text(encoding='utf-8'),
        'still_system_prompt.txt': (root/'still_system_prompt.txt').read_text(encoding='utf-8'),
    }


def cache_key(hashes, instruction, aspect, resolution, quality, model_digest, cfg,
              triggers, variation, routing_mode):
    payload = {
        'v': 1, 'images': hashes, 'instruction': instruction, 'aspect': aspect,
        'resolution': resolution, 'quality': quality, 'model': model_digest,
        'preview_edge': cfg['image_max_edge'], 'context': cfg['context_length'],
        'triggers': triggers, 'variation': int(variation), 'routing_mode': routing_mode,
        'policy': prompt_policy(),
        'options': {k: cfg.get(k) for k in (
            'think', 'temperature', 'top_p', 'top_k', 'analysis_model',
            'analysis_temperature', 'analysis_max_output_tokens', 'max_output_tokens')},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _round32(value):
    return max(32, int(round(float(value)/32.0))*32)


def _ratio_from_image(image):
    if not hasattr(image, 'shape') or len(image.shape) < 3:
        raise ValueError('Reference image dimensions are unavailable.')
    h, w = int(image.shape[1]), int(image.shape[2])
    if h < 1 or w < 1:
        raise ValueError('Reference image dimensions are invalid.')
    return w / h


def aspect_source_index(references, routing):
    """Prefer an explicit pose/scene guide for Match guide/primary geometry."""
    if not references:
        raise ValueError('At least one reference is required.')
    ledger = (routing or {}).get('ledger', [])
    for preferred in ('pose_camera', 'scene'):
        for row in ledger:
            if row.get('role') == preferred and row.get('delivery') != 'ignored':
                return int(row['source_image']) - 1
    native = (routing or {}).get('native_source_indices') or []
    return int(native[0]) if native else 0


def _target_ratio(aspect, source_image):
    if aspect not in STILL_ASPECTS:
        raise ValueError('Unsupported still-image aspect ratio.')
    if aspect == 'Match guide/primary':
        return _ratio_from_image(source_image)
    return FIXED_RATIOS[aspect]


def _canvas_for_ratio(ratio, megapixels):
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('Invalid still-image aspect ratio.')
    pixels = float(megapixels) * 1024 * 1024
    width = math.sqrt(pixels * ratio)
    height = math.sqrt(pixels / ratio)
    # H3 dimensions are a 32-pixel grid. Avoid pathological axes even if a
    # source photograph is extremely panoramic or tall.
    width, height = _round32(width), _round32(height)
    minimum, maximum = 256, 3072
    if min(width, height) < minimum:
        scale = minimum / min(width, height)
        width, height = _round32(width*scale), _round32(height*scale)
    if max(width, height) > maximum:
        scale = maximum / max(width, height)
        width, height = _round32(width*scale), _round32(height*scale)
    return int(width), int(height)


def _exact_crop(width, height, ratio):
    """Largest centered integer crop that closely matches the requested ratio."""
    actual = width / height
    if abs(actual-ratio) < 1e-6:
        return width, height
    if actual > ratio:
        out_h = height
        out_w = max(2, int(round(out_h*ratio)))
    else:
        out_w = width
        out_h = max(2, int(round(out_w/ratio)))
    # PNG does not require even dimensions, but even outputs are friendlier to
    # later video chaining and encoders.
    out_w -= out_w % 2
    out_h -= out_h % 2
    return max(2, out_w), max(2, out_h)


def geometry(aspect, resolution, quality, source_image):
    if resolution not in STILL_RESOLUTIONS:
        raise ValueError('Unsupported still-image resolution preset.')
    if quality not in STILL_QUALITIES:
        raise ValueError('Unsupported still-image quality preset.')
    ratio = _target_ratio(aspect, source_image)
    w, h = _canvas_for_ratio(ratio, TARGET_MEGAPIXELS[resolution])
    ow, oh = _exact_crop(w, h, ratio)
    steps = {'High fidelity': 20, 'Standard': 16, 'Fast preview': 12}[quality]
    ref_size = 'max' if quality == 'High fidelity' else 'match'
    return {
        'task': 'still', 'aspect': aspect, 'resolution': resolution, 'quality': quality,
        'target_ratio': ratio, 'width': w, 'height': h,
        'output_width': ow, 'output_height': oh,
        # Five frames is native H3's minimum packet and the community/default
        # still profile. Only one decoded frame is delivered.
        'length': 5, 'fps': 24, 'actual_seconds': 5/24,
        'steps': steps, 'ref_image_size': ref_size,
        'sampler': 'res_multistep', 'scheduler': 'simple',
    }


def analysis_recipe(aspect, quality):
    """Only the fields the Ollama transport needs before final canvas routing."""
    return {'task': 'still', 'aspect': aspect, 'quality': quality,
            'length': 5, 'fps': 24, 'actual_seconds': 5/24}


def format_prompt(obj, user_direction, recipe, trigger_words=''):
    routing = obj.get('_routing')
    if routing:
        user_direction = rr.remap_text(user_direction, routing['source_to_h3'])
    shot = obj['shot'].strip()
    if '[Shot 1]' not in shot:
        shot = '[Shot 1] ' + shot
    summary = obj['summary'].strip()
    if not summary.startswith('[reference generation]'):
        summary = '[reference generation] ' + summary
    guidance = obj.get('_guidance', [])
    if guidance:
        guide_text = '; '.join(g['target_subject'] + ': ' + g['description'] for g in guidance)
        shot += ('\nSTATIC GUIDE TRANSFER: Apply this geometry/environment to the target subject only: '
                 + guide_text)
    text = '\n\n'.join([
        'subject_definitions:\n' + logic.subject_definitions(obj['references']),
        'summary:\n' + summary,
        'retention_analysis:\n' + (obj['retention'].strip() or 'Preserve the requested subject identity and assigned reference roles.'),
        'detailed_description:\n' + shot,
        'overall_soundscape:\nN/A - still-image delivery; do not invent dialogue or sound-driven action.',
        'non_diegetic_music:\nN/A',
        (f'DELIVERY: ONE finished still image. MiniMax H3 internally renders a five-frame Ref2VA packet at '
         f'{recipe["width"]}x{recipe["height"]}; only one stable decoded frame is saved. '
         'Locked camera, no temporal action, no cuts, no transition, no motion blur unless explicitly requested.'),
        'AUTHORITATIVE USER DIRECTION:\n' + user_direction.strip(),
    ])
    if trigger_words.strip():
        text += '\n\nLoRA trigger words:\n' + trigger_words.strip()
    return text


def crop_frames(images, recipe):
    if images.ndim == 5 and images.shape[0] == 1:
        images = images[0]
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError('Expected decoded H3 frames [frames,height,width,channels].')
    h, w = int(images.shape[1]), int(images.shape[2])
    if (w, h) != (int(recipe['width']), int(recipe['height'])):
        raise ValueError('Decoded H3 still packet does not match the requested canvas.')
    ow, oh = int(recipe['output_width']), int(recipe['output_height'])
    x = max(0, (w-ow)//2); y = max(0, (h-oh)//2)
    return images[:, y:y+oh, x:x+ow, :]


def _minmax(values):
    spread = values.max() - values.min()
    if float(spread.abs()) < 1e-8:
        return values*0
    return (values-values.min())/spread


def stable_quality_index(images, max_side=256):
    """Select a sharp, clean frame that is stable against its neighbors."""
    import torch
    import torch.nn.functional as F
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError('A non-empty IMAGE frame batch is required.')
    n = int(images.shape[0])
    if n == 1:
        return 0, 1.0
    x = images[..., :3].movedim(-1, 1).float().clamp(0, 1)
    h, w = x.shape[-2:]
    scale = min(1.0, float(max_side)/max(h, w))
    if scale < 1.0:
        x = F.interpolate(x, size=(max(16, round(h*scale)), max(16, round(w*scale))),
                          mode='bilinear', align_corners=False, antialias=True)
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(dim=(1,2,3))
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(dim=(1,2,3))
    sharp = _minmax(dx + dy)
    contrast = _minmax(x.flatten(1).std(dim=1))
    mean = x.flatten(1).mean(dim=1)
    exposure = (1.0 - (mean-0.5).abs()*2.0).clamp(0, 1)
    delta = torch.empty(n, device=x.device, dtype=x.dtype)
    delta[0] = (x[0]-x[1]).abs().mean()
    delta[-1] = (x[-1]-x[-2]).abs().mean()
    if n > 2:
        delta[1:-1] = 0.5*(x[1:-1]-x[:-2]).abs().mean(dim=(1,2,3)) + 0.5*(x[1:-1]-x[2:]).abs().mean(dim=(1,2,3))
    stability = 1.0 - _minmax(delta)
    middle = (n-1)/2.0
    center = torch.tensor([1.0-abs(i-middle)/max(1.0,middle) for i in range(n)], device=x.device, dtype=x.dtype)
    score = 0.58*sharp + 0.17*contrast + 0.10*exposure + 0.13*stability + 0.02*center
    index = int(torch.argmax(score).item())
    return index, float(score[index].item())


def select_frame(images, method):
    if method not in STILL_SELECTION:
        raise ValueError('Unknown still-frame selection method.')
    n = int(images.shape[0])
    if n < 1:
        raise ValueError('No decoded H3 frames are available.')
    if method == 'Best stable quality':
        index, score = stable_quality_index(images)
    elif method == 'Middle frame':
        index, score = n//2, 1.0
    elif method == 'First frame':
        index, score = 0, 1.0
    else:
        index, score = n-1, 1.0
    return images[index:index+1].clone(), index, score
