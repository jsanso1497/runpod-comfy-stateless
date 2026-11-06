"""Pure planning rules. No weights, downloads, UI code, or GPU work."""
import math
import re

NONE = '(none)'
MODES = ['Auto', 'Generate', 'Edit', 'References']
PRESETS = ['Quality - 50 steps, no Turbo', 'Developer fast - 20 steps, Turbo 0.38']
ASPECTS = ['1:1', '16:9', '9:16', '2:3', '3:2', '4:5', '5:4', '4:3', '3:4']
FL2VA = 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'
REF2VA = 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'
TURBO = 'minimax_h3_turbo_v4_step600_ema.safetensors'
ENCODER = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
VAE = 'minimax_h3_video_vae_int8_convrot.safetensors'


def resolve_mode(mode, count):
    if mode not in MODES:
        raise ValueError('Choose Auto, Generate, Edit, or References.')
    if not isinstance(count, int) or not 0 <= count <= 9:
        raise ValueError('Use zero to nine still-image references.')
    if mode == 'Auto':
        return 'Generate' if count == 0 else 'Edit' if count == 1 else 'References'
    if mode == 'Generate' and count:
        raise ValueError('Generate uses no photos. Clear/disable the photos or select Auto/References.')
    if mode == 'Edit' and count != 1:
        raise ValueError('Edit requires exactly one active photo. For several photos choose References.')
    if mode == 'References' and count == 0:
        raise ValueError('References requires at least one active photo.')
    return mode


def dimensions(ratio, megapixels):
    if not math.isfinite(float(ratio)) or not 1/64 <= ratio <= 64:
        raise ValueError('The photo aspect ratio is too extreme. Crop it before using it as the edit canvas.')
    if not math.isfinite(float(megapixels)) or not 0.25 <= megapixels <= 8:
        raise ValueError('Output size must be between 0.25 and 8 megapixels.')
    h = math.sqrt(megapixels * 1024 * 1024 / ratio)
    w = h * ratio
    scale = min(1., 4096 / max(w, h))
    return tuple(max(64, min(4096, round(v * scale / 32) * 32)) for v in (w, h))


def plan_values(mode, count, aspect_ratio, megapixels, edit_megapixels, preset,
                source_size=None):
    selected = resolve_mode(mode, count)
    if aspect_ratio not in ASPECTS:
        raise ValueError('Unsupported output aspect ratio.')
    if preset not in PRESETS:
        raise ValueError('Unknown sampling preset.')
    # No FL2VA Turbo adapter is silently applied to a REF2VA checkpoint.
    if selected == 'References' and preset == PRESETS[1]:
        raise ValueError('Reference generation uses REF2VA. Select Quality; the supplied Turbo recipe is FL2VA-only.')
    if selected == 'Edit':
        if not source_size or min(source_size) < 1:
            raise ValueError('Edit needs a valid source image.')
        ratio = source_size[0] / source_size[1]
        width, height = dimensions(ratio, edit_megapixels)
    else:
        a, b = map(int, aspect_ratio.split(':'))
        width, height = dimensions(a / b, megapixels)
    return {'mode': selected, 'count': count, 'width': width, 'height': height,
            'steps': 20 if preset == PRESETS[1] else 50,
            'turbo_strength': 0.38 if preset == PRESETS[1] else 0.,
            'model': REF2VA if selected == 'References' else FL2VA,
            'precision': 'INT8/ConvRot', 'preset': preset}


def check_prompt(prompt, count):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Enter a description or an edit instruction in the prompt node.')
    for value in re.findall(r'<Picture\s+(\d+)>', prompt, flags=re.IGNORECASE):
        if not 1 <= int(value) <= count:
            raise ValueError('The prompt names <Picture %s>, but only %s active reference(s) are loaded.' % (value, count))


def reference_size(width, height, megapixels):
    if min(width, height) < 1 or not math.isfinite(megapixels) or not 0.25 <= megapixels <= 8:
        raise ValueError('Invalid reference dimensions or megapixel cap.')
    scale = min(1., math.sqrt(megapixels * 1024 * 1024 / (width * height)))
    # Only alignment can round a very small image upward. It does not synthesize detail.
    return tuple(max(32, round(v * scale / 32) * 32) for v in (width, height))
