"""Pure routing and bounds for H3 photo/video still generation. No model loads."""
import math
import re
from .logic import dimensions, reference_size, FL2VA, REF2VA, TURBO, ASPECTS

VERSION = '2026-11-07.2'
MEDIA_MODES = ['Auto', 'Generate', 'Edit', 'References', 'Replace person']
MEDIA_PRESETS = ['Quality - 50 steps, no Turbo',
                 'REF2VA Turbo - 8 steps',
                 'Developer FL2VA fast - 20 steps']
REF_TURBO = 'minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'


def aligned_frames(count):
    if count < 5:
        raise ValueError('The selected video section contains fewer than five usable 24 fps frames.')
    return 5 + 17 * ((int(count) - 5) // 17)


def video_dimensions(width, height, megapixels):
    if width < 1 or height < 1 or not math.isfinite(megapixels) or not .1 <= megapixels <= 1.:
        raise ValueError('Video reference cap must be 0.1 to 1.0 MP and dimensions must be positive.')
    # Stay below the native H3 reference-video canvas ceiling. Never stretch to the output canvas.
    scale = min(1., math.sqrt(min(megapixels * 1024 * 1024, 768 * 1344) / (width * height)),
                2048. / max(width, height))
    return tuple(max(32, int(v * scale) // 32 * 32) for v in (width, height))


def media_plan(mode, photo_count, video_count, aspect, generation_mp, edit_mp,
               preset, source_size=None, target_active=False, protect_background=False):
    if mode not in MEDIA_MODES or preset not in MEDIA_PRESETS:
        raise ValueError('Unsupported H3 media mode or sampling preset.')
    if not 0 <= photo_count <= 9 or video_count not in (0, 1):
        raise ValueError('Use at most nine photographs and one short video.')
    if mode == 'Auto':
        selected = 'References' if video_count or photo_count > 1 else 'Edit' if photo_count else 'Generate'
    else:
        selected = mode
    if selected == 'Generate' and (photo_count or video_count):
        raise ValueError('Generate requires no active media. Disable the photos/video or choose References.')
    if selected == 'Edit' and (photo_count != 1 or video_count):
        raise ValueError('Edit uses exactly one photo and no video. Use References or Replace person for additional media.')
    if selected == 'References' and not (photo_count or video_count):
        raise ValueError('Upload at least one photo or enable a video for References.')
    if selected == 'Replace person':
        if not target_active:
            raise ValueError('Replace person requires the target photograph in slot 1.')
        if photo_count < 2 and not video_count:
            raise ValueError('Add an identity photo in slots 2-9, or a video, to replace the person in image 1.')
    if protect_background and selected != 'Replace person':
        raise ValueError('Background mask protection is only for Replace person mode.')
    if aspect not in ASPECTS:
        raise ValueError('Unsupported output aspect ratio.')
    if selected in ('Edit', 'Replace person'):
        if not source_size or min(source_size) < 1:
            raise ValueError('A valid target photograph is required.')
        ratio = source_size[0] / source_size[1]
        width, height = dimensions(ratio, edit_mp)
    else:
        a, b = map(int, aspect.split(':'))
        width, height = dimensions(a / b, generation_mp)
    is_ref = selected in ('References', 'Replace person')
    if preset == MEDIA_PRESETS[1] and not is_ref:
        raise ValueError('REF2VA Turbo is for References / Replace person. Use Quality for text generation or one-photo Edit.')
    if preset == MEDIA_PRESETS[2] and is_ref:
        raise ValueError('The developer FL2VA recipe is not for REF2VA. Choose Quality or REF2VA Turbo.')
    steps = 8 if preset == MEDIA_PRESETS[1] else 20 if preset == MEDIA_PRESETS[2] else 50
    return dict(mode=selected, count=photo_count, video_count=video_count, width=width, height=height,
                model=REF2VA if is_ref else FL2VA, steps=steps, precision='INT8/ConvRot', preset=preset,
                sampler='euler' if preset == MEDIA_PRESETS[1] else 'er_sde', scheduler='simple',
                adapter=REF_TURBO if preset == MEDIA_PRESETS[1] else TURBO if preset == MEDIA_PRESETS[2] else '',
                adapter_strength=1. if preset == MEDIA_PRESETS[1] else .38 if preset == MEDIA_PRESETS[2] else 0.,
                # Legacy model helper must not also add its FL2VA adapter.
                turbo_strength=0., protect_background=bool(protect_background), version=VERSION)


def validate_media_prompt(prompt, photos, videos):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Enter a still-image description or edit instruction.')
    for kind, limit in [('Picture', photos), ('Video', videos)]:
        for number in re.findall(r'<' + kind + r'\s+(\d+)>', prompt, re.I):
            if not 1 <= int(number) <= limit:
                raise ValueError('Prompt uses <%s %s> but only %s %s reference(s) are active.' %
                                 (kind, number, limit, kind.lower()))
    if re.search(r'<Audio\s+\d+>', prompt, re.I):
        raise ValueError('This still workflow intentionally ignores audio; remove Audio reference tags.')
