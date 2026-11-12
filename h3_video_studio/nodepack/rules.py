"""Pure routing and geometry rules for H3 Video Still Studio. No GPU imports."""
from __future__ import annotations
import math
import re

VERSION = '2026-11-11.1'
ENGINES = ['MiniMax H3', 'Eros Beta 7 Fix1', 'P-H3 Image']
PRECISIONS = ['INT8 HQ', 'BF16 diffusion', 'BF16 full']
MODES = ['Auto', 'Generate', 'Edit', 'References', 'Replace person']
ASPECTS = ['1:1', '2:3', '3:2', '4:5', '5:4', '9:16', '16:9', '3:4', '4:3']
NONE = '(none)'
PH3_SIGMAS = [1., .984753, .967905, .951688, .925372, .881923, .810083, .528718, 0.]


def dimensions(width, height, megapixels, down_only=False):
    if min(width, height) <= 0 or not math.isfinite(megapixels) or not .1 <= megapixels <= 8:
        raise ValueError('Use positive dimensions and 0.1 to 8 output megapixels.')
    scale = math.sqrt(megapixels * 1024 * 1024 / (width * height))
    if down_only:
        scale = min(1., scale)
    return tuple(max(32, int(round(v * scale / 32)) * 32) for v in (width, height))


def crop_box(width, height, left=0., top=0., right=0., bottom=0.):
    values = (left, top, right, bottom)
    if any(not math.isfinite(v) or not 0 <= v < 100 for v in values):
        raise ValueError('Crop margins must be finite percentages from 0 to less than 100.')
    if left + right >= 95 or top + bottom >= 95:
        raise ValueError('The crop must retain at least five percent in each direction.')
    box = (round(width * left / 100), round(height * top / 100),
           round(width * (1 - right / 100)), round(height * (1 - bottom / 100)))
    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
        raise ValueError('The remaining crop is smaller than 32 pixels.')
    return box


def frame_times(duration, count, explicit=''):
    if not math.isfinite(duration) or not .25 <= duration <= 15:
        raise ValueError('Video section duration must be 0.25 to 15 seconds.')
    if explicit.strip():
        try:
            times = [float(x.strip()) for x in explicit.split(',')]
        except ValueError as error:
            raise ValueError('Enter comma-separated seconds relative to the selected section.') from error
    else:
        if not 1 <= count <= 8:
            raise ValueError('Choose 1 to 8 video keyframes.')
        # Sample bin centers, never the exact EOF or an assumed duplicate last frame.
        times = [(i + .5) * duration / count for i in range(count)]
    if not 1 <= len(times) <= 8 or len(set(times)) != len(times):
        raise ValueError('Provide 1 to 8 distinct keyframe timestamps.')
    if any(not math.isfinite(t) or t < 0 or t >= duration for t in times):
        raise ValueError('Every keyframe time must be inside the selected video section.')
    return times


def use_frames(engine, strategy):
    if engine not in ENGINES or strategy not in ('Auto', 'Keyframes for all engines'):
        raise ValueError('Unknown model or video-reference strategy.')
    return engine == 'P-H3 Image' or strategy == 'Keyframes for all engines'


def resolve_mode(mode, photo_slots, has_video):
    if mode not in MODES:
        raise ValueError('Unknown generation mode.')
    n = len(photo_slots)
    selected = ('References' if has_video or n > 1 else 'Edit' if n else 'Generate') if mode == 'Auto' else mode
    if selected == 'Generate' and (n or has_video):
        raise ValueError('Generate requires no active media. Select References for a new image based on media.')
    if selected == 'Edit' and (n != 1 or has_video):
        raise ValueError('Edit takes one photo only. Use References or Replace person with a video.')
    if selected == 'References' and not (n or has_video):
        raise ValueError('References needs at least one enabled photo or video.')
    if selected == 'Replace person' and (1 not in photo_slots or (n < 2 and not has_video)):
        raise ValueError('Replace person needs the target in photo slot 1 and at least one identity photo or video.')
    return selected


def reference_budget(engine, photo_count, frame_count):
    maximum = 8 if engine == 'P-H3 Image' else 9
    if photo_count + frame_count > maximum:
        raise ValueError('%s accepts at most %d image references here. You selected %d photos plus %d video keyframes. Reduce the keyframe count or disable photos; none were silently dropped.' %
                         (engine, maximum, photo_count, frame_count))


def translate_prompt(prompt, photo_count, frame_count, native_video, ph3=False):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Enter a prompt. No prompt enhancer is installed.')
    if re.search(r'<Audio\s+\d+>', prompt, re.I):
        raise ValueError('This still-image workspace ignores audio. Remove Audio tags.')
    def video_tag(match):
        if int(match.group(1)) != 1 or not (native_video or frame_count):
            raise ValueError('The prompt references an unavailable video.')
        if native_video:
            return '<Video 1>'
        return ', '.join('<Picture %d>' % i for i in range(photo_count + 1, photo_count + frame_count + 1))
    prompt = re.sub(r'<Video\s+(\d+)>', video_tag, prompt.strip(), flags=re.I)
    total = photo_count + frame_count
    for number in re.findall(r'<Picture\s+(\d+)>', prompt, flags=re.I):
        if not 1 <= int(number) <= total:
            raise ValueError('The prompt references Picture %s but only %d pictures are active.' % (number, total))
    # Make the native P-H3 automatic edit prefix visible in the audit, not hidden.
    if ph3 and total and not re.match(r'\s*(Edit <Picture \d+>|subject_definitions:)', prompt):
        prompt = 'Edit <Picture 1>. ' + prompt
    return prompt


def recipe(engine, standard_steps=50, eros_steps=8):
    if engine == 'MiniMax H3':
        if not 20 <= standard_steps <= 100:
            raise ValueError('Standard H3 quality steps must be 20 to 100.')
        return dict(steps=standard_steps, sampler='er_sde', scheduler='simple', shift=None, sigmas=None)
    if engine == 'Eros Beta 7 Fix1':
        if not 6 <= eros_steps <= 12:
            raise ValueError('Eros Fix1 has baked-in Turbo: use 6 to 12 steps, not 50.')
        return dict(steps=eros_steps, sampler='euler', scheduler='simple', shift=[12., 3.], sigmas=None)
    if engine == 'P-H3 Image':
        return dict(steps=8, sampler='euler', scheduler='manual', shift=[1., 1.], sigmas=PH3_SIGMAS)
    raise ValueError('Unknown model family.')


def model_names(engine, precision, mode):
    if engine not in ENGINES or precision not in PRECISIONS:
        raise ValueError('Unknown engine or precision.')
    full_diffusion = precision != 'INT8 HQ'
    full_encoder = precision == 'BF16 full'
    if engine == 'P-H3 Image':
        diffusion = 'p-h3-image_%s.safetensors' % ('bf16' if full_diffusion else 'int8_convrot')
        encoder = 'p-h3-image_text_encoder_%s.safetensors' % ('bf16' if full_encoder else 'int8_convrot')
        vae = 'p-h3-image_vae_%s.safetensors' % ('bf16' if full_encoder else 'int8_convrot')
    else:
        route = 'ref2va' if mode in ('References', 'Replace person') else 'fl2va'
        diffusion = 'minimax_h3_%s_pruned_%s.safetensors' % (route, 'bf16' if full_diffusion else 'int8_convrot')
        if engine == 'Eros Beta 7 Fix1':
            diffusion = '10Eros_Max_h3_8stepTURBO_beta7_fix1%s.safetensors' % ('' if full_diffusion else '-native_int8_protected_edges_HQ')
        encoder = 'qwen3vl_32b_minimax_h3_%s.safetensors' % ('bf16' if full_encoder else 'int8_convrot')
        vae = 'minimax_h3_video_vae_fp16.safetensors'
    return diffusion, encoder, vae
