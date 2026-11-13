"""Pure routing and geometry rules for H3 Video Still Studio. No GPU imports."""
from __future__ import annotations
import math
import re

VERSION = '2026-11-13.a40.1'
ENGINES = ['MiniMax H3', 'Eros Beta 7 Fix1']
PRECISIONS = ['INT8 HQ']
MODES = ['Auto', 'References', 'Replace person']
ASPECTS = ['1:1', '2:3', '3:2', '4:5', '5:4', '9:16', '16:9', '3:4', '4:3']
NONE = '(none)'


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
    return strategy == 'Keyframes for all engines'


def resolve_mode(mode, photo_slots, has_video):
    if mode not in MODES:
        raise ValueError('This focused profile uses References or Replace person.')
    if not photo_slots and not has_video:
        raise ValueError('Add a video or at least one photo. This A40 profile is reference-to-still, not text-only.')
    selected = 'References' if mode == 'Auto' else mode
    if selected == 'Replace person' and (1 not in photo_slots or (len(photo_slots) < 2 and not has_video)):
        raise ValueError('Replacement needs the target in photo slot 1 plus an identity photo or video.')
    return selected


def reference_budget(engine, photo_count, frame_count):
    maximum = 9
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
    raise ValueError('Unknown model family.')


def model_names(engine, precision, mode):
    if engine not in ENGINES or precision != 'INT8 HQ':
        raise ValueError('The A40 profile supports MiniMax H3 / Eros with INT8 HQ only.')
    diffusion = ('minimax_h3_ref2va_pruned_int8_convrot.safetensors' if engine == 'MiniMax H3'
                 else '10Eros_Max_h3_8stepTURBO_beta7_fix1-native_int8_protected_edges_HQ.safetensors')
    return (diffusion, 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
            'minimax_h3_video_vae_fp16.safetensors')
