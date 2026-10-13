"""Local-only helpers for the Hearmeman MiniMax Reference Pack workflow.

The upstream Reference Pack owns upload/edit/reference preparation and local VLM
prompt writing. These nodes only provide profile-aware render geometry, a Draft
Only gate, and exact delivery cropping around the native H3 graph.
"""
from __future__ import annotations
import os

ASPECTS = ('9:16', '2:3', '16:9')
QUALITIES = ('High fidelity', 'Standard')


def _require_ollama_idle():
    """Late-bind the runtime handoff so tests and hot-reloads can replace it safely."""
    from .local_runtime import require_idle
    return require_idle()


def _profile():
    value = (os.environ.get('H3_PORTRAIT_PROFILE') or 'full').strip().lower()
    if value != 'full':raise ValueError('Only the full HQ workspace is available.')
    return 'full'


def geometry(aspect, quality, seconds, profile=None):
    if aspect not in ASPECTS:
        raise ValueError('Choose 9:16, 2:3 or 16:9.')
    if quality not in QUALITIES:
        raise ValueError('Choose Preview, Standard or High fidelity.')
    seconds = int(seconds)
    if not 3 <= seconds <= 15:
        raise ValueError('Duration must be 3 through 15 seconds.')
    frames = max(5, round(seconds * 24))
    frames += (5 - frames % 17) % 17
    profile = profile or _profile()
    if quality == 'Preview':
        canvases = {
            '9:16': (576, 1024, 576, 1024),
            '2:3': (576, 864, 576, 864),
            '16:9': (1024, 576, 1024, 576),
        }
    else:
        canvases = {
            '9:16': (768, 1344, 756, 1344),
            '2:3': (768, 1152, 768, 1152),
            '16:9': (1344, 768, 1344, 756),
        }
    width, height, output_width, output_height = canvases[aspect]
    steps = {'Preview': 12, 'Standard': 20, 'High fidelity': 25}[quality]
    return {
        'width': width, 'height': height,
        'output_width': output_width, 'output_height': output_height,
        'frames': frames, 'actual_seconds': frames / 24, 'steps': steps,
    }


class H3ReferenceVideoSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'aspect': (list(ASPECTS), {'default': '9:16'}),
            'quality': (list(QUALITIES), {'default': 'High fidelity'}),
            'seconds': ('INT', {'default': 5, 'min': 3, 'max': 15, 'step': 1}),
        }}
    RETURN_TYPES = ('INT', 'INT', 'INT', 'FLOAT', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('width', 'height', 'frames', 'actual_seconds', 'steps', 'output_width', 'output_height')
    FUNCTION = 'settings'
    CATEGORY = 'H3 Portrait/Reference video'
    DESCRIPTION = 'Profile-aware H3 canvas, duration grid, quality steps and exact delivery crop.'

    def settings(self, aspect, quality, seconds):
        g = geometry(aspect, quality, seconds)
        return (g['width'], g['height'], g['frames'], g['actual_seconds'], g['steps'], g['output_width'], g['output_height'])


class H3ReferenceVideoDraftGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'prompt': ('STRING', {'forceInput': True}),
            'debug': ('STRING', {'forceInput': True}),
            'subject_image': ('IMAGE', {'forceInput': True}),
            'reference_video': ('IMAGE', {'forceInput': True}),
            'mode': (['Draft only', 'Generate video'], {'default': 'Draft only'}),
        }}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('approved_prompt',)
    FUNCTION = 'review'
    CATEGORY = 'H3 Portrait/Reference video'
    OUTPUT_NODE = True
    DESCRIPTION = 'Draft only runs local prompt/reference preparation but blocks H3 sampling. Review the returned prompt, then switch to Generate video.'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Recheck live residency even when the expensive Reference Pack is cached.
        return float('nan')

    def review(self, prompt, debug, subject_image, reference_video, mode):
        from comfy_execution.graph_utils import ExecutionBlocker
        preview = (prompt or '').strip()
        debug = (debug or '').strip()
        ui_text = preview + (('\n\n--- Reference Pack debug ---\n' + debug) if debug else '')
        if mode == 'Draft only':
            return {'ui': {'text': [ui_text]}, 'result': (ExecutionBlocker(None),)}
        if not preview:
            raise ValueError('The local Reference Pack returned an empty prompt. Run Draft only and inspect its debug output first.')
        if mode != 'Generate video':
            raise ValueError('Choose Draft only or Generate video.')
        if subject_image is None or reference_video is None:
            raise ValueError('Add at least one subject image and one reference video.')
        _require_ollama_idle()
        return {'ui': {'text': [ui_text]}, 'result': (preview,)}


class H3ReferenceVideoCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'images': ('IMAGE',),
            'width': ('INT', {'forceInput': True}),
            'height': ('INT', {'forceInput': True}),
        }}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'crop'
    CATEGORY = 'H3 Portrait/Reference video'
    DESCRIPTION = 'Center-crop the decoded H3 frames to the exact delivery aspect without stretching.'

    def crop(self, images, width, height):
        if images.ndim == 5 and images.shape[0] == 1:
            images = images[0]
        if images.ndim != 4:
            raise ValueError('Expected [frames,height,width,channels].')
        h, w = images.shape[1:3]
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0 or width > w or height > h:
            raise ValueError(f'Cannot crop decoded {w}x{h} frames to {width}x{height}.')
        x = (w - width) // 2
        y = (h - height) // 2
        return (images[:, y:y + height, x:x + width, :],)
