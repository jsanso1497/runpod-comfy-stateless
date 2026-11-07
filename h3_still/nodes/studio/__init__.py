"""Small optional-input/routing helpers around native ComfyUI and Fizgig nodes.

Sampling, diffusion loading, reference encoding, LoRA loading and still decoding
remain upstream implementations. No enhancer, hidden image model or API service.
"""
from pathlib import Path
import hashlib
import json
import math
import os

from .logic import (NONE, MODES, PRESETS, ASPECTS, FL2VA, REF2VA, TURBO,
                    plan_values, check_prompt, reference_size)

CATEGORY = 'H3 Still Studio'


def native_result(value, index=0):
    """Handle native V3 NodeOutput and legacy tuple results."""
    if hasattr(value, 'result'):
        value = value.result
    if isinstance(value, dict):
        value = value['result']
    return value[index]


def image_path(name):
    import folder_paths
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('Choose an image uploaded into ComfyUI, not an external file path.')
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Upload the selected reference photo again: ' + name)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('Reference exceeds 128 MB. Export a smaller still image first.')
    return path


def resize_image(image, width, height):
    import comfy.utils
    if tuple(image.shape[1:3]) == (height, width):
        return image
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height,
                                      'lanczos', 'disabled').movedim(1, -1)


class H3StudioOptionalImage:
    @classmethod
    def INPUT_TYPES(cls):
        import nodes
        names = list(nodes.LoadImage.INPUT_TYPES()['required']['image'][0])
        return {'required': {
            'image': ([NONE] + [x for x in names if x != NONE], {'image_upload': True}),
            'enabled': ('BOOLEAN', {'default': True}),
        }}
    RETURN_TYPES = ('H3_STILL_REFERENCE',)
    RETURN_NAMES = ('reference',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Upload a photo, or leave (none). Disabled/empty slots are genuinely ignored.'

    @classmethod
    def VALIDATE_INPUTS(cls, image, enabled=True):
        if not enabled or image == NONE:
            return True
        try:
            image_path(image)
            return True
        except (ValueError, OSError) as error:
            return str(error)

    @classmethod
    def IS_CHANGED(cls, image, enabled=True):
        if not enabled or image == NONE:
            return 'empty'
        h = hashlib.sha256()
        with image_path(image).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def load(self, image, enabled=True):
        if not enabled or image == NONE:
            return ({'image': None, 'name': ''},)
        from PIL import Image
        import nodes
        path = image_path(image)
        with Image.open(path) as photo:
            if photo.width * photo.height > 64 * 1024 * 1024:
                raise ValueError('Reference exceeds 64 MP. Resize it before uploading.')
            if getattr(photo, 'n_frames', 1) != 1:
                raise ValueError('Use a single still image, not an animation or multi-page image.')
        pixels = nodes.LoadImage().load_image(image)[0]
        if pixels.shape[0] != 1:
            raise ValueError('One photo per reference slot is required.')
        return ({'image': pixels[..., :3], 'name': image},)


class H3StudioPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'mode': (MODES, {'default': 'Auto'}),
            'aspect_ratio': (ASPECTS, {'default': '1:1'}),
            'generation_megapixels': ('FLOAT', {'default': 4., 'min': .25, 'max': 8., 'step': .25}),
            'edit_megapixels': ('FLOAT', {'default': 2.5, 'min': .25, 'max': 8., 'step': .25}),
            'preset': (PRESETS, {'default': PRESETS[0]}),
            'reference_cap_megapixels': ('FLOAT', {'default': 2.5, 'min': .25, 'max': 8., 'step': .25}),
        }, 'optional': {'reference_%s' % i: ('H3_STILL_REFERENCE',) for i in range(1, 10)}}
    RETURN_TYPES = ('H3_STILL_PLAN', 'INT', 'INT', 'INT')
    RETURN_NAMES = ('plan', 'width', 'height', 'steps')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Auto: 0 photos = Generate, 1 = Edit, 2-9 = References. Select References to generate from just one photo.'

    def prepare(self, mode, aspect_ratio, generation_megapixels, edit_megapixels,
                preset, reference_cap_megapixels, **kwargs):
        selected = []
        for slot in range(1, 10):
            ref = kwargs.get('reference_%s' % slot)
            if ref and ref.get('image') is not None:
                selected.append((slot, ref))
        source = selected[0][1]['image'] if selected else None
        shape = (int(source.shape[2]), int(source.shape[1])) if source is not None else None
        plan = plan_values(mode, len(selected), aspect_ratio, generation_megapixels,
                           edit_megapixels, preset, shape)
        refs, mapping = [], []
        for number, (slot, ref) in enumerate(selected, 1):
            image = ref['image']
            if plan['mode'] == 'Edit':
                w, h = plan['width'], plan['height']
            else:
                w, h = reference_size(int(image.shape[2]), int(image.shape[1]), reference_cap_megapixels)
            image = resize_image(image, w, h)
            refs.append(image)
            mapping.append({'tag': '<Picture %s>' % number, 'slot': slot,
                            'file': ref['name'], 'encoded_input_size': [w, h]})
        summary = dict(plan, reference_mapping=mapping)
        plan.update({'images': refs, 'original': source, 'summary': summary})
        print('H3 STILL STUDIO: ' + json.dumps(summary), flush=True)
        return (plan, plan['width'], plan['height'], plan['steps'])


class H3StudioModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan': ('H3_STILL_PLAN',)}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Loads only the selected quantized checkpoint. FL2VA for Generate/Edit; REF2VA for References.'

    def __init__(self):
        self._cached_name = None
        self._cached_model = None
        self._turbo_loader = None

    def load(self, plan):
        import folder_paths
        import nodes
        name = plan['model']
        if name not in (FL2VA, REF2VA):
            raise ValueError('This workspace permits only its two locked INT8 checkpoints.')
        path = folder_paths.get_full_path_or_raise('diffusion_models', name)
        stat = Path(path).stat()
        cache_key = (name, stat.st_size, stat.st_mtime_ns)
        if cache_key != self._cached_name:
            self._cached_model = None
            self._cached_model = nodes.UNETLoader().load_unet(name, 'default')[0]
            self._cached_name = cache_key
        model = self._cached_model
        if plan['turbo_strength']:
            if name != FL2VA:
                raise ValueError('The FL2VA Turbo preset must not be applied to REF2VA.')
            if self._turbo_loader is None:
                self._turbo_loader = nodes.LoraLoaderModelOnly()
            model = self._turbo_loader.load_lora_model_only(model, TURBO, plan['turbo_strength'])[0]
        return (model,)


class H3StudioLora:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        files = [name for name in folder_paths.get_filename_list('loras')
                 if name.lower().endswith('.safetensors') and Path(name).name != TURBO]
        return {'required': {'model': ('MODEL',), 'enabled': ('BOOLEAN', {'default': False}),
            'lora_name': ([NONE] + files, {'default': NONE}),
            'strength': ('FLOAT', {'default': 1., 'min': -2., 'max': 2., 'step': .05})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Optional H3-compatible safetensors LoRA. OFF or (none) means true bypass. Turbo is controlled separately by the plan preset.'

    def __init__(self):
        self._loader = None

    def apply(self, model, enabled, lora_name, strength):
        if not enabled or lora_name == NONE or strength == 0:
            return (model,)
        if not math.isfinite(strength) or not -2 <= strength <= 2:
            raise ValueError('Invalid LoRA strength.')
        if not lora_name.lower().endswith('.safetensors') or Path(lora_name).name == TURBO:
            raise ValueError('Choose an H3-compatible safetensors adapter. Use the preset for Turbo.')
        import nodes
        import folder_paths
        folder_paths.get_full_path_or_raise('loras', lora_name)
        if self._loader is None:
            self._loader = nodes.LoraLoaderModelOnly()
        updated = self._loader.load_lora_model_only(model, lora_name, strength)[0]
        # Detect a completely unmatched LoRA when the native patcher exposes counts.
        if hasattr(model, 'patches') and hasattr(updated, 'patches'):
            before = sum(len(v) for v in model.patches.values())
            after = sum(len(v) for v in updated.patches.values())
            if after <= before:
                raise ValueError('No new model patches matched this LoRA. Check that it was trained for H3 and the selected checkpoint.')
        return (updated,)


class H3StudioConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan': ('H3_STILL_PLAN',), 'clip': ('CLIP',), 'vae': ('VAE',),
            'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True})}}
    RETURN_TYPES = ('CONDITIONING',)
    FUNCTION = 'encode'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Direct prompt, no rewriting. References are numbered by active slot order. The unused native video latent is discarded.'

    def encode(self, plan, clip, vae, prompt):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
        check_prompt(prompt, plan['count'])
        common = dict(clip=clip, vae=vae, prompt=prompt, width=plan['width'], height=plan['height'], length=5)
        if plan['mode'] == 'Generate':
            result = MiniMaxH3ImageToVideo.execute(**common)
        else:
            result = MiniMaxH3ReferenceToVideo.execute(**common, ref_image_size='match',
                ref_images={'ref_image_%s' % i: im for i, im in enumerate(plan['images'])})
        return (native_result(result),)


class H3StudioAudit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan': ('H3_STILL_PLAN',)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'show'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Inspect the selected route, checkpoint, dimensions and Picture tag mapping. No weights are loaded by this report.'

    def show(self, plan):
        return {'ui': {'text': [json.dumps(plan['summary'], indent=2)]}, 'result': ()}


class H3StudioCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan': ('H3_STILL_PLAN',), 'images': ('IMAGE',),
                             'save_comparison': ('BOOLEAN', {'default': True})},
                'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'save'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Edit mode only: separate comparison PNG, original resized for display on the left. Does not alter the final PNG.'

    def save(self, plan, images, save_comparison=True, prompt=None, extra_pnginfo=None):
        if not save_comparison or plan['mode'] != 'Edit' or plan['original'] is None:
            return {'ui': {'text': ['Comparison skipped: enabled only for single-photo Edit mode.']}, 'result': ()}
        import torch
        import nodes
        before = resize_image(plan['original'], int(images.shape[2]), int(images.shape[1]))
        side_by_side = torch.cat([before.to(images), images[:1]], dim=2)
        return nodes.SaveImage().save_images(side_by_side, 'H3_Studio/Compare_Source_Left', prompt, extra_pnginfo)


NODE_CLASS_MAPPINGS = {
    'H3StudioOptionalImage': H3StudioOptionalImage, 'H3StudioPlan': H3StudioPlan,
    'H3StudioModel': H3StudioModel, 'H3StudioLora': H3StudioLora,
    'H3StudioConditioning': H3StudioConditioning, 'H3StudioAudit': H3StudioAudit,
    'H3StudioCompare': H3StudioCompare,
}
NODE_DISPLAY_NAME_MAPPINGS = {name: name.replace('H3Studio', 'H3 Still - ') for name in NODE_CLASS_MAPPINGS}

WEB_DIRECTORY = "./web"

# Additive v2 nodes preserve old Studio and the user's working standalone Turbo graph.
from .media_nodes import MEDIA_NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS.update(MEDIA_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update({name: name.replace('H3Studio', 'H3 Studio - ')
                                 for name in MEDIA_NODE_CLASS_MAPPINGS})
