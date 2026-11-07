"""New media-aware nodes. Legacy Studio nodes remain unchanged for old workflows."""
import json
from .logic import ASPECTS, reference_size
from .media_rules import MEDIA_MODES, MEDIA_PRESETS, media_plan, validate_media_prompt
from .video_input import H3StudioOptionalVideo

CATEGORY = 'H3 Still Studio'


def native(value, index=0):
    if hasattr(value, 'result'):
        value = value.result
    if isinstance(value, dict):
        value = value['result']
    return value[index]


def resized(image, width, height):
    from . import resize_image
    return resize_image(image, width, height)


def usable_mask(mask, image):
    import torch
    if mask is None or mask.numel() == 0 or not torch.isfinite(mask).all() or float(mask.max()) <= 0:
        raise ValueError('Paint a mask over the person in image 1 using its Mask Editor, or switch protect_background OFF. White means replace; black means keep.')
    if mask.ndim != 3 or mask.shape[0] != 1 or tuple(mask.shape[1:]) != tuple(image.shape[1:3]):
        raise ValueError('The target mask must match the original image dimensions. Save it again in the target image Mask Editor.')
    return mask.clamp(0, 1)


class H3StudioTargetPhoto:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'mask': ('MASK',)}}
    RETURN_TYPES = ('H3_STILL_REFERENCE',)
    RETURN_NAMES = ('target_image_1',)
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Image 1 is the existing scene. Connect the native Load Image photo and its Mask Editor output.'

    def prepare(self, image, mask):
        if image.shape[0] != 1:
            raise ValueError('Use one target photograph.')
        return ({'image': image[..., :3], 'mask': mask, 'name': 'TARGET: Image 1'},)


class H3StudioMediaPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'mode': (MEDIA_MODES, {'default': 'Auto'}),
            'aspect_ratio': (ASPECTS, {'default': '2:3'}),
            'generation_megapixels': ('FLOAT', {'default': 2.5, 'min': .25, 'max': 8., 'step': .25}),
            'edit_megapixels': ('FLOAT', {'default': 2.5, 'min': .25, 'max': 8., 'step': .25}),
            'preset': (MEDIA_PRESETS, {'default': MEDIA_PRESETS[0]}),
            'reference_cap_megapixels': ('FLOAT', {'default': 2.5, 'min': .25, 'max': 8., 'step': .25}),
            'protect_background': ('BOOLEAN', {'default': False}),
        }, 'optional': dict({'reference_%s' % i: ('H3_STILL_REFERENCE',) for i in range(1, 10)},
                           video=('H3_STILL_VIDEO',))}
    RETURN_TYPES = ('H3_STILL_PLAN', 'INT', 'INT')
    RETURN_NAMES = ('plan', 'width', 'height')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Auto selects References when a video is enabled. Replace person reserves image 1 for the target scene.'

    def prepare(self, mode, aspect_ratio, generation_megapixels, edit_megapixels,
                preset, reference_cap_megapixels, protect_background=False, video=None, **kwargs):
        selected = [(i, kwargs['reference_%s' % i]) for i in range(1, 10)
                    if kwargs.get('reference_%s' % i) and kwargs['reference_%s' % i].get('image') is not None]
        frames = (video or {}).get('frames')
        source = selected[0][1]['image'] if selected else None
        shape = (int(source.shape[2]), int(source.shape[1])) if source is not None else None
        target_active = bool(selected and selected[0][0] == 1)
        plan = media_plan(mode, len(selected), int(frames is not None), aspect_ratio,
                          generation_megapixels, edit_megapixels, preset, shape, target_active, protect_background)
        mask = selected[0][1].get('mask') if selected else None
        if protect_background:
            mask = usable_mask(mask, source)
        images, mapping = [], []
        for number, (slot, ref) in enumerate(selected, 1):
            photo = ref['image']
            if plan['mode'] in ('Edit', 'Replace person') and number == 1:
                w, h = plan['width'], plan['height']
            else:
                w, h = reference_size(int(photo.shape[2]), int(photo.shape[1]), reference_cap_megapixels)
            images.append(resized(photo, w, h))
            mapping.append(dict(tag='<Picture %s>' % number, slot=slot, file=ref['name'], size=[w, h],
                                role='target scene' if plan['mode'] == 'Replace person' and number == 1 else 'reference'))
        length = int(frames.shape[0]) if frames is not None else 5
        # Crucial: the video reference is NOT passed through the legacy length=5 path.
        plan['conditioning_length'] = length
        summary = dict(plan, reference_mapping=mapping, video=(video or {}).get('summary'),
                       final_output='one still image',
                       final_dimensions=list(shape) if protect_background else [plan['width'], plan['height']],
                       mask_usage='final composite only; not inpainting conditioning' if protect_background else 'none')
        plan.update(images=images, videos=[] if frames is None else [frames], original=source,
                    original_mask=mask, summary=summary)
        print('H3 MEDIA PLAN: ' + json.dumps(summary), flush=True)
        return plan, plan['width'], plan['height']


class H3StudioMediaModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan': ('H3_STILL_PLAN',)}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY

    def __init__(self):
        self.loader = None

    def load(self, plan):
        from . import H3StudioModel
        if self.loader is None:
            self.loader = H3StudioModel()
        return self.loader.load(dict(plan, turbo_strength=0.))


class H3StudioMediaLora:
    @classmethod
    def INPUT_TYPES(cls):
        from . import H3StudioLora
        from .media_rules import REF_TURBO
        schema = H3StudioLora.INPUT_TYPES()
        values, options = schema['required']['lora_name']
        schema['required']['lora_name'] = ([v for v in values if v.rsplit('/',1)[-1] != REF_TURBO], options)
        return schema
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Personal H3 LoRA only. Both Turbo adapters are controlled by the sampling preset, not these slots.'

    def __init__(self):
        self.loader = None

    def apply(self, model, enabled, lora_name, strength):
        from . import H3StudioLora
        from .media_rules import REF_TURBO
        if enabled and strength and lora_name.rsplit('/',1)[-1] == REF_TURBO:
            raise ValueError('Use the REF2VA Turbo sampling preset; do not apply Turbo again in a personal LoRA slot.')
        if self.loader is None:
            self.loader = H3StudioLora()
        return self.loader.apply(model, enabled, lora_name, strength)


class H3StudioMediaSampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'plan': ('H3_STILL_PLAN',)}}
    RETURN_TYPES = ('MODEL', 'SAMPLER', 'SIGMAS')
    RETURN_NAMES = ('sampling_model', 'sampler', 'sigmas')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Applies the correct Turbo only when selected; scheduler and guider use the same shifted model.'

    def __init__(self):
        self.loader = None

    def prepare(self, model, plan):
        import nodes
        import folder_paths
        import comfy.samplers
        from comfy_extras.nodes_custom_sampler import BasicScheduler
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
        if plan['adapter']:
            folder_paths.get_full_path_or_raise('loras', plan['adapter'])
            if self.loader is None:
                self.loader = nodes.LoraLoaderModelOnly()
            model = self.loader.load_lora_model_only(model, plan['adapter'], plan['adapter_strength'])[0]
        if plan['preset'] == MEDIA_PRESETS[1]:
            model = native(MiniMaxH3SigmaShift.execute(model, 12., 3.))
        sampler = comfy.samplers.sampler_object(plan['sampler'])
        sigmas = native(BasicScheduler.execute(model, plan['scheduler'], plan['steps'], 1.))
        return model, sampler, sigmas


class H3StudioMediaConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan': ('H3_STILL_PLAN',), 'clip': ('CLIP',), 'vae': ('VAE',),
                             'prompt': ('STRING', {'multiline': True, 'dynamicPrompts': True})}}
    RETURN_TYPES = ('CONDITIONING',)
    FUNCTION = 'encode'
    CATEGORY = CATEGORY

    def encode(self, plan, clip, vae, prompt):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
        validate_media_prompt(prompt, plan['count'], plan['video_count'])
        common = dict(clip=clip, vae=vae, prompt=prompt, width=plan['width'], height=plan['height'],
                      length=plan['conditioning_length'])
        if plan['mode'] == 'Generate':
            result = MiniMaxH3ImageToVideo.execute(**common)
        else:
            result = MiniMaxH3ReferenceToVideo.execute(**common, ref_image_size='match',
                ref_images={'ref_image_%s' % i: im for i, im in enumerate(plan['images'])},
                ref_videos={'ref_video_%s' % i: frames for i, frames in enumerate(plan['videos'])})
        # The large native video latent is discarded. The sampler gets Fizgig T=1 separately.
        return (native(result),)


class H3StudioMediaFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan': ('H3_STILL_PLAN',), 'images': ('IMAGE',)}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'finish'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Optional original-size masked composite for Replace person. Pixels outside the mask stay original.'

    def finish(self, plan, images):
        if not plan['protect_background']:
            return (images,)
        import torch
        original = plan['original']
        mask = usable_mask(plan['original_mask'], original).unsqueeze(-1)
        candidate = resized(images[:1], int(original.shape[2]), int(original.shape[1]))
        base = original.to(candidate)
        mask = mask.to(candidate)
        blended = candidate * mask + base * (1. - mask)
        # Exact tensor equality is guaranteed at zero-mask pixels, including soft-edge cases.
        output = torch.where(mask == 0, base, blended)
        return (output,)


class H3StudioMediaCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan': ('H3_STILL_PLAN',), 'images': ('IMAGE',),
                             'save_comparison': ('BOOLEAN', {'default': True})},
                'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'save'
    CATEGORY = CATEGORY

    def save(self, plan, images, save_comparison=True, prompt=None, extra_pnginfo=None):
        if not save_comparison or plan['original'] is None or plan['mode'] not in ('Edit', 'Replace person'):
            return {'ui': {'text': ['Comparison skipped: no target photo.']}, 'result': ()}
        import torch
        import nodes
        before = resized(plan['original'], int(images.shape[2]), int(images.shape[1])).to(images)
        return nodes.SaveImage().save_images(torch.cat([before, images[:1]], dim=2),
            'H3_Studio_v2/Compare_Target_Left', prompt, extra_pnginfo)


MEDIA_NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in
    (H3StudioOptionalVideo, H3StudioTargetPhoto, H3StudioMediaPlan, H3StudioMediaModel,
     H3StudioMediaLora, H3StudioMediaSampling, H3StudioMediaConditioning, H3StudioMediaFinish, H3StudioMediaCompare)}
