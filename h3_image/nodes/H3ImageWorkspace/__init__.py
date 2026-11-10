"""Manual-prompt controls around native MiniMax H3 Image. No generative implementation is replaced."""
import hashlib
import json
import math
from pathlib import Path
import re
import time

CATEGORY = 'H3 Image / Manual workspace'
NONE = '(none)'
SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
ASPECTS = ['1:1', '2:3', '3:2', '3:4', '4:3', '9:16', '16:9', '21:9']


def result(value):
    if hasattr(value, 'result'):
        return value.result
    if isinstance(value, dict):
        return value['result']
    return value


def dimensions(aspect_ratio, megapixels, follow_source=False, source_size=None):
    if aspect_ratio not in ASPECTS or not 4.0 <= float(megapixels) <= 20.0:
        raise ValueError('Choose 4-20 MP and a supported aspect ratio.')
    a, b = map(int, aspect_ratio.split(':'))
    ratio = a / b
    if follow_source and source_size is not None:
        width, height = source_size
        if min(width, height) < 1:
            raise ValueError('Invalid source dimensions.')
        ratio = width / height
    if not 1 / 4 <= ratio <= 4:
        raise ValueError('Crop an extreme-aspect reference before following its aspect ratio.')
    h = math.sqrt(float(megapixels) * 1024 ** 2 / ratio)
    w = h * ratio
    return int(round(w / 32) * 32), int(round(h / 32) * 32)


def select_references(references, minimum):
    selected = []
    for slot in range(1, 17):
        ref = references.get('reference_%s' % slot)
        if ref and ref.get('image') is not None:
            selected.append((slot, ref))
    if len(selected) < minimum:
        raise ValueError('This preset needs at least %s active reference(s). Upload them or select the general workflow.' % minimum)
    return selected


def validate_prompt(prompt, count):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Write your full prompt. This workspace does not add or expand any wording.')
    for number in re.findall(r'<Picture\s+(\d+)>', prompt, re.I):
        if not 1 <= int(number) <= count:
            raise ValueError('Prompt refers to Picture %s but only %s references are active.' % (number, count))


class H3ImageOptionalReference:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        import nodes
        base = nodes.LoadImage.INPUT_TYPES()['required']['image'][0]
        return {'required': {'image': ([NONE] + list(base), {'image_upload': True}),
                             'enabled': ('BOOLEAN', {'default': True})}}
    RETURN_TYPES = ('H3_IMAGE_REFERENCE',)
    RETURN_NAMES = ('reference',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(cls, image, enabled=True):
        if not enabled or image == NONE:
            return True
        import nodes
        return nodes.LoadImage.VALIDATE_INPUTS(image)

    @classmethod
    def IS_CHANGED(cls, image, enabled=True):
        if not enabled or image == NONE:
            return 'inactive'
        import nodes
        return nodes.LoadImage.IS_CHANGED(image)

    def load(self, image, enabled=True):
        if not enabled or image == NONE:
            return ({'image': None, 'file': NONE},)
        import nodes
        pixels, _ = nodes.LoadImage().load_image(image)
        if pixels.shape[0] != 1:
            raise ValueError('Use one still image per slot, not an animated image.')
        return ({'image': pixels[..., :3], 'file': image},)


class H3ImageManualConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'clip': ('CLIP',), 'vae': ('VAE',),
            'prompt': ('STRING', {'default': '', 'multiline': True, 'dynamicPrompts': False}),
            'aspect_ratio': (ASPECTS, {'default': '1:1'}),
            'output_megapixels': ('FLOAT', {'default': 4., 'min': 4., 'max': 20., 'step': 1.}),
            'follow_image_1_aspect': ('BOOLEAN', {'default': False}),
            'minimum_references': ('INT', {'default': 0, 'min': 0, 'max': 16})},
            'optional': dict({'reference_%s' % i: ('H3_IMAGE_REFERENCE',) for i in range(1, 17)},
                aspect_source_slot=('INT', {'default': 0, 'min': 0, 'max': 16, 'tooltip': '0 follows the first active picture. Use 3 for the body/face/scene-last layout. Only affects sizing, never image order or prompt.'}))}
    RETURN_TYPES = ('CONDITIONING', 'LATENT', 'STRING')
    RETURN_NAMES = ('conditioning', 'image_latent', 'reference_report')
    FUNCTION = 'encode'
    CATEGORY = CATEGORY

    def encode(self, clip, vae, prompt, aspect_ratio, output_megapixels,
               follow_image_1_aspect, minimum_references, aspect_source_slot=0, **references):
        from comfy_extras import nodes_minimax_h3 as native
        MiniMaxH3ImageConditioning = native.MiniMaxH3ImageConditioning
        started = time.perf_counter()
        selected = select_references(references, minimum_references)
        validate_prompt(prompt, len(selected))
        source_ref = selected[0][1] if selected else None
        if follow_image_1_aspect and aspect_source_slot:
            source_ref = references.get('reference_%s' % aspect_source_slot)
            if not source_ref or source_ref.get('image') is None:
                raise ValueError('The selected aspect source slot is empty or disabled.')
        source = source_ref['image'] if source_ref is not None else None
        source_size = (int(source.shape[2]), int(source.shape[1])) if source is not None else None
        w, h = dimensions(aspect_ratio, output_megapixels, follow_image_1_aspect, source_size)
        images = {'image_%s' % i: ref['image'] for i, (_, ref) in enumerate(selected, 1)}
        encoded = result(MiniMaxH3ImageConditioning.execute(clip=clip, vae=vae, prompt=prompt,
                                                          width=w, height=h, images=images))
        report = {'model_family': 'MiniMax-H3-Image (not P-H3, not video H3)',
                  'output_size': [w, h], 'reference_count': len(selected),
                  'prompt_unchanged': True, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'references': [{'slot': slot, 'tag': '<Picture %s>' % i, 'file': ref['file'],
                                 'source_size': [int(ref['image'].shape[2]), int(ref['image'].shape[1])]}
                                 for i, (slot, ref) in enumerate(selected, 1)],
                  'aspect_source_slot': aspect_source_slot if follow_image_1_aspect else None,
                  'reference_resizing': 'Native H3 Image short-edge rules, independent of output size. Small inputs may be upscaled.',
                  'conditioning_seconds': time.perf_counter() - started,
                  'timing_note': 'Conditioning wall time includes loads and encoding. A cached node report is not the duration of a subsequent queue.',
                  'preservation': 'Whole-image regeneration, not pixel-protected inpainting.'}
        short = native.IMAGE_REF_SHORT_EDGE
        upper = native.IMAGE_REF_VAE_MAX_SHORT_EDGE
        for row in report['references']:
            iw, ih = row['source_size']
            row['vision_size'] = list(native._size_short_edge(iw, ih, short, short))
            row['vae_size'] = list(native._size_short_edge(iw, ih, short, upper))
            row['vae_image_tokens_estimate'] = (row['vae_size'][0] // 32) * (row['vae_size'][1] // 32)
        report['target_image_tokens_estimate'] = (w // 32) * (h // 32)
        report['reference_image_tokens_estimate'] = sum(row['vae_image_tokens_estimate'] for row in report['references'])
        print('H3 IMAGE INPUTS: ' + json.dumps(report), flush=True)
        return encoded[0], encoded[1], json.dumps(report, indent=2)


class H3ImageSixStepSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',)}}
    RETURN_TYPES = ('SAMPLER', 'SIGMAS')
    RETURN_NAMES = ('euler_sampler', 'six_step_sigmas')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY

    def prepare(self, model):
        import torch
        import comfy.samplers
        sampling = model.get_model_object('model_sampling')
        grid = getattr(sampling, 'grid', None)
        expected = torch.tensor(SIGMAS, dtype=torch.float32)
        if grid is None or tuple(grid.shape) != tuple(expected.shape) or not torch.allclose(grid.detach().cpu().flip(0), expected, atol=1e-6, rtol=0):
            raise ValueError('This checkpoint does not expose the verified six-step H3 Image PDD grid. '
                             'Do not use video H3, Eros, P-H3 or Turbo adapters in this workspace.')
        return comfy.samplers.sampler_object('euler'), expected


def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class H3ImageTimedSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'noise': ('NOISE',), 'guider': ('GUIDER',), 'sampler': ('SAMPLER',),
            'sigmas': ('SIGMAS',), 'latent_image': ('LATENT',), 'reference_report': ('STRING', {'forceInput': True}),
            'run_label': ('STRING', {'default': 'native_test'})}}
    RETURN_TYPES = ('LATENT', 'STRING')
    RETURN_NAMES = ('samples', 'sampling_report')
    FUNCTION = 'sample'
    CATEGORY = CATEGORY

    def sample(self, noise, guider, sampler, sigmas, latent_image, reference_report, run_label):
        import torch
        from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
        cuda_sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        samples = result(SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent_image))[0]
        cuda_sync()
        report = json.loads(reference_report)
        report.update(sampling_seconds=time.perf_counter()-started, steps=len(sigmas)-1,
                      seed=getattr(noise, 'seed', None), run_label=run_label,
                      report_version='2026-11-10.2')
        if torch.cuda.is_available():
            report.update(gpu=torch.cuda.get_device_name(),
                          sampling_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          sampling_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        report['memory_note'] = 'PyTorch process peaks, not all GPU or container memory.'
        return samples, json.dumps(report, indent=2)


class H3ImageTimedDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples': ('LATENT',), 'vae': ('VAE',), 'sampling_report': ('STRING', {'forceInput': True})}}
    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('image', 'complete_report')
    FUNCTION = 'decode'
    CATEGORY = CATEGORY

    def decode(self, samples, vae, sampling_report):
        import nodes
        cuda_sync()
        start = time.perf_counter()
        image = result(nodes.VAEDecode().decode(samples=samples, vae=vae))[0]
        cuda_sync()
        report = json.loads(sampling_report)
        report.update(decode_seconds=time.perf_counter()-start, decoded_size=[image.shape[2], image.shape[1]])
        return image, json.dumps(report, indent=2)


class H3ImageCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'reference': ('H3_IMAGE_REFERENCE',), 'generated': ('IMAGE',)}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('source_left_result_right',)
    FUNCTION = 'compare'
    CATEGORY = CATEGORY

    def compare(self, reference, generated):
        import torch
        import comfy.utils
        source = reference.get('image')
        if source is None:
            raise ValueError('The source/result comparison needs its source photo enabled.')
        # Contain, never stretch. This comparison is distinct from the full-size final PNG.
        gh, gw = generated.shape[1:3]
        scale = min(1., 1280. / max(gh, gw))
        w, h = max(1, round(gw*scale)), max(1, round(gh*scale))
        out = comfy.utils.common_upscale(generated[:1,...,:3].movedim(-1,1), w, h, 'bilinear', 'disabled').movedim(1,-1)
        sh, sw = source.shape[1:3]
        fit = min(w/sw,h/sh)
        nw, nh = max(1, round(sw*fit)), max(1, round(sh*fit))
        before = comfy.utils.common_upscale(source[:1,...,:3].movedim(-1,1), nw, nh, 'bilinear', 'disabled').movedim(1,-1).to(out)
        canvas = torch.full_like(out, .5)
        x, y = (w-nw)//2, (h-nh)//2
        canvas[:,y:y+nh,x:x+nw,:] = before
        return (torch.cat([canvas,out], dim=2),)


class H3ImageReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'report': ('STRING', {'forceInput': True})},
                'optional': {'save_report': ('BOOLEAN', {'default': True})}}
    RETURN_TYPES = ()
    FUNCTION = 'show'
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def show(self, report, save_report=True):
        import uuid
        import folder_paths
        data = json.loads(report)
        display = report
        if save_report:
            root = Path(folder_paths.get_output_directory()) / 'H3_Image' / 'reports'
            root.mkdir(parents=True, exist_ok=True)
            filename = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.json'
            path = root / filename
            with path.open('x', encoding='utf-8') as stream:
                json.dump(data,stream,indent=2)
            display += '\nSaved: ' + str(path)
        print('H3 IMAGE REPORT: ' + report, flush=True)
        return {'ui': {'text': [display]}, 'result': ()}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3ImageOptionalReference, H3ImageManualConditioning,
    H3ImageSixStepSchedule, H3ImageReport, H3ImageTimedSampler, H3ImageTimedDecode, H3ImageCompare)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'H3ImageOptionalReference': 'H3 Image - Optional Reference',
    'H3ImageManualConditioning': 'H3 Image - Your Prompt + References',
    'H3ImageSixStepSchedule': 'H3 Image - Verified Six-Step Schedule',
    'H3ImageTimedSampler': 'H3 Image - Native Sampling + Timing',
    'H3ImageTimedDecode': 'H3 Image - Native Decode + Timing',
    'H3ImageCompare': 'H3 Image - Source / Result Compare',
    'H3ImageReport': 'H3 Image - Execution Report'}
