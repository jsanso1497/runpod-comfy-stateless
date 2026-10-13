"""Green suit geometry reference and ordered GPT edit for Comfy Workbench HQ.

No model weights, credentials, downloads, or remote calls at import time.
Local inference delegates to the already-installed Qwen, SAM and SeedVR2 nodes.
The only paid operation is an explicitly enabled call to the OpenAI Images API.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter, binary_erosion
import torch

VERSION = '1.0.0'
CATEGORY = 'Workbench/Green Suit Overlay'
SAM_FILE = 'sam3.1_multiplex_fp16.safetensors'
REF_SIZE = (2720, 4080)
API_SIZE = (2336, 3504)
MODES = ['1 - Inspect SAM mask only', '2 - Build Qwen + 4K reference', '3 - Run GPT + compare all']

QWEN_PROMPT = '''Edit <image1> to create a precise green-suit geometry reference of the same person.
Replace all visible clothing and visible skin of the selected person, including the face, neck, hands and feet wherever present in this crop, with one continuous opaque chroma-green lycra bodysuit. The face is covered by fitted green fabric following its existing surface contours. Leave all hair completely uncovered and unchanged, including the hairline and loose strands.
Use smooth fitted matte stretch fabric with subtle realistic folds and shading, no logos, no patterns, no transparency and no loose garment volume. Follow the person's existing anatomy and visible contours; do not idealize, enlarge, slim, lengthen or invent hidden anatomy.
Keep exactly the same body pose, head angle, body rotation, perspective, body proportions, subject position, camera distance, framing and canvas boundaries. Do not crop, zoom, recenter, mirror, outpaint or reveal any part outside the existing image. Preserve occlusions and the background. Replace vacated clothing volume with the consistent surrounding background, not additional body parts. Change only the selected person; hair and everything outside the editable mask are protected by compositing.'''

GPT_PROMPT = '''Create one realistic edited photograph from the two attached images. Their roles are deliberately different.

IMAGE 1 is the exclusive appearance reference: use the same person's facial identity, individual facial features, skin tone, age, expression, gaze, hair, hairstyle, hairline, original clothing, clothing colors, fabric textures, accessories, background and lighting from IMAGE 1. Do not invent a new person or blend identities.

IMAGE 2 is only the geometric template. Its green fabric is a temporary geometry aid, not wardrobe to reproduce. Transfer the visible body shape, silhouette, body proportions, relative sizes, pose, head and body turn angles, perspective, limb placement, occlusions, subject scale, framing, crop and exact location from IMAGE 2. Reconstruct the appearance from IMAGE 1 on this geometry.

Register the output to the full canvas of IMAGE 2. Keep all corresponding body contour points and anatomical landmarks at the same normalized pixel coordinates. Match the head top, hair outline, shoulders, elbows, wrists, hips, knees and feet only where they are actually visible. The silhouette should overlay IMAGE 2 without translation, scaling, rotation or perspective adjustment. Preserve the exact intersections with every canvas edge. Show no more and no less of the person than IMAGE 2 shows. Do not zoom out, zoom in, center the subject, mirror, crop, expand the scene, or complete off-frame body parts.

Restore the original wardrobe from IMAGE 1, fitted to the geometric envelope in IMAGE 2. Do not retain the green suit, its artificial facial covering, green reflections, or green color spill. Do not take identity, clothing design, colors or background from IMAGE 2. When loose original clothing conflicts with the target silhouette, keep its design and texture but fit its visible outline to IMAGE 2; do not copy the original loose silhouette.

Keep the original background and original hair from IMAGE 1 at their existing coordinates. Preserve the facial expression and gaze from IMAGE 1 while matching the visible head angle in IMAGE 2. Output a single finished 2:3 portrait photograph, not a comparison, collage, diagram, overlay or mask. Treat exact geometric registration as a requirement, not an approximate pose suggestion.'''


def _output(value):
    if hasattr(value, 'result'):
        return value.result
    if isinstance(value, dict):
        return value['result']
    return value


def _node(name, **values):
    """Invoke the actual registered node, supporting classic and v3 entrypoints."""
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f'Missing installed node {name}. Rebuild the Qwen HQ workspace and prepare this task.')
    # v3 classes inherit compatibility shims; use their actual execute method.
    if hasattr(cls, 'define_schema') and hasattr(cls, 'execute'):
        method = cls.execute
    else:
        instance = cls()
        method = getattr(instance, getattr(cls, 'FUNCTION', 'execute'))
    signature = inspect.signature(method)
    accepts_extra = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    unknown = set(values) - set(signature.parameters)
    if unknown and not accepts_extra:
        raise RuntimeError(f'{name} has an incompatible interface: {sorted(unknown)}. No setting was silently dropped.')
    return _output(method(**values))


def _release():
    import comfy.model_management as mm
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()


def _image(image):
    if not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
        raise ValueError('Use exactly one RGB image, not a batch or video.')
    image = image[..., :3].detach().float().cpu()
    if not torch.isfinite(image).all():
        raise ValueError('Image contains non-finite values.')
    return image.clamp(0, 1)


def _portrait(image):
    image = _image(image)
    h, w = image.shape[1:3]
    if min(w, h) < 128 or abs((w / h) / (2 / 3) - 1) > 0.002:
        raise ValueError(f'Expected a 2:3 portrait, received {w}x{h}. No automatic crop or outpainting is performed.')
    return image


def _u8(image):
    return np.rint(_image(image)[0].numpy() * 255).clip(0, 255).astype(np.uint8)


def _tensor(array):
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy() / 255.0).unsqueeze(0)


def _resize(image, size):
    image = _image(image)
    if (image.shape[2], image.shape[1]) == tuple(size):
        return image.clone()
    pil = Image.fromarray(_u8(image))
    return _tensor(pil.resize(tuple(size), Image.Resampling.LANCZOS))


def _mask(mask, h, w):
    if not torch.is_tensor(mask) or mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError('Expected one mask in source coordinates.')
    if tuple(mask.shape[1:]) != (h, w):
        raise ValueError('Mask and image dimensions differ. Refusing an unregistered mask.')
    if not torch.isfinite(mask).all():
        raise ValueError('Mask contains non-finite values.')
    return mask.detach().float().cpu().clamp(0, 1)


def _resize_mask(mask, size):
    return torch.nn.functional.interpolate(mask[:, None].float(), (size[1], size[0]), mode='nearest')[:, 0]


def _grow(mask, pixels):
    if pixels <= 0:
        return (mask > .5).float()
    a = maximum_filter((mask[0].numpy() > .5), size=2 * int(pixels) + 1, mode='constant')
    return torch.from_numpy(a.copy()).float().unsqueeze(0)


def _inward(mask, feather):
    binary = mask[0].numpy() > .5
    if feather <= 0:
        return torch.from_numpy(binary.astype(np.float32)).unsqueeze(0)
    # A canvas crop is not a silhouette edge. Continue border occupancy instead
    # of fading away body parts that legitimately intersect the image boundary.
    if binary.all():
        return torch.ones_like(mask, dtype=torch.float32)
    pad = max(1, int(np.ceil(feather)))
    distance = distance_transform_edt(np.pad(binary, pad, mode='edge'))[pad:-pad, pad:-pad]
    return torch.from_numpy(np.clip(distance / float(feather), 0, 1).astype(np.float32)).unsqueeze(0)


def _composite(base, candidate, alpha):
    base, candidate = _image(base), _image(candidate)
    if base.shape != candidate.shape:
        raise ValueError('Composite inputs differ in dimensions.')
    alpha = _mask(alpha, base.shape[1], base.shape[2])[..., None]
    blended = candidate * alpha + base * (1 - alpha)
    # Exactly preserve floating-point source values wherever alpha is zero.
    return torch.where(alpha > 0, blended, base)


def _png(image):
    stream = io.BytesIO()
    Image.fromarray(_u8(image)).save(stream, format='PNG', compress_level=4)
    return stream.getvalue()


def _stage_result(results, image, label):
    # A UI preview is not an OUTPUT_NODE and does not bypass the lazy run gate.
    # It leaves the completed local stage inspectable if a later stage fails.
    import folder_paths
    temp = Path(folder_paths.get_temp_directory())
    temp.mkdir(parents=True, exist_ok=True)
    name = f'wbgs_{label}_{uuid.uuid4().hex}.png'
    (temp / name).write_bytes(_png(image))
    return {'result': results, 'ui': {'images': [{'filename': name, 'subfolder': '', 'type': 'temp'}]}}


def _api_mask(mask):
    """Comfy white=edit -> OpenAI alpha=0 means edit. Not an RGB-only mask."""
    mask = mask.detach().float().cpu().clamp(0, 1)
    h, w = mask.shape[1:]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = np.rint((1 - mask[0].numpy()) * 255).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgba).save(stream, format='PNG')
    return stream.getvalue()


def _choose(masks, reference=None, index=-1):
    if not torch.is_tensor(masks) or masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError('SAM found no matching instance. Refine the object prompt or use the saved manual protection mask.')
    masks = (masks.detach().float().cpu() > .5).float()
    if index >= 0:
        if index >= len(masks):
            raise ValueError(f'SAM found {len(masks)} instance(s); index {index} is unavailable.')
        chosen = masks[index:index + 1]
    elif reference is not None:
        reference = reference[0] > .5
        scores = [float(((m > .5) & reference).sum()) / max(1, int(((m > .5) | reference).sum())) for m in masks]
        chosen = masks[int(np.argmax(scores)):int(np.argmax(scores)) + 1]
    else:
        sizes = masks.sum(dim=(1, 2))
        i = int(sizes.argmax())
        chosen = masks[i:i + 1]
    if not chosen.any():
        raise ValueError('SAM returned an empty selected mask. There is no full-image fallback.')
    return chosen


def _sam(image, prompts, threshold=.35, refine_iterations=2):
    import folder_paths
    import nodes
    from comfy_extras.nodes_sam3 import SAM3_Detect
    if folder_paths.get_full_path('checkpoints', SAM_FILE) is None:
        raise FileNotFoundError(f'Prepare SAM in Workbench: models/checkpoints/{SAM_FILE} is missing.')
    model, clip, _ = nodes.CheckpointLoaderSimple().load_checkpoint(SAM_FILE)
    results = []
    try:
        for prompt in prompts:
            if not prompt.strip():
                results.append(None)
                continue
            cond = nodes.CLIPTextEncode().encode(clip, prompt.strip())[0]
            result = _output(SAM3_Detect.execute(model=model, image=image, conditioning=cond,
                            threshold=threshold, refine_iterations=refine_iterations, individual_masks=True))[0]
            if not torch.is_tensor(result) or result.ndim != 3 or tuple(result.shape[1:]) != tuple(image.shape[1:3]):
                raise ValueError('SAM returned a mask outside the input coordinate system.')
            results.append(result.detach().float().cpu())
        return results
    finally:
        del model, clip
        _release()


def _preview(source, mask):
    alpha = .35 * mask[..., None]
    tint = torch.tensor([0., 1., 0.]).reshape(1, 1, 1, 3)
    return source * (1 - alpha) + tint * alpha


class WBGSBodyMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source': ('IMAGE',),
            'person_prompt': ('STRING', {'default': 'person'}),
            'hair_prompt': ('STRING', {'default': 'hair'}),
            'threshold': ('FLOAT', {'default': .35, 'min': .05, 'max': .95, 'step': .05}),
            'person_index': ('INT', {'default': -1, 'min': -1, 'max': 63, 'tooltip': '-1 selects the largest person, not all people.'}),
            'hair_required': ('BOOLEAN', {'default': True}),
            'hair_guard_pixels': ('INT', {'default': 3, 'min': 0, 'max': 32}),
            'edit_expand_pixels': ('INT', {'default': 8, 'min': 0, 'max': 128}),
            'edit_feather_pixels': ('INT', {'default': 6, 'min': 0, 'max': 128}),
        }, 'optional': {'manual_protection': ('MASK',)}}
    RETURN_TYPES = ('WB_GS_MASK', 'MASK', 'IMAGE')
    RETURN_NAMES = ('source_and_masks', 'edit_mask', 'mask_preview')
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Select one person, subtract hair AFTER growth. Entire visible body includes the face. Painted mask adds protection.'

    @torch.inference_mode()
    def run(self, source, person_prompt='person', hair_prompt='hair', threshold=.35,
            person_index=-1, hair_required=True, hair_guard_pixels=3, edit_expand_pixels=8, edit_feather_pixels=6,
            manual_protection=None):
        source = _portrait(source)
        h, w = source.shape[1:3]
        if not person_prompt.strip():
            raise ValueError('Person prompt cannot be empty.')
        detections = _sam(source, [person_prompt, hair_prompt], threshold)
        person = _choose(detections[0], index=person_index)
        hair = torch.zeros_like(person)
        if detections[1] is not None and len(detections[1]):
            hair = detections[1].amax(dim=0, keepdim=True).gt(.5).float() * _grow(person, 16)
        manual_has_pixels = manual_protection is not None and bool(torch.any(manual_protection > 0))
        if hair_required and not hair.any() and not manual_has_pixels:
            raise ValueError('No hair detected for this person. Refine hair_prompt or paint and save a hair protection mask. Turn hair_required OFF only when hair is absent/out of frame.')
        protect = _grow(hair, hair_guard_pixels)
        if manual_protection is not None and torch.any(manual_protection > 0):
            protect = torch.maximum(protect, _mask(manual_protection, h, w).gt(.5).float())
        edit = _inward(_grow(person, edit_expand_pixels), edit_feather_pixels) * (1 - protect)
        if not edit.any():
            raise ValueError('The protected hair/manual mask leaves no body to edit.')
        report = {'source_size': [w, h], 'person_prompt': person_prompt, 'hair_prompt': hair_prompt,
                  'person_index': person_index, 'face_included': True, 'mask_pixels': int(edit.sum()),
                  'hair_guard_pixels': hair_guard_pixels, 'edit_expand_pixels': edit_expand_pixels,
                  'manual_hair_fallback': bool(not hair.any() and manual_has_pixels)}
        packet = {'source': source, 'person': person, 'protect': protect, 'edit': edit, 'mask_report': report,
                  'person_prompt': person_prompt, 'threshold': threshold}
        return packet, edit, _preview(source, edit)


class WBGSQwenSuit:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source_and_masks': ('WB_GS_MASK',),
            'prompt': ('STRING', {'default': QWEN_PROMPT, 'multiline': True}),
            'seed': ('INT', {'default': 12345, 'min': 0, 'max': 2**63-1, 'control_after_generate': False}),
            'steps': ('INT', {'default': 40, 'min': 10, 'max': 100}),
            'context_pixels': ('INT', {'default': 128, 'min': 0, 'max': 512}),
            'feather_pixels': ('INT', {'default': 2, 'min': 0, 'max': 32}),
        }, 'optional': {'prompt_enhancer': ('WB_QWEN_PROMPT',)}}
    RETURN_TYPES = ('WB_GS_QWEN', 'IMAGE')
    RETURN_NAMES = ('qwen_result', 'green_suit_original_size')
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Existing Qwen Image 2.1 BF16 path: 40 steps, CFG 1, Euler/simple. Exact source-size protected stitch.'

    @torch.inference_mode()
    def run(self, source_and_masks, prompt=QWEN_PROMPT, seed=12345, steps=40,
            context_pixels=128, feather_pixels=2, prompt_enhancer=None):
        p = source_and_masks
        crop, _ = _node('Q21PhotoCrop', source=p['source'], mask=p['edit'],
                        mask_report=json.dumps(p['mask_report']), resolution=2048, max_long_side=3072,
                        context_pixels=context_pixels, upscale_small_crops=True)
        model = clip = vae = None
        try:
            model, clip, vae = _node('Q21PhotoModels', crop_data=crop, precision='BF16 quality',
                          text_encoder_device='default', cache_device='auto', checkpoint='qwen_image_2.1_bf16.safetensors')
            positive, negative, latent, report = _node('Q21PhotoEncode', clip=clip, vae=vae,
                          crop_data=crop, prompt=prompt, prompt_enhancer=prompt_enhancer)
            sampled = _node('KSampler', model=model, seed=seed, steps=steps, cfg=1.0,
                          sampler_name='euler', scheduler='simple', positive=positive,
                          negative=negative, latent_image=latent, denoise=1.0)[0]
            decoded = _node('VAEDecode', samples=sampled, vae=vae)[0]
            stitched = _node('Q21PhotoStitch', crop_data=crop, generated_crop=decoded,
                          reference_report=report, feather_pixels=feather_pixels, boundary_color_strength=0.0)[0]
            green = _image(stitched['image'])
            protected = p['edit'] == 0
            if green.shape != p['source'].shape or not torch.equal(green[protected], p['source'][protected]):
                raise ValueError('Qwen source protection failed. No reference was passed downstream.')
            packet = {**p, 'green': green, 'qwen_report': stitched['report']}
            return _stage_result((packet, green), green, 'qwen')
        finally:
            del model, clip, vae
            _release()


def _seedvr2(image, seed):
    # Exact same HQ model pair as the repository's standard restoration graph.
    import folder_paths
    for filename in ('seedvr2_ema_7b_fp16.safetensors', 'ema_vae_fp16.safetensors'):
        if folder_paths.get_full_path('SEEDVR2', filename) is None:
            raise FileNotFoundError(f'Prepare SeedVR2 in Workbench: {filename} is missing.')
    _release()
    dit = _node('SeedVR2LoadDiTModel', model='seedvr2_ema_7b_fp16.safetensors', device='cuda:0',
                blocks_to_swap=0, swap_io_components=False, offload_device='cpu', cache_model=False,
                attention_mode='sdpa')[0]
    vae = _node('SeedVR2LoadVAEModel', model='ema_vae_fp16.safetensors', device='cuda:0',
                offload_device='cpu', cache_model=False, encode_tiled=False, decode_tiled=False,
                encode_tile_size=1024, encode_tile_overlap=128, decode_tile_size=1024,
                decode_tile_overlap=128, tile_debug='false')[0]
    try:
        return _image(_node('SeedVR2VideoUpscaler', image=image, dit=dit, vae=vae,
                seed=int(seed) % 2**32, resolution=REF_SIZE[0], max_resolution=REF_SIZE[1],
                batch_size=1, uniform_batch_size=False, temporal_overlap=0, prepend_frames=0,
                color_correction='lab', input_noise_scale=0.0, latent_noise_scale=0.0,
                offload_device='cpu', enable_debug=False)[0])
    finally:
        del dit, vae
        _release()


class WBGS4KReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'qwen_result': ('WB_GS_QWEN',),
            'upscale_method': (['SeedVR2 7B FP16', 'Lanczos - no synthesized detail'],),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 2**32-1, 'control_after_generate': False}),
        }}
    RETURN_TYPES = ('WB_GS_REFERENCE', 'IMAGE')
    RETURN_NAMES = ('reference_package', 'green_reference_4k')
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    DESCRIPTION = '2720x4080 exact 2:3. Re-protect original hair/background after upscaling. SAM measures this actual reference silhouette.'

    @torch.inference_mode()
    def run(self, qwen_result, upscale_method='SeedVR2 7B FP16', seed=42):
        p = qwen_result
        if upscale_method not in ('SeedVR2 7B FP16', 'Lanczos - no synthesized detail'):
            raise ValueError('Unknown upscaling method.')
        upscaled = _seedvr2(p['green'], seed) if upscale_method == 'SeedVR2 7B FP16' else p['green']
        upscaled = _resize(upscaled, REF_SIZE)
        original4k = _resize(p['source'], REF_SIZE)
        protect4k = _resize_mask(p['protect'], REF_SIZE)
        edit4k = _resize_mask(p['edit'], REF_SIZE) * (1 - protect4k)
        reference = _composite(original4k, upscaled, _inward(edit4k, 2))
        detections = _sam(reference, [p['person_prompt']], p['threshold'])[0]
        target = _choose(detections, reference=_resize_mask(p['person'], REF_SIZE))
        body = target * (1 - protect4k) * edit4k
        if not body.any():
            raise ValueError('No editable body detected in the finished green reference.')
        rgb = reference[0]
        green_pixels = (rgb[..., 1] > rgb[..., 0] * 1.15) & (rgb[..., 1] > rgb[..., 2] * 1.15) & (rgb[..., 1] > .2)
        coverage = float((green_pixels & body[0].bool()).sum()) / max(1, int(body.sum()))
        if coverage < .10:
            raise ValueError('The Qwen reference has less than 10% green coverage in the selected body. Inspect the Qwen result/prompt before any paid GPT call.')
        packet = {**p, 'reference4k': reference, 'original4k': original4k, 'edit4k': edit4k,
                  'protect4k': protect4k, 'target_person4k': target, 'target_body4k': body,
                  'upscale_method': upscale_method, 'green_coverage': coverage}
        return _stage_result((packet, reference), reference, 'reference')


def _valid_api_size(size):
    w, h = size
    return (w % 16 == h % 16 == 0 and max(w, h) <= 3840 and
            655360 <= w * h <= 8294400 and max(w, h) <= 3 * min(w, h))


def _cache_digest(params, image1, image2, mask, variant):
    h = hashlib.sha256()
    for value in (json.dumps(params, sort_keys=True).encode(), image1, image2, mask or b'', str(variant).encode()):
        h.update(len(value).to_bytes(8, 'big'))
        h.update(value)
    return h.hexdigest()


def _decode_api_image(data, expected):
    if not data or len(data) > 64 * 1024 * 1024:
        raise ValueError('Invalid or oversized image response.')
    with Image.open(io.BytesIO(data)) as im:
        if im.size != tuple(expected):
            raise ValueError(f'API returned {im.size}, expected {tuple(expected)}. Refusing a silent resize/crop.')
        return _tensor(im.convert('RGB'))


class WBGSPartnerInputs:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'reference_package': ('WB_GS_REFERENCE',)}}
    RETURN_TYPES = ('IMAGE', 'IMAGE')
    RETURN_NAMES = ('image_1_original', 'image_2_green_geometry')
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Ordered original and 4K reference for the native ComfyUI Partner GPT node. No OpenAI API key.'
    def run(self, reference_package):
        return reference_package['source'], reference_package['reference4k']


class WBGSPartnerResult:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'reference_package': ('WB_GS_REFERENCE',), 'gpt_image': ('IMAGE',)}}
    RETURN_TYPES = ('WB_GS_API', 'IMAGE')
    RETURN_NAMES = ('gpt_result', 'raw_gpt_native')
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Accept the logged-in Comfy Partner node image and pass it to matte lock and comparison.'
    def run(self, reference_package, gpt_image):
        if gpt_image.ndim != 4 or gpt_image.shape[0] != 1:
            raise ValueError('Partner GPT must return exactly one image (n=1).')
        if tuple(gpt_image.shape[1:3]) != (API_SIZE[1], API_SIZE[0]):
            raise ValueError(f'Partner GPT image is {tuple(gpt_image.shape[1:3])}; expected {API_SIZE[1]}x{API_SIZE[0]}. Check custom size.')
        native = gpt_image[..., :3].detach().float().cpu().clamp(0,1)
        return ({**reference_package, 'raw_native': native, 'raw_png': _png(native),
            'api_report': {'provider': 'ComfyUI Partner Node', 'model': 'gpt-image-2.5-sunburst',
                           'size': list(API_SIZE), 'auth': 'Comfy account login and Comfy Credits',
                           'pixel_perfect_pose_verified': False}}, native)


def _overlay_metrics(target, candidate):
    a, b = target[0].numpy() > .5, candidate[0].numpy() > .5
    union, intersect = int((a | b).sum()), int((a & b).sum())
    report = {'mask_iou': intersect / max(1, union), 'mask_disagreement_pixels': int((a != b).sum()),
              'pixel_perfect_pose_verified': False,
              'interpretation': 'SAM silhouette diagnostics only. Matching segmentation does not prove matching internal pose, identity or clothing.'}
    if not a.any() or not b.any():
        return {**report, 'review_required': True, 'warning': 'One silhouette is empty.'}
    ay, ax = np.where(a); by, bx = np.where(b)
    report['centroid_offset_xy_pixels'] = [float(bx.mean()-ax.mean()), float(by.mean()-ay.mean())]
    report['bbox_delta_ltrb_pixels'] = [int(bx.min()-ax.min()), int(by.min()-ay.min()),
                                       int(bx.max()-ax.max()), int(by.max()-ay.max())]
    ea = a & ~binary_erosion(a, border_value=0)
    eb = b & ~binary_erosion(b, border_value=0)
    distances = np.concatenate((distance_transform_edt(~ea)[eb], distance_transform_edt(~eb)[ea]))
    report['symmetric_boundary_p95_pixels'] = float(np.percentile(distances, 95))
    report['review_required'] = report['mask_iou'] < .99 or report['symmetric_boundary_p95_pixels'] > 2
    return report


def _finish_at(p, size, edge_feather):
    original = _resize(p['source'], size)
    guide = _resize(p['reference4k'], size)
    raw = _resize(p['raw_native'], size)
    protect = _resize_mask(p['protect'], size)
    edit = _resize_mask(p['edit'], size) * (1 - protect)
    body = _resize_mask(p['target_body4k'], size) * edit
    binary = body[0].numpy() > .5
    if not binary.any():
        raise ValueError('The target matte is empty.')
    # Green-free boundary plate. Only a narrow inside feather samples this plate.
    # Interior RGB is supplied by GPT, not this nearest-exterior fill.
    if binary.all():
        # A legitimate full-frame subject needs no exterior plate or edge fade.
        plate = raw.clone()
    else:
        idx = distance_transform_edt(binary, return_distances=False, return_indices=True)
        ga = guide[0].numpy()
        plate = np.where(binary[..., None], ga[idx[0], idx[1]], ga)
        plate = torch.from_numpy(plate.copy()).unsqueeze(0)
    alpha = _inward(body, edge_feather)
    finished = _composite(plate, raw, alpha)
    # Reapply immutable source pixels after EVERY resize and composite.
    protected = (edit == 0) | (protect > .5)
    finished = torch.where(protected[..., None], original, finished)
    changed = int(torch.count_nonzero(finished[protected] != original[protected]))
    if changed:
        raise ValueError('Protected-pixel verification failed.')
    return finished, {'size': list(size), 'protected_changed_channels': changed,
                      'matte_pixels': int(binary.sum()), 'edge_feather_pixels': edge_feather,
                      'exterior_plate': 'Qwen reference inside original edit area; original source elsewhere',
                      'unseen_background': 'Vacated original garment areas require Qwen-synthesized background.'}


class WBGSFinalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'gpt_result': ('WB_GS_API',),
            'edge_feather_pixels': ('INT', {'default': 2, 'min': 0, 'max': 16}),
            'measure_raw_silhouette': ('BOOLEAN', {'default': True}),
        }}
    RETURN_TYPES = ('WB_GS_FINAL', 'IMAGE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('final_package', 'locked_native', 'locked_4k', 'diagnostics')
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Keeps hair/outside-mask pixels exact. Uses the reference matte, not a claim that GPT pose is pixel-perfect. Saves raw GPT separately.'

    @torch.inference_mode()
    def run(self, gpt_result, edge_feather_pixels=2, measure_raw_silhouette=True):
        p = gpt_result
        native, native_report = _finish_at(p, API_SIZE, edge_feather_pixels)
        fourk, fourk_report = _finish_at(p, REF_SIZE, edge_feather_pixels)
        diagnostics = {'pixel_perfect_pose_verified': False, 'review_required': True,
                       'interpretation': 'Raw silhouette measurement disabled.'}
        if measure_raw_silhouette:
            try:
                masks = _sam(p['raw_native'], [p['person_prompt']], p['threshold'])[0]
                target = _resize_mask(p['target_person4k'], API_SIZE)
                candidate = _choose(masks, reference=target)
                diagnostics = _overlay_metrics(target, candidate)
            except Exception as exc:
                # The paid image and finished images remain available even if QA fails.
                diagnostics = {'pixel_perfect_pose_verified': False, 'review_required': True,
                    'interpretation': 'SAM diagnostic failed; inspect the saved raw result.',
                    'error_type': type(exc).__name__}
        report = {'version': VERSION, 'mask': p['mask_report'], 'api': p['api_report'],
                  'reference_size': list(REF_SIZE), 'upscale_method': p['upscale_method'],
                  'green_suit_coverage': p['green_coverage'], 'native': native_report,
                  'fourk': fourk_report, 'geometry_diagnostics': diagnostics,
                  'fourk_final_is': 'Lanczos-resized GPT candidate composited at 2720x4080, NOT native 4K GPT generation',
                  'limitations': 'A fixed matte constrains the compositing boundary, not internal anatomy. Background/holes inside an undersized GPT person are not repaired by matte clipping.'}
        text = json.dumps(report, indent=2)
        return {**p, 'locked_native': native, 'locked4k': fourk, 'report': report}, native, fourk, text


class WBGSReviewCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'source_and_masks': ('WB_GS_MASK',),
            'run_mode': (MODES,),
            'filename_prefix': ('STRING', {'default': 'GreenSuit/Progression'}),
        }, 'optional': {
            'reference_package': ('WB_GS_REFERENCE', {'lazy': True}),
            'final_package': ('WB_GS_FINAL', {'lazy': True}),
        }}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'run'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Single output gate. Mask-only runs no Qwen/upscaler/API. Reference-only never runs GPT. A/B wipe and 50% overlay compare every saved stage.'

    def check_lazy_status(self, source_and_masks, run_mode=MODES[0], filename_prefix='GreenSuit/Progression',
                          reference_package=None, final_package=None):
        if run_mode == MODES[2] and final_package is None:
            return ['final_package']
        if run_mode == MODES[1] and reference_package is None:
            return ['reference_package']
        return []

    def run(self, source_and_masks, run_mode=MODES[0], filename_prefix='GreenSuit/Progression',
            reference_package=None, final_package=None):
        import folder_paths
        if run_mode not in MODES:
            raise ValueError('Unknown run mode.')
        p = source_and_masks
        stages = [('Original', p['source'], None), ('SAM mask', _preview(p['source'], p['edit']), None)]
        report = {'run_mode': run_mode, 'version': VERSION, 'mask': p['mask_report']}
        if run_mode == MODES[1]:
            if reference_package is None:
                raise ValueError('The 4K reference output is not connected.')
            p = reference_package
        elif run_mode == MODES[2]:
            if final_package is None:
                raise ValueError('The final package output is not connected.')
            p = final_package
        if run_mode != MODES[0]:
            stages += [('Qwen green suit', p['green'], None), ('4K geometry reference', p['reference4k'], None)]
            report.update(reference_size=list(REF_SIZE), upscale_method=p['upscale_method'], green_suit_coverage=p['green_coverage'])
        if run_mode == MODES[2]:
            stages += [('Raw GPT native', p['raw_native'], p['raw_png']),
                       ('Matte-locked native', p['locked_native'], None), ('Matte-locked 4K', p['locked4k'], None)]
            report = p['report']
        prefix = filename_prefix.strip()
        if not prefix or Path(prefix).is_absolute() or '..' in Path(prefix).parts or '\\' in prefix:
            raise ValueError('Use a relative output prefix without parent-directory traversal.')
        directory, base, counter, subfolder, _ = folder_paths.get_save_image_path(prefix, folder_paths.get_output_directory(), *REF_SIZE)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        stem = f'{base}_{counter:05}_{uuid.uuid4().hex[:8]}'
        temp = Path(folder_paths.get_temp_directory()); temp.mkdir(parents=True, exist_ok=True)
        previews, saved = [], []
        for i, (label, image, raw_bytes) in enumerate(stages):
            name = f'{stem}_{i:02}.png'
            (root / name).write_bytes(raw_bytes or _png(image))
            saved.append({'filename': name, 'subfolder': subfolder, 'type': 'output', 'label': label})
            thumb_name = f'wbgscmp_{uuid.uuid4().hex}_{i}.png'
            (temp / thumb_name).write_bytes(_png(_resize(image, (640, 960))))
            previews.append({'filename': thumb_name, 'subfolder': '', 'type': 'temp', 'label': label})
        Image.fromarray(np.rint(source_and_masks['edit'][0].numpy()*255).astype(np.uint8)).save(root / (stem + '_source_edit_mask.png'))
        if run_mode != MODES[0]:
            Image.fromarray(np.rint(p['target_person4k'][0].numpy()*255).astype(np.uint8)).save(root / (stem + '_target_silhouette_4k.png'))
        (root / (stem + '_report.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
        return {'ui': {'images': saved[-1:], 'wbgs_images': previews, 'wbgs_saved': saved,
                       'wbgs_report': [json.dumps(report, indent=2)]}}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (WBGSBodyMask, WBGSQwenSuit, WBGS4KReference,
                          WBGSPartnerInputs, WBGSPartnerResult, WBGSFinalize, WBGSReviewCompare)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'WBGSBodyMask': 'Green Suit | 1. SAM person minus hair',
    'WBGSQwenSuit': 'Green Suit | 2. Qwen BF16 suit edit',
    'WBGS4KReference': 'Green Suit | 3. Protected 4K reference',
    'WBGSPartnerInputs': 'Green Suit | 4a. Partner reference inputs',
    'WBGSPartnerResult': 'Green Suit | 4b. Partner result adapter',
    'WBGSFinalize': 'Green Suit | 5. Matte lock and geometry QA',
    'WBGSReviewCompare': 'Green Suit | 6. Run / compare / save progression',
}
