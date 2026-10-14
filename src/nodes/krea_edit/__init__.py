"""Krea 2 reference editing with shared Qwen vision and optional photo tools.

The upstream editor has TWO native image inputs. Extra references are visibly
combined into a labeled reference sheet, never claimed as native N-image support.
"""
from __future__ import annotations
import base64
import hashlib
import io
import math
import re
import numpy as np
import torch
from PIL import Image, ImageDraw
import folder_paths
import nodes
from workbench import registry
from .prompt_nodes import WBKreaPrompt, WBKreaStandardPrompt
from .photo_nodes import load_srgb

MODEL = 'auto'
EDIT_ADAPTER = 'krea2_identity_edit_v1_2.safetensors'
NONE = 'none'
CATEGORY = 'Workbench/Krea 2'


def canvas_size(width: int, height: int, megapixels: float) -> tuple[int, int]:
    if min(width, height) < 64 or not 0.25 <= megapixels <= 2.0:
        raise ValueError('Use an image at least 64 px per side and 0.25 to 2.0 MP.')
    scale = min(1.0, math.sqrt(megapixels * 1_000_000 / (width * height)))
    w, h = int(width * scale) // 16 * 16, int(height * scale) // 16 * 16
    if min(w, h) < 64:
        raise ValueError('The source aspect ratio is too extreme for this canvas size.')
    return w, h


def pil_image(image):
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError('Each source/reference must be one still RGB/RGBA image, not a batch.')
    if not torch.isfinite(image).all():
        raise ValueError('Source/reference contains invalid pixels.')
    return Image.fromarray((image[0, :, :, :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8))


def tensor_image(image):
    return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).float().unsqueeze(0) / 255


def make_board(refs):
    if not refs:
        return None, ''
    if len(refs) == 1:
        return refs[0]['image'], 'Reference R1 is the entire second image: ' + refs[0]['role']
    cols = 2
    rows = math.ceil(len(refs) / cols)
    side, header = 704, 40
    board = Image.new('RGB', (cols * side, rows * side), (235, 235, 235))
    draw = ImageDraw.Draw(board)
    labels = []
    for i, ref in enumerate(refs):
        x, y = (i % cols) * side, (i // cols) * side
        photo = pil_image(ref['image'])
        photo.thumbnail((side - 16, side - header - 16), Image.Resampling.LANCZOS)
        board.paste(photo, (x + (side - photo.width) // 2, y + header + (side - header - photo.height) // 2))
        draw.text((x + 16, y + 8), 'R' + str(i + 1), fill=(0, 0, 0), font_size=24)
        place = ('top' if i < 2 else 'bottom') + '-' + ('left' if i % 2 == 0 else 'right')
        labels.append(f'R{i+1} ({place}): {ref["role"]}')
    return tensor_image(board), '; '.join(labels)


def vision_image(image):
    """Legacy utility retained for saved integrations; runtime uses shared VLM I/O."""
    photo = pil_image(image)
    photo.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    photo.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def clean_response(data):
    """Legacy response parser, retained for callers outside the workflow nodes."""
    if data.get('error') or data.get('done_reason') in ('length', 'max_tokens'):
        raise RuntimeError('Qwen enhancement failed or was truncated. Increase max_tokens or disable enhancement explicitly.')
    text = str(data.get('message', {}).get('content', '')).strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    if text.startswith('```'):
        text = re.sub(r'^```[^\n]*\n|\n```$', '', text).strip()
    if not text or '<think>' in text or len(text) > 16000:
        raise RuntimeError('Qwen returned no usable editing instruction. No silent prompt fallback was applied.')
    return text


class WBKreaReference:
    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list('input') if 'input' in folder_paths.folder_names_and_paths else []
        if not files:
            from pathlib import Path
            root = Path(folder_paths.get_input_directory())
            files = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'))
        return {'required': {'enabled': ('BOOLEAN', {'default': False}),
                             'image': ([NONE] + files, {'image_upload': True}),
                             'role': ('STRING', {'default': 'facial identity', 'multiline': True})}}
    RETURN_TYPES = ('WB_KREA_REF',)
    RETURN_NAMES = ('reference',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    @classmethod
    def VALIDATE_INPUTS(cls, enabled, image, **kwargs):
        if not enabled:
            return True
        return True if image != NONE and folder_paths.exists_annotated_filepath(image) else 'Upload/select an image for the enabled reference, or turn it off.'
    @classmethod
    def IS_CHANGED(cls, enabled, image, **kwargs):
        if not enabled:
            return 'disabled'
        path = folder_paths.get_annotated_filepath(image)
        try:
            with open(path, 'rb') as handle:
                return hashlib.file_digest(handle, 'sha256').hexdigest()
        except OSError:
            return float('nan')
    def load(self, enabled, image, role):
        if not enabled:
            return (None,)
        if image == NONE:
            raise ValueError('Select a photo for the enabled reference.')
        photo = load_srgb(image)[0]
        pil_image(photo)
        return ({'image': photo, 'role': role.strip() or 'appearance reference'},)


class WBKreaPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source': ('IMAGE',), 'megapixels': ('FLOAT', {'default': 1.9, 'min': .25, 'max': 2.0, 'step': .05})},
                'optional': {f'reference_{i}': ('WB_KREA_REF',) for i in range(1, 5)}}
    RETURN_TYPES = ('WB_KREA_PACKAGE', 'IMAGE')
    RETURN_NAMES = ('editing_inputs', 'reference_sheet_preview')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY
    def prepare(self, source, megapixels, **kwargs):
        photo = pil_image(source)
        refs = [kwargs[f'reference_{i}'] for i in range(1, 5) if kwargs.get(f'reference_{i}') is not None]
        board, roles = make_board(refs)
        package = {'source': source, 'references': refs, 'board': board, 'roles': roles,
                   'size': canvas_size(photo.width, photo.height, megapixels)}
        return package, board if board is not None else source


class WBKreaOptionalLoRAs:
    @classmethod
    def INPUT_TYPES(cls):
        choices = [name for name in registry.choices('krea', 'loras') if name != EDIT_ADAPTER]
        return {'required': {'model': ('MODEL',),
            'lora_1': ([NONE] + choices,), 'strength_1': ('FLOAT', {'default': 1.0, 'min': -2.0, 'max': 2.0, 'step': .05}),
            'lora_2': ([NONE] + choices,), 'strength_2': ('FLOAT', {'default': 1.0, 'min': -2.0, 'max': 2.0, 'step': .05})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY
    def apply(self, model, lora_1, strength_1, lora_2, strength_2):
        applied = set()
        for name, strength in ((lora_1, strength_1), (lora_2, strength_2)):
            if name == NONE or strength == 0:
                continue
            if name == EDIT_ADAPTER:
                raise ValueError('Use the Identity Edit workflow for the Identity Edit adapter.')
            if name in applied:
                raise ValueError('The same LoRA is selected more than once.')
            registry.check_selection('krea', 'loras', name)
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, name, strength)[0]
            applied.add(name)
        return (model,)


class WBKreaModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'ready_prompt': ('STRING', {'forceInput': True}),
            'checkpoint': (registry.choices('krea', 'diffusion_models'),),
            'text_encoder': (registry.choices('krea', 'text_encoders'),),
            'vae_name': (registry.choices('krea', 'vae'),),
            'sampling_preset': (['Auto', 'Raw', 'Turbo', 'Custom'],),
            'custom_steps': ('INT', {'default': 20, 'min': 1, 'max': 100}),
            'custom_cfg': ('FLOAT', {'default': 3.0, 'min': 1.0, 'max': 10.0, 'step': .1})}}
    RETURN_TYPES = ('MODEL', 'CLIP', 'VAE', 'INT', 'FLOAT')
    RETURN_NAMES = ('model', 'krea_encoder', 'vae', 'steps', 'cfg')
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    def load(self, ready_prompt, checkpoint, text_encoder, vae_name, sampling_preset, custom_steps, custom_cfg):
        for kind, name in [('diffusion_models', checkpoint), ('text_encoders', text_encoder), ('vae', vae_name)]:
            registry.check_selection('krea', kind, name)
        if sampling_preset == 'Auto':
            if checkpoint == 'krea2_raw_bf16.safetensors':
                sampling_preset = 'Raw'
            elif checkpoint == 'krea2_turbo_bf16.safetensors':
                sampling_preset = 'Turbo'
            else:
                raise ValueError('For a custom Krea 2 checkpoint, select Raw, Turbo or Custom sampling explicitly.')
        steps, cfg = {'Raw': (20, 3.0), 'Turbo': (8, 1.0), 'Custom': (custom_steps, custom_cfg)}[sampling_preset]
        model = nodes.UNETLoader().load_unet(checkpoint, 'default')[0]
        if not hasattr(model.model.diffusion_model, '_unpack_context'):
            raise ValueError('This checkpoint is not a supported Krea 2 model. Qwen/FLUX/SD checkpoints cannot be substituted here.')
        clip = nodes.CLIPLoader().load_clip(text_encoder, type='krea2')[0]
        vae = nodes.VAELoader().load_vae(vae_name)[0]
        return model, clip, vae, steps, cfg


class WBKreaStandardModels(WBKreaModels):
    """Native T2I uses its own Raw recipe, not Identity Edit's removal recipe.

    Official Krea Raw example: 52 steps, guidance 3.5 in Krea's convention.
    ComfyUI KSampler uses the conventional CFG baseline, hence 4.5 here.
    Turbo remains the recommended native-generation default (8 steps, CFG 1).
    """
    def load(self, ready_prompt, checkpoint, text_encoder, vae_name,
             sampling_preset, custom_steps, custom_cfg):
        native_raw = sampling_preset == 'Raw' or (
            sampling_preset == 'Auto' and checkpoint == 'krea2_raw_bf16.safetensors')
        if native_raw:
            sampling_preset, custom_steps, custom_cfg = 'Custom', 52, 4.5
        return super().load(ready_prompt, checkpoint, text_encoder, vae_name,
                            sampling_preset, custom_steps, custom_cfg)


class WBKreaLoRAs:
    @classmethod
    def INPUT_TYPES(cls):
        choices = registry.choices('krea', 'loras')
        return {'required': {'model': ('MODEL',), 'edit_adapter': ([EDIT_ADAPTER],),
            'edit_strength': ('FLOAT', {'default': 1.0, 'min': 0.05, 'max': 2.0, 'step': .05}),
            'lora_1': ([NONE] + choices,), 'strength_1': ('FLOAT', {'default': 1.0, 'min': -2.0, 'max': 2.0, 'step': .05}),
            'lora_2': ([NONE] + choices,), 'strength_2': ('FLOAT', {'default': 1.0, 'min': -2.0, 'max': 2.0, 'step': .05})}}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'apply'
    CATEGORY = CATEGORY
    def apply(self, model, edit_adapter, edit_strength, lora_1, strength_1, lora_2, strength_2):
        if edit_adapter != EDIT_ADAPTER or edit_strength <= 0:
            raise ValueError('The Identity Edit adapter must stay enabled for this editing workflow.')
        applied = set()
        for name, strength in [(edit_adapter, edit_strength), (lora_1, strength_1), (lora_2, strength_2)]:
            if name == NONE or strength == 0:
                continue
            if name in applied:
                raise ValueError('The same LoRA is selected more than once. Use one slot and adjust its strength.')
            registry.check_selection('krea', 'loras', name)
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, name, strength)[0]
            applied.add(name)
        return (model,)


class WBKreaCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'clip': ('CLIP',), 'vae': ('VAE',),
            'editing_inputs': ('WB_KREA_PACKAGE',), 'prompt': ('STRING', {'forceInput': True}),
            'scene_strength': ('FLOAT', {'default': 1.0, 'min': .1, 'max': 10.0, 'step': .1, 'tooltip': 'Scene strength for image 1 when a second reference is present. No effect in single-image mode.'}),
            'reference_strength': ('FLOAT', {'default': 4.0, 'min': .1, 'max': 10.0, 'step': .1, 'tooltip': 'Fidelity of the last reference: your source in single-image mode, or the identity reference/sheet in two-input mode. Start at 4; lower when an intended edit is resisted.'}),
            'grounding_px': ('INT', {'default': 768, 'min': 384, 'max': 1024, 'step': 64})}}
    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT')
    RETURN_NAMES = ('edit_model', 'positive', 'grounded_negative', 'target_latent')
    FUNCTION = 'condition'
    CATEGORY = CATEGORY
    def condition(self, model, clip, vae, editing_inputs, prompt, scene_strength, reference_strength, grounding_px):
        required = ('Krea2EditModelPatch', 'Krea2EditGroundedEncode', 'EmptySD3LatentImage')
        if any(name not in nodes.NODE_CLASS_MAPPINGS for name in required):
            raise RuntimeError('Krea2Edit or native Krea dependencies are missing. Build the krea workspace, not a Qwen image.')
        p = editing_inputs
        w, h = p['size']
        latent = nodes.NODE_CLASS_MAPPINGS['EmptySD3LatentImage']().generate(w, h, 1)[0]
        encoder = nodes.NODE_CLASS_MAPPINGS['Krea2EditGroundedEncode']()
        positive = encoder.encode(clip, prompt, image=p['source'], image_b=p['board'], grounding_px=grounding_px)[0]
        negative = encoder.encode(clip, '', image=p['source'], image_b=p['board'], grounding_px=grounding_px)[0]
        patched = nodes.NODE_CLASS_MAPPINGS['Krea2EditModelPatch']().patch(
            model, source_latent=latent, vae=vae, source_image=p['source'], source_image_b=p['board'],
            target_latent=latent, fit_mode='fit', ref_boost_a=scene_strength,
            ref_boost=reference_strength)[0]
        return patched, positive, negative, latent


class WBKreaStandardCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'clip': ('CLIP',),
            'editing_inputs': ('WB_KREA_PACKAGE',), 'prompt': ('STRING', {'forceInput': True})}}
    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT')
    RETURN_NAMES = ('model', 'positive', 'negative', 'target_latent')
    FUNCTION = 'condition'
    CATEGORY = CATEGORY
    def condition(self, model, clip, editing_inputs, prompt):
        w, h = editing_inputs['size']
        latent = nodes.NODE_CLASS_MAPPINGS['EmptySD3LatentImage']().generate(w, h, 1)[0]
        encoder = nodes.NODE_CLASS_MAPPINGS['CLIPTextEncode']()
        positive = encoder.encode(clip, prompt)[0]
        negative = encoder.encode(clip, '')[0]
        return model, positive, negative, latent


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (WBKreaReference, WBKreaPrepare,
    WBKreaPrompt, WBKreaModels, WBKreaStandardModels, WBKreaLoRAs, WBKreaCondition, WBKreaStandardCondition,
    WBKreaStandardPrompt, WBKreaOptionalLoRAs)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'WBKreaReference': 'Krea Optional Reference', 'WBKreaPrepare': 'Krea Source + Reference Sheet',
    'WBKreaPrompt': 'Krea | Shared Qwen Vision Prompt Enhancer',
    'WBKreaModels': 'Krea Checkpoint + Correct Encoder',
    'WBKreaStandardModels': 'Krea Native Generation Checkpoint + Correct Encoder',
    'WBKreaLoRAs': 'Krea Edit Adapter + Optional LoRAs',
    'WBKreaCondition': 'Krea FIT Editing Conditioning',
    'WBKreaStandardCondition': 'Krea Standard Native Conditioning (no Edit LoRA)',
    'WBKreaStandardPrompt': 'Krea Standard Target-Image Prompt',
    'WBKreaOptionalLoRAs': 'Krea Optional LoRAs (no Identity Edit)'}
from .photo_nodes import NODE_CLASS_MAPPINGS as _PHOTO_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as _PHOTO_NAMES
NODE_CLASS_MAPPINGS.update(_PHOTO_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_PHOTO_NAMES)
# Compatibility alias for the shared source-mask Toolbox; no Qwen generator is registered.
NODE_CLASS_MAPPINGS['Q21PhotoMask'] = _PHOTO_CLASSES['WBKreaMask']
NODE_DISPLAY_NAME_MAPPINGS['Q21PhotoMask'] = 'Krea | Shared Manual / SAM 3.1 Mask'
WEB_DIRECTORY = './web'
