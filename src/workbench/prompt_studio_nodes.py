"""Standalone visual prompt studio. No image-generation model is loaded here."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

from workbench.prompt_studio import (VERSION, TASKS, DETAILS, DEFAULT_GUIDANCE,
                                     StudioOptions, StudioError, generate_prompt)

NONE = 'None'
CATEGORY = 'Workbench / Qwen Prompt Studio'
SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


class WBPromptStudioImage:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        root = Path(folder_paths.get_input_directory())
        files = sorted(str(p.relative_to(root)).replace('\\', '/') for p in root.rglob('*')
                       if p.is_file() and p.suffix.lower() in SUFFIXES)
        return {'required': {
            'image': ([NONE] + files, {'default': NONE, 'image_upload': True}),
            'role_hint': ('STRING', {'default': '', 'multiline': True,
                'tooltip': 'Optional. Examples: original scene; replacement dog face; same dog full body. Blank lets the model infer the role.'}),
        }}

    RETURN_TYPES = ('WB_PROMPT_IMAGE',)
    RETURN_NAMES = ('reference',)
    FUNCTION = 'load'
    CATEGORY = CATEGORY
    DESCRIPTION = 'Upload/select a still image. None disables the slot. All eight slots are already connected; fill them in numbered order.'

    @classmethod
    def VALIDATE_INPUTS(cls, image, role_hint=''):
        if image == NONE:
            return True
        import folder_paths
        if not folder_paths.exists_annotated_filepath(image):
            return 'Select an uploaded image, or leave this optional slot set to None.'
        return True

    @classmethod
    def IS_CHANGED(cls, image, role_hint=''):
        if image == NONE:
            return NONE
        import folder_paths
        try:
            path = folder_paths.get_annotated_filepath(image)
            h = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            return h.hexdigest()
        except OSError:
            return float('nan')

    def load(self, image=NONE, role_hint=''):
        if image == NONE:
            return (None,)
        if not isinstance(role_hint, str) or len(role_hint) > 2000:
            raise StudioError('Image role hint must be text under 2,000 characters.')
        import nodes
        value = nodes.LoadImage().load_image(image)[0]
        if value.ndim != 4 or value.shape[0] != 1:
            raise StudioError('Use one still image per slot, not an animated image or batch.')
        return ({'image': value, 'role': role_hint.strip()},)


def collect_references(reference_1, optional):
    allowed = {f'reference_{i}' for i in range(2, 9)}
    if set(optional) - allowed:
        raise StudioError('Unexpected reference input. Use Image 1 through Image 8.')
    refs = []
    gap = False
    for i in range(1, 9):
        value = reference_1 if i == 1 else optional.get(f'reference_{i}')
        if value is None:
            gap = True
        elif gap:
            raise StudioError('Fill image slots consecutively starting with Image 1. '
                              f'Image {i} follows an empty slot; no images were silently renumbered.')
        else:
            refs.append(value)
    if not refs:
        raise StudioError('Load an image in Image 1. Additional slots may stay None.')
    return refs


def save_text(prompt, prefix='qwen_prompt'):
    import folder_paths
    if not isinstance(prefix, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', prefix):
        raise StudioError('Filename prefix must be 1-80 letters, digits, dots, underscores or hyphens; no folders.')
    root = Path(folder_paths.get_output_directory()).resolve()
    target_dir = root / 'prompt_studio'
    if target_dir.is_symlink():
        raise StudioError('Refusing a symlinked prompt output folder.')
    target_dir.mkdir(parents=True, exist_ok=True)
    if not target_dir.resolve().is_relative_to(root):
        raise StudioError('Prompt output must remain inside ComfyUI output.')
    filename = prefix + '_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:10] + '.txt'
    path = target_dir / filename
    # Unique, exclusive creation. Export no inputs, analysis, JSON or metadata.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(prompt + '\n')
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return filename, 'prompt_studio/' + filename


class WBQwenPromptStudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'reference_1': ('WB_PROMPT_IMAGE',),
            'request': ('STRING', {'default': 'Write a faithful text-to-image prompt that recreates image 1.', 'multiline': True}),
            'task': (list(TASKS), {'default': TASKS[0]}),
            'detail': (list(DETAILS), {'default': 'Detailed'}),
            'save_txt': ('BOOLEAN', {'default': True}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 2147483647,
                             'control_after_generate': False}),
            'writing_guidance': ('STRING', {'default': DEFAULT_GUIDANCE, 'multiline': True,
                'tooltip': 'Editable writing instructions for THIS workflow, not the old automatic enhancer.'}),
            'keep_exact': ('STRING', {'default': '', 'multiline': True,
                'tooltip': 'Optional exact text or LoRA trigger, one per line. Does not load or train a LoRA.'}),
            'vision_edge': ('INT', {'default': 1024, 'min': 512, 'max': 1536, 'step': 128}),
            'max_tokens': ('INT', {'default': 4096, 'min': 1024, 'max': 8192, 'step': 512}),
            'filename_prefix': ('STRING', {'default': 'qwen_prompt'}),
        }, 'optional': {f'reference_{i}': ('WB_PROMPT_IMAGE',) for i in range(2, 9)}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('prompt',)
    OUTPUT_NODE = True
    FUNCTION = 'write_prompt'
    CATEGORY = CATEGORY
    DESCRIPTION = ('Analyzes 1-8 images and writes a Qwen Image 2.1 edit, new-scene, recreation or description prompt. '
                   'Only final text is exported. Uses WB_QWEN_PROMPT_MODEL, not the image diffusion model.')

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        from workbench.qwen_prompt_local import cache_identity
        return VERSION + '|' + cache_identity()

    def write_prompt(self, reference_1, request, task=TASKS[0], detail='Detailed',
                     save_txt=True, seed=42, writing_guidance=DEFAULT_GUIDANCE,
                     keep_exact='', vision_edge=1024, max_tokens=4096,
                     filename_prefix='qwen_prompt', **references):
        refs = collect_references(reference_1, references)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', filename_prefix):
            raise StudioError('Use a simple filename prefix without folders.')
        options = StudioOptions(task=task, detail=detail, guidance=writing_guidance,
                                keep_exact=keep_exact, seed=seed,
                                vision_edge=vision_edge, max_tokens=max_tokens)
        prompt, report = generate_prompt(request, refs, options)
        filename, relative = ('qwen_prompt.txt', '')
        if save_txt:
            filename, relative = save_text(prompt, filename_prefix)
        status = (f'{report["task"]} | {report["image_count"]} image(s) | '
                  f'{report["model"]} | {report["precision"]} | vision model unloaded')
        roles = '\n'.join(f'Image {r["index"]}: {r["use_for"]}' for r in report['roles'])
        if report['uncertainties']:
            roles += '\nReview: ' + '; '.join(report['uncertainties'])
        # UI metadata is not part of the STRING output or exported .txt.
        return {'ui': {'text': [prompt], 'studio_status': [status],
                       'studio_roles': [roles], 'studio_filename': [filename],
                       'studio_saved': [relative]}, 'result': (prompt,)}


NODE_CLASS_MAPPINGS = {'WBPromptStudioImage': WBPromptStudioImage,
                       'WBQwenPromptStudio': WBQwenPromptStudio}
NODE_DISPLAY_NAME_MAPPINGS = {'WBPromptStudioImage': 'Prompt Studio | Optional Image + Role',
                              'WBQwenPromptStudio': 'Qwen | Smart Visual Prompt Studio (TEXT ONLY)'}
