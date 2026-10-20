"""I/O helpers only. H3 conditioning and sampling use inspected native Comfy nodes."""
from __future__ import annotations
from pathlib import Path
import re
import uuid

import folder_paths


def _roots():
    return {'output': Path(folder_paths.get_output_directory()).resolve(),
            'input': Path(folder_paths.get_input_directory()).resolve()}


def _resolve_selection(selection):
    match = re.fullmatch(r'(input|output)::(.+)', selection)
    if not match:
        raise ValueError('Select a saved native H3 .mkv master. Refresh ComfyUI after generation.')
    kind, relative = match.groups()
    root = _roots()[kind]
    candidate = root / relative
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
        raise ValueError('The selected master must remain inside ComfyUI input or output storage.')
    if not candidate.is_file() or candidate.suffix.lower() != '.mkv':
        raise ValueError('The selected MKV master is unavailable.')
    return candidate


class WBH3LosslessMaster:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',),
                            'fps': ('FLOAT', {'default': 24.0, 'min': 1.0, 'max': 120.0}),
                            'filename_prefix': ('STRING', {'default': 'video/H3_First_Last/native/H3_native'})},
                'optional': {'audio': ('AUDIO',)}}
    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('images', 'master_file')
    FUNCTION = 'save'
    CATEGORY = 'Workbench/H3 HQ'
    DESCRIPTION = 'Write an RGB16 FFV1 MKV master, then pass the frames to the MP4 preview. No face restoration or frame replacement.'

    def save(self, images, fps, filename_prefix, audio=None):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from workbench.h3_hq_media import save_master
        if not re.fullmatch(r'[A-Za-z0-9_./ -]{1,180}', filename_prefix):
            raise ValueError('Use a simple filename prefix containing letters, numbers, spaces, underscores or slashes.')
        parts = Path(filename_prefix).parts
        if Path(filename_prefix).is_absolute() or '..' in parts:
            raise ValueError('The output prefix must stay inside ComfyUI output storage.')
        root = _roots()['output']
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, str(root), int(images.shape[2]), int(images.shape[1]))
        dest = Path(folder) / f'{name}_{counter:05d}_{uuid.uuid4().hex[:10]}.mkv'
        if not dest.resolve().is_relative_to(root):
            raise ValueError('The output folder escapes ComfyUI output storage.')
        record = save_master(images, dest, fps, audio, throw_exception_if_processing_interrupted)
        print('[H3 HQ] Saved lossless RGB16 master: ' + record['path'])
        return images, dest.relative_to(root).as_posix()


class WBH3LoadNativeMaster:
    @classmethod
    def INPUT_TYPES(cls):
        choices = []
        for kind, root in _roots().items():
            for path in sorted(root.rglob('*.mkv')):
                if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
                    relative = path.relative_to(root).as_posix()
                    if '/4k/' not in '/' + relative.lower():
                        choices.append(kind + '::' + relative)
        return {'required': {'master_file': (choices or ['Select a native H3 MKV master after generation'],)}}
    RETURN_TYPES = ('IMAGE', 'AUDIO', 'FLOAT')
    RETURN_NAMES = ('images', 'audio', 'fps')
    FUNCTION = 'load'
    CATEGORY = 'Workbench/H3 HQ'
    DESCRIPTION = 'Read the native lossless MKV directly from output or input. No download/re-upload and no 8-bit MP4 intermediate.'

    @classmethod
    def IS_CHANGED(cls, master_file):
        path = _resolve_selection(master_file)
        stat = path.stat()
        return f'{stat.st_mtime_ns}:{stat.st_size}'

    @classmethod
    def VALIDATE_INPUTS(cls, master_file):
        try:
            _resolve_selection(master_file)
        except ValueError as exc:
            return str(exc)
        return True

    def load(self, master_file):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from workbench.h3_hq_media import read_native_master
        return read_native_master(_resolve_selection(master_file), throw_exception_if_processing_interrupted)


class WBH3ReleaseMemory:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',)}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'release'
    CATEGORY = 'Workbench/H3 HQ'
    DESCRIPTION = 'Offload Comfy-managed models before the separate SeedVR2 pass. Does not delete model files or user outputs.'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Memory residency is external state; do not reuse a cached offload operation.
        return float('nan')

    def release(self, images):
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
        return (images,)


NODE_CLASS_MAPPINGS = {'WBH3LosslessMaster': WBH3LosslessMaster,
                       'WBH3LoadNativeMaster': WBH3LoadNativeMaster,
                       'WBH3ReleaseMemory': WBH3ReleaseMemory}
NODE_DISPLAY_NAME_MAPPINGS = {'WBH3LosslessMaster': 'H3 HQ - Save Lossless RGB16 Master',
                              'WBH3LoadNativeMaster': 'H3 HQ - Load Native Master',
                              'WBH3ReleaseMemory': 'H3 HQ - Release VRAM Before Finishing'}
