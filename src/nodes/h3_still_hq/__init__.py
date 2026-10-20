"""H3 still size control and a 16-bit PNG master exporter. Generation is upstream."""
from pathlib import Path
import re
from workbench.h3_still_tools import ASPECTS, still_dimensions, write_rgb16_png


class WBH3StillSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'target_megapixels': ('FLOAT', {'default': 8.4, 'min': 0.25, 'max': 16.777216, 'step': 0.1}),
            'aspect_ratio': (list(ASPECTS), {'default': 'source'}),
        }, 'optional': {'reference': ('IMAGE',)}}
    RETURN_TYPES = ('INT', 'INT', 'STRING')
    RETURN_NAMES = ('width', 'height', 'size_report')
    FUNCTION = 'size'
    CATEGORY = 'Workbench/H3 Still HQ'

    def size(self, target_megapixels, aspect_ratio, reference=None):
        source_size = None
        if reference is not None:
            if reference.ndim != 4 or reference.shape[0] != 1 or reference.shape[-1] < 3:
                raise ValueError('Connect one RGB reference image, not an image batch.')
            source_size = (int(reference.shape[2]), int(reference.shape[1]))
        return still_dimensions(target_megapixels, aspect_ratio, source_size)


class WBH3SaveStillMaster:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'filename_prefix': ('STRING', {'default': 'h3_still/H3_Identity_RGB16'})}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('images',)
    FUNCTION = 'save'
    CATEGORY = 'Workbench/H3 Still HQ'
    OUTPUT_NODE = True

    def save(self, images, filename_prefix):
        import folder_paths
        if images.ndim != 4 or images.shape[0] != 1 or images.shape[-1] != 3:
            raise ValueError('Expected exactly one RGB still image.')
        if not re.fullmatch(r'[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*', filename_prefix):
            raise ValueError('Prefix must use safe relative folder names, letters, numbers, underscores and hyphens.')
        root = Path(folder_paths.get_output_directory()).resolve()
        folder, stem, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, str(root), images.shape[2], images.shape[1])
        folder = Path(folder)
        if not folder.resolve().is_relative_to(root):
            raise ValueError('Output folder escapes ComfyUI output directory.')
        for _ in range(100):
            filename = f'{stem}_{counter:05}_RGB16.png'
            target = folder / filename
            try:
                write_rgb16_png(images[0], target)
                break
            except FileExistsError:
                counter += 1
        else:
            raise ValueError('Could not reserve a unique output filename.')
        return {'ui': {'images': [{'filename': filename, 'subfolder': subfolder, 'type': 'output'}]}, 'result': (images,)}


NODE_CLASS_MAPPINGS = {'WBH3StillSize': WBH3StillSize, 'WBH3SaveStillMaster': WBH3SaveStillMaster}
NODE_DISPLAY_NAME_MAPPINGS = {'WBH3StillSize': 'H3 Still: shared output size', 'WBH3SaveStillMaster': 'H3 Still: save RGB16 PNG master'}
