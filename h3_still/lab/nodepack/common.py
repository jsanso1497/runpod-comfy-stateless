"""Local H3 Studio Lab utilities. No network access or automatic model downloads."""
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

VERSION = '2026-11-08.1'
CATEGORY = 'H3 Studio Lab'
NONE = '(none)'
ROOT = Path(os.environ.get('H3_STILL_DATA_ROOT', '/workspace/h3-still'))
LAB = Path(os.environ.get('H3_LAB_PACKAGE', '/opt/h3-still/lab'))
PROVIDERS = {}


def outputs(value):
    if hasattr(value, 'result'):
        value = value.result
    if isinstance(value, dict):
        value = value['result']
    return tuple(value)


def invoke(cls, **values):
    if hasattr(cls, 'execute'):
        return outputs(cls.execute(**values))
    return outputs(getattr(cls(), cls.FUNCTION)(**values))


def registry(name):
    import nodes
    if name not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError('Required node is unavailable: %s. Use the Studio Lab image, not the old image.' % name)
    return nodes.NODE_CLASS_MAPPINGS[name]


def provider(name):
    if name in PROVIDERS:
        return PROVIDERS[name]
    root = LAB / 'vendor' / name
    if name == 'spark':
        root = ROOT / 'lab' / 'providers' / 'spark' / 'ComfyUI-Spark-H3'
    entry = root / '__init__.py'
    if not entry.is_file():
        raise RuntimeError('Optional provider %s is not installed. See H3_STUDIO_LAB_START_HERE.md.' % name)
    module_name = '_h3lab_vendor_' + name
    spec = importlib.util.spec_from_file_location(module_name, entry, submodule_search_locations=[str(root)])
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load provider: ' + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for key in list(sys.modules):
            if key == module_name or key.startswith(module_name + '.'):
                sys.modules.pop(key, None)
        raise
    if not getattr(module, 'NODE_CLASS_MAPPINGS', None):
        raise RuntimeError('Provider does not expose its expected node interface: ' + name)
    PROVIDERS[name] = module
    return module


def provider_node(pack, name):
    mapping = provider(pack).NODE_CLASS_MAPPINGS
    if name not in mapping:
        raise RuntimeError('Pinned %s provider is missing %s. Do not substitute unreviewed versions.' % (pack, name))
    return mapping[name]


def required_defaults(cls, overrides):
    """Respect an upstream legacy node's real required inputs; never invent missing data."""
    schema = cls.INPUT_TYPES()
    values = {}
    for name, definition in schema.get('required', {}).items():
        if name in overrides:
            values[name] = overrides[name]
            continue
        kind = definition[0]
        options = definition[1] if len(definition) > 1 else {}
        if 'default' in options:
            values[name] = options['default']
        elif isinstance(kind, (list, tuple)) and kind:
            values[name] = kind[0]
        else:
            raise RuntimeError('No reviewed default for upstream input: ' + name)
    accepted = set(schema.get('required', {})) | set(schema.get('optional', {}))
    unknown = set(overrides) - accepted
    if unknown:
        raise RuntimeError('Upstream contract changed; unknown inputs: ' + ', '.join(sorted(unknown)))
    values.update(overrides)
    return values


def short_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name):
        raise ValueError('Use a name containing only letters, numbers, underscores or hyphens, maximum 64 characters.')
    return name


def contained(root, relative):
    part = Path(relative)
    if not relative or part.is_absolute() or '..' in part.parts or '\\' in relative:
        raise ValueError('Unsafe relative path.')
    target = (root / part).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Path leaves its allowed folder.')
    return target


def crop_bounds(width, height, left, top, right, bottom):
    vals = (left, top, right, bottom)
    if any(not math.isfinite(v) or not 0 <= v < 100 for v in vals):
        raise ValueError('Crop margins must be between 0 and less than 100 percent.')
    if left + right >= 100 or top + bottom >= 100:
        raise ValueError('Crop margins remove the entire photograph.')
    x0, y0 = int(width * left / 100), int(height * top / 100)
    x1, y1 = max(x0 + 1, int(width * (1-right/100))), max(y0 + 1, int(height * (1-bottom/100)))
    return x0, y0, x1, y1


def dimensions(width, height, mp, down_only=False):
    if width <= 0 or height <= 0 or not math.isfinite(mp) or not .25 <= mp <= 8:
        raise ValueError('Invalid image dimensions or megapixel value.')
    scale = math.sqrt(mp * 1024 * 1024 / (width * height))
    if down_only:
        scale = min(1., scale)
    scale = min(scale, 4096/max(width, height))
    return tuple(max(32, round(v*scale/32)*32) for v in (width, height))


def resize(image, width, height):
    import comfy.utils
    if image.shape[2] == width and image.shape[1] == height:
        return image
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, 'lanczos', 'disabled').movedim(1, -1)


def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def text_ui(text, result=()):
    return {'ui': {'text': [text]}, 'result': result}


def run_filename(prefix='run'):
    from uuid import uuid4
    return time.strftime('%Y%m%d-%H%M%S') + '-' + prefix + '-' + uuid4().hex[:8]
