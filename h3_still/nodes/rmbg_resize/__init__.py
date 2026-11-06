"""Register the developer's actual RMBG resize class without other RMBG models.

The entire unmodified pinned GPL-3.0 source and license remain at
/opt/vendor/ComfyUI-RMBG. No background-removal model is loaded by this shim.
"""
from pathlib import Path
import importlib.util
import sys

SOURCE = Path('/opt/vendor/ComfyUI-RMBG/py')
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
_spec = importlib.util.spec_from_file_location(
    'h3_still_official_rmbg_image_mask_tools', SOURCE / 'AILab_ImageMaskTools.py')
if _spec is None or _spec.loader is None:
    raise ImportError('Pinned RMBG image utilities are missing.')
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
NODE_CLASS_MAPPINGS = {'AILab_ImageResize': _module.AILab_ImageResize}
NODE_DISPLAY_NAME_MAPPINGS = {'AILab_ImageResize': 'Image Resize (RMBG)'}
