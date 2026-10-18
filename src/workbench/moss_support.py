"""Build-time compatibility checks for the pinned Qwen + MOSS ComfyUI source."""
from __future__ import annotations
from pathlib import Path
from .moss_catalog import COMFY_COMMIT

REQUIRED_REQUIREMENTS = (
    'comfyui-frontend-package==1.55.17',
    'comfy-kitchen==0.2.38',
    'comfy-aimdo==0.5.6',
)


def verify_core(comfy_root: Path, commit: str) -> None:
    """Do not rewrite the new upstream aimdo pin as if it were the old version."""
    if commit != COMFY_COMMIT:
        raise ValueError('Unexpected ComfyUI version for the MOSS-enabled Qwen build.')
    root = Path(comfy_root)
    text = (root / 'requirements.txt').read_text(encoding='utf-8')
    lines = text.splitlines()
    for pin in REQUIRED_REQUIREMENTS:
        if lines.count(pin) != 1:
            raise ValueError('Pinned MOSS build dependency differs: ' + pin)
    required_files = ('comfy/ldm/moss_image/model.py', 'comfy/text_encoders/moss_image.py',
                      'comfy_extras/nodes_moss_image.py', 'comfy_extras/nodes_qwen.py',
                      'comfy_extras/nodes_sam3.py')
    for name in required_files:
        if not (root / name).is_file():
            raise ValueError('Missing native MOSS/Qwen/SAM support: ' + name)
    if 'moss_image' not in (root / 'nodes.py').read_text(encoding='utf-8'):
        raise ValueError('Native CLIP loader does not expose moss_image.')
    if '--disable-comfy-compiler' not in (root / 'comfy/cli_args.py').read_text(encoding='utf-8'):
        raise ValueError('Qwen compiler-disable compatibility option is missing.')
