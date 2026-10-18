"""Fetch and preserve the official MOSS template. No image generation or weights.

The source Git blob ID is checked BEFORE parsing. Both the archived source and
runtime workflow are exact copies: no model, prompt, sampler or subgraph edits.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen
from .moss_catalog import TASK, WORKFLOW, I2I_WORKFLOW, MODEL_FILES, COMFY_COMMIT

TEMPLATE_REVISION = '7d8ed32f0bf6b8d52129976535422d6f518fe463'
TEMPLATE_PATH = 'templates/image_moss_image_t2i.json'
TEMPLATE_BLOB = '7f00641fd40a3c543f461b0d956c4a7cebe0975b'
TEMPLATE_URL = ('https://raw.githubusercontent.com/Comfy-Org/workflow_templates/' +
                TEMPLATE_REVISION + '/' + TEMPLATE_PATH)
MAX_BYTES = 8 * 1024 * 1024
DEFAULTS = {
    'switch': True, 'steps': 25, 'cfg': 2.5, 'sampler_name': 'er_sde',
    'scheduler': 'simple', 'switch_1': False,
    'unet_name': 'moss_image_sft_int8_convrot.safetensors',
    'clip_name': 'qwen3.5_9b_w4a8_gptq.safetensors',
    'vae_name': 'flux2-vae_small.safetensors',
}


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def verify_template(raw: bytes) -> dict:
    if len(raw) > MAX_BYTES or git_blob(raw) != TEMPLATE_BLOB:
        raise ValueError('MOSS official template checksum mismatch. No workflow was written.')
    graph = json.loads(raw)
    if not isinstance(graph, dict) or not isinstance(graph.get('nodes'), list):
        raise ValueError('MOSS source is not a ComfyUI frontend workflow.')
    matches = [n for n in graph['nodes'] if
               'unet_name' in n.get('widgets_values_named', {})]
    if len(matches) != 1:
        raise ValueError('Expected one stock MOSS text-to-image subgraph.')
    values = matches[0]['widgets_values_named']
    for name, expected in DEFAULTS.items():
        if values.get(name) != expected:
            raise ValueError('Unexpected stock MOSS default: ' + name)
    # The immutable blob check also covers nested edges, system prompt, node IDs
    # and positional widgets. Do not build a fixture in place of the real source.
    return graph


def _download() -> bytes:
    for attempt in range(3):
        try:
            request = Request(TEMPLATE_URL, headers={'User-Agent': 'Workbench-MOSS-Stock/1.0'})
            with urlopen(request, timeout=60) as response:
                if response.geturl() != TEMPLATE_URL:
                    raise ValueError('Unexpected redirect fetching the pinned MOSS workflow.')
                raw = response.read(MAX_BYTES + 1)
            verify_template(raw)
            return raw
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError('Pinned MOSS workflow could not be downloaded.')


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Refusing symlinked MOSS workflow output.')
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.moss-')
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(raw)
        os.chmod(name, 0o644)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def build(root: Path, offline: bool = False) -> dict:
    root = Path(root)
    if not (root / 'catalog/tasks.moss.json').is_file():
        return {'task': TASK, 'installed': False}
    archive = root / 'vendor/moss-stock' / (TEMPLATE_REVISION + '.json')
    if archive.is_symlink():
        raise ValueError('Refusing a symlinked MOSS source snapshot.')
    if archive.is_file():
        raw = archive.read_bytes()
    elif offline:
        raise FileNotFoundError('MOSS source snapshot missing. Build once with network access.')
    else:
        raw = _download()
    graph = verify_template(raw)
    from .moss_i2i_workflows import make_img2img, encode_graph, provenance
    image_to_image = make_img2img(graph)
    _write(archive, raw)
    _write(root / WORKFLOW, raw)
    _write(root / I2I_WORKFLOW, encode_graph(image_to_image))
    report = {
        'task': TASK, 'template_url': TEMPLATE_URL, 'template_revision': TEMPLATE_REVISION,
        'source_git_blob': TEMPLATE_BLOB, 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'workflow': WORKFLOW, 'copied_byte_for_byte': True, 'model_files': sorted(MODEL_FILES),
        'qwen_comfy_commit': COMFY_COMMIT, 'gpu_tested': False,
        'image_to_image': provenance(image_to_image),
    }
    _write(archive.parent / 'provenance.json', (json.dumps(report, indent=2) + '\n').encode())
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.offline), indent=2))
