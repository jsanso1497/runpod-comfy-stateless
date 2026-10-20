"""Build approved-profile software only. Never download model weights here."""
from __future__ import annotations
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/opt/workbench')
COMFY = Path('/opt/ComfyUI')
from workbench.sam_local_edit_profile import COMFY_COMMIT, LOCAL_NODES


def run(*args):
    subprocess.run(args, check=True)


def main():
    if COMFY.exists():
        raise RuntimeError('Expected an empty ComfyUI install destination.')
    COMFY.mkdir(parents=True)
    run('git', 'init', '-q', str(COMFY))
    run('git', '-C', str(COMFY), 'remote', 'add', 'origin', 'https://github.com/Comfy-Org/ComfyUI.git')
    run('git', '-C', str(COMFY), 'fetch', '--depth', '1', 'origin', COMFY_COMMIT)
    run('git', '-C', str(COMFY), 'checkout', '--detach', 'FETCH_HEAD')
    actual = subprocess.check_output(['git', '-C', str(COMFY), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMFY_COMMIT:
        raise ValueError('Unexpected ComfyUI source revision.')
    shutil.rmtree(COMFY / '.git')
    requirements = (COMFY / 'requirements.txt').read_text()
    for pin in ('comfyui-frontend-package==1.55.17', 'comfy-kitchen==0.2.38', 'comfy-aimdo==0.5.6'):
        if requirements.splitlines().count(pin) != 1:
            raise ValueError('Review the pinned dependency: ' + pin)
    for name in ('comfy_extras/nodes_sam3.py', 'comfy_extras/nodes_qwen.py', 'comfy_extras/nodes_compositing.py'):
        if not (COMFY / name).is_file():
            raise ValueError('Missing core support: ' + name)
    if '--disable-comfy-compiler' not in (COMFY / 'comfy/cli_args.py').read_text():
        raise ValueError('Missing compiler-disable option.')
    constraints = Path('/tmp/local-edit-torch-constraints.txt')
    pins = ['comfy-aimdo==0.5.6']
    for package in ('torch', 'torchvision', 'torchaudio'):
        try:
            pins.append(package + '==' + metadata.version(package))
        except metadata.PackageNotFoundError:
            pass
    constraints.write_text('\n'.join(pins) + '\n')
    run(sys.executable, '-m', 'pip', 'install', '--constraint', str(constraints),
        '-r', str(COMFY / 'requirements.txt'), '-r', str(ROOT / 'requirements.txt'))
    for name in LOCAL_NODES:
        source = ROOT / 'src/nodes' / name
        if not (source / '__init__.py').is_file():
            raise ValueError('Missing required local node module: ' + name)
        (COMFY / 'custom_nodes' / ('Workbench_' + name)).symlink_to(source, target_is_directory=True)
    run(sys.executable, '-m', 'pip', 'check')
    (ROOT / 'BUILD_WORKSPACE').write_text('qwen\n')
    (ROOT / 'build-provenance.json').write_text(json.dumps({
        'workspace': 'qwen', 'profile': 'approved-local-edit-1', 'comfy_commit': actual,
        'local_nodes': LOCAL_NODES, 'external_nodes': {}, 'model_downloads_at_build': False,
        'live_gpu_validated': False,
    }, indent=2) + '\n')
    site = ROOT / 'site'
    site.mkdir(exist_ok=True)
    (site / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>SAM-Guided Local Edit</title><body>'
        '<h1>SAM-Guided Local Edit</h1><p>Only approved workflow: QLOCAL.</p>'
        '<p>Only Qwen BF16 and SAM models are prepared. No prompt enhancer.</p>'
        '<p><a href="/">Open ComfyUI</a>, then use the Approved sidebar or '
        'Workflows &gt; 00_Approved.</p></body></html>')


if __name__ == '__main__':
    main()
