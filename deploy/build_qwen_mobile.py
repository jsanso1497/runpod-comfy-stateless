#!/usr/bin/env python3
"""Build-time Qwen app contract and isolated Jupyter installation; no model downloads."""
from __future__ import annotations
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path('/opt/workbench')


def run(*args):
    subprocess.run(list(args), check=True)


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: build_qwen_mobile.py WORKSPACE')
    if sys.argv[1] != 'qwen':
        return
    from workbench.qwen_mobile import COMFY_COMMIT, FRONTEND_VERSION, WORKFLOW_PATH, make_workflow
    installed = version('comfyui-frontend-package')
    if installed != FRONTEND_VERSION:
        raise RuntimeError('Review QAPP against the new ComfyUI frontend before changing its pin: ' + installed)
    provenance = json.loads((ROOT / 'build-provenance.json').read_text())
    if provenance.get('comfy_commit') != COMFY_COMMIT:
        raise RuntimeError('Review QAPP native node schemas before changing the ComfyUI pin.')
    path = ROOT / WORKFLOW_PATH
    if json.loads(path.read_text()) != make_workflow():
        raise RuntimeError('QAPP checked-in JSON differs from its tested generator.')
    venv_dir = Path('/opt/jupyter-venv')
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = str(venv_dir / 'bin/python')
    run(python, '-m', 'pip', 'install', '--no-cache-dir', '-r', str(ROOT / 'deploy/jupyter-requirements.txt'))
    run(python, '-m', 'pip', 'check')
    packages = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True).splitlines()
    (ROOT / 'qwen-mobile-build-provenance.json').write_text(json.dumps({
        'version': '1.0.0', 'comfy_commit': COMFY_COMMIT, 'frontend': installed,
        'workflow_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'jupyter_packages': packages, 'gpu_tested': False, 'browser_tested': False,
    }, indent=2) + '\n')
    print('Qwen mobile graph and isolated Jupyter installed. GPU/browser testing is still required.')


if __name__ == '__main__':
    main()
