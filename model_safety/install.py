#!/usr/bin/env python3
"""Install protected model-loader source without model downloads."""
import argparse
from pathlib import Path
import shutil


def install(home):
    source = Path(__file__).resolve().parent / 'node'
    dest = Path(home) / 'custom_nodes/ComfyUI-ModelSafety'
    shutil.copytree(source, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    print('MODEL SAFETY 1.5.3: protected KREA / H3 loaders installed.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--comfy-home', required=True)
    install(p.parse_args().comfy_home)
