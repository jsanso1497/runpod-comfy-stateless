#!/usr/bin/env python3
"""Add Krea to any of the three existing images without changing H3 recipes.

Node source is bundled at Docker build. Large weights are an explicit runtime
opt-in. No model files, cloud inference, private URLs or trained subject LoRAs
are added to the Docker layers.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
VERSION = '1.0.0'
EXPECTED_COMFY = '65787d668397d230bf5839d69a0a7239e2dad378'


def flag(name, default='0'):
    value = os.environ.get(name, default)
    if value not in ('0','1'):
        raise ValueError(name+' must be 0 or 1.')
    return value == '1'


def install_workflows(home, upscale=False, source=HERE):
    source, home = Path(source), Path(home)
    target = home/'user/default/workflows'
    target.mkdir(parents=True, exist_ok=True)
    installed=[]
    for file in sorted((source/'workflows').glob('*.json')):
        if '04_Upscale' in file.name and not upscale:
            continue
        out = target/file.name
        if not out.exists():
            # Avoid overwriting an edited workflow or a concurrent install.
            try:
                with out.open('x', encoding='utf-8') as handle:
                    handle.write(file.read_text())
            except FileExistsError:
                pass  # Another installer won; never replace its/user's workflow.
        print('KREA IDENTITY WORKFLOW: '+out.name, flush=True)
        installed.append(out)
    return installed


def build(home, scripts, seed_source, constraints):
    home, scripts = Path(home), Path(scripts)
    pin_file = Path('/opt/runpod-comfy/comfy-build-ref.txt')
    if pin_file.exists() and pin_file.read_text().strip() != EXPECTED_COMFY:
        raise ValueError('Krea module and inherited ComfyUI revisions differ. Rebuild against the pinned base.')
    subprocess.run([sys.executable, str(scripts/'install_nodes.py'), '--manifest',
                    str(HERE/'config/custom_nodes.json'), '--comfy-home',str(home),
                    '--constraints',str(constraints)], check=True)
    dest=home/'custom_nodes/ComfyUI-KreaIdentity'
    shutil.copytree(HERE/'node',dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Both portrait images deliberately clear inherited nodes. Restore ONLY this
    # already-bundled upscaler, not unrelated H3/RefMod/utility packs.
    if not Path(seed_source).is_dir():
        raise FileNotFoundError('The pinned base image is missing /opt/seedvr2.')
    shutil.copytree(seed_source,home/'custom_nodes/ComfyUI-SeedVR2_VideoUpscaler',
                    dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE/'tests'),'-p','test_*.py'],check=True)
    print('KREA IDENTITY SOURCE INSTALLED. Weights are deferred to opt-in Pod startup.',flush=True)


def runtime(home, scripts):
    enabled, upscale = flag('ENABLE_KREA_IDENTITY'), flag('KREA_IDENTITY_UPSCALE')
    if upscale and not enabled:
        raise ValueError('KREA_IDENTITY_UPSCALE=1 also requires ENABLE_KREA_IDENTITY=1.')
    if not enabled:
        print('KREA IDENTITY disabled. Enable with ENABLE_KREA_IDENTITY=1; H3 is unchanged.',flush=True)
        return []
    home, scripts=Path(home),Path(scripts)
    for name in ('ComfyUI-KreaIdentity','comfyui-krea2edit','Rebalance-Pack'):
        if not (home/'custom_nodes'/name).is_dir():
            raise RuntimeError('Missing bundled node '+name+'. Build the updated image first.')
    spec=importlib.util.spec_from_file_location('_krea_identity_catalog',scripts/'catalog.py')
    catalog=importlib.util.module_from_spec(spec);spec.loader.exec_module(catalog)
    profiles={'krea2'} | ({'seedvr2'} if upscale else set())
    # Same verified hashes and destinations as the general stack: reused weights
    # are hash-checked instead of downloaded under a second filename.
    catalog.run_downloads(HERE/'config',home,profiles)
    installed=install_workflows(home,upscale)
    print('KREA IDENTITY READY: Turbo BF16 + Identity Edit v1.2 + optional screenshot rebalancer.',flush=True)
    return installed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('build','runtime'))
    p.add_argument('--comfy-home',type=Path,default=Path('/workspace/ComfyUI'))
    p.add_argument('--scripts',type=Path,default=Path('/opt/runpod-comfy/scripts'))
    p.add_argument('--seed-source',type=Path,default=Path('/opt/seedvr2'))
    p.add_argument('--constraints',type=Path,default=Path('/opt/video-restore/constraints.txt'))
    a=p.parse_args()
    if a.action=='build':build(a.comfy_home,a.scripts,a.seed_source,a.constraints)
    else:runtime(a.comfy_home,a.scripts)

if __name__=='__main__':main()
