#!/usr/bin/env python3
"""Bundle a native-H3 input adapter; download full BF16 weights only by opt-in."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
VERSION='1.2.0'
EXPECTED_COMFY='65787d668397d230bf5839d69a0a7239e2dad378'


def enabled():
    val=os.environ.get('ENABLE_H3_MEDIA','0')
    if val not in ('0','1'):raise ValueError('ENABLE_H3_MEDIA must be 0 or 1.')
    return val=='1'


def install_workflows(home,source=HERE):
    target=Path(home)/'user/default/workflows';target.mkdir(parents=True,exist_ok=True)
    result=[]
    for p in sorted((Path(source)/'workflows').glob('*.json')):
        out=target/p.name
        try:
            with out.open('x',encoding='utf-8') as f:f.write(p.read_text())
        except FileExistsError:pass
        result.append(out)
        print('H3 MEDIA WORKFLOW: '+out.name,flush=True)
    return result


def build(home):
    pin=Path('/opt/runpod-comfy/comfy-build-ref.txt')
    if pin.exists() and pin.read_text().strip()!=EXPECTED_COMFY:
        raise ValueError('H3 Media requires the repository-pinned native H3 ComfyUI build.')
    for binary in ('ffmpeg','ffprobe'):
        if not shutil.which(binary):raise RuntimeError('Missing existing media dependency: '+binary)
    shutil.copytree(HERE/'node',Path(home)/'custom_nodes/ComfyUI-H3MediaHQ',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE/'tests'),'-p','test_*.py'],check=True)
    print('H3 MEDIA NODE SOURCE INSTALLED. No weights downloaded.',flush=True)


def runtime(home,scripts):
    if not enabled():
        print('H3 MEDIA disabled; opt in with ENABLE_H3_MEDIA=1. Existing Krea/H3 workflows unchanged.',flush=True)
        return []
    home=Path(home)
    if not (home/'custom_nodes/ComfyUI-H3MediaHQ').is_dir():
        raise RuntimeError('Missing H3 Media node bundle. Rebuild the image from the updated source.')
    spec=importlib.util.spec_from_file_location('_h3_media_catalog',Path(scripts)/'catalog.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.run_downloads(HERE/'config',home,{'h3'})
    return install_workflows(home)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','runtime'])
    p.add_argument('--comfy-home',type=Path,default=Path('/workspace/ComfyUI'))
    p.add_argument('--scripts',type=Path,default=Path('/opt/runpod-comfy/scripts'))
    a=p.parse_args()
    if a.action=='build':build(a.comfy_home)
    else:runtime(a.comfy_home,a.scripts)

if __name__=='__main__':main()
