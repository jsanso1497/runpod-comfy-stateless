#!/usr/bin/env python3
"""Fetch reviewed source commits, not moving branches or auto-updates."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def checkout(url: str, commit: str, dest: Path) -> None:
    if dest.exists():
        raise RuntimeError(f'Refusing to replace an existing checkout: {dest}')
    dest.mkdir(parents=True)
    for command in (['git','init',str(dest)], ['git','-C',str(dest),'remote','add','origin',url],
                    ['git','-C',str(dest),'fetch','--depth','1','origin',commit],
                    ['git','-C',str(dest),'checkout','--detach','FETCH_HEAD']):
        subprocess.run(command, check=True)
    actual = subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'], text=True).strip()
    if actual != commit:
        raise RuntimeError(f'Commit mismatch for {dest}: {actual}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--comfy-dir',type=Path,default=Path('/opt/ComfyUI'))
    p.add_argument('--profile',choices=['identity','upscale'],default='identity')
    args=p.parse_args()
    pins=json.loads((ROOT/'config/upstream.json').read_text())
    for key,dest in [('comfyui',args.comfy_dir),('ausboss',args.comfy_dir/'custom_nodes/ComfyUI-AusBoss')]:
        x=pins[key];checkout(x['url'],x['commit'],dest)
    if args.profile=='upscale':
        x=pins['seedvr2'];checkout(x['url'],x['commit'],args.comfy_dir/'custom_nodes/ComfyUI-SeedVR2_VideoUpscaler')
if __name__=='__main__':main()
