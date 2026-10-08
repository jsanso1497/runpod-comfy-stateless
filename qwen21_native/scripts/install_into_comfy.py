#!/usr/bin/env python3
"""Install this suite alongside an existing ComfyUI. No server restart or core update."""
from __future__ import annotations
import argparse,json,os,shutil,sys,time
from pathlib import Path
from validate_workflows import selected_catalog
ROOT=Path(__file__).resolve().parents[1]


def install(comfy:Path,user:Path,profile:str='identity',include_bfs:bool=False):
    comfy=comfy.resolve();user=user.resolve()
    qwen=comfy/'comfy_extras/nodes_qwen.py'
    if not qwen.is_file() or 'class TextEncodeQwenImage21' not in qwen.read_text():
        raise RuntimeError('This ComfyUI lacks native Qwen 2.1. Use the supplied pinned Docker build; no automatic core upgrade was attempted.')
    ausboss=comfy/'custom_nodes/ComfyUI-AusBoss/nodes/node_inpaint_crop_stitch.py'
    if not ausboss.is_file() or 'AUSBOSS_NODES_StitchInpaint' not in ausboss.read_text():
        raise RuntimeError('Required AusBoss crop/stitch nodes are absent. Use the supplied Docker build or install the pinned AusBoss revision first.')
    if profile=='upscale' and not (comfy/'custom_nodes/ComfyUI-SeedVR2_VideoUpscaler').is_dir():
        raise RuntimeError('Upscale profile requires SeedVR2 already installed. Install native workflows with --profile identity or use the upscale Docker image.')
    src=ROOT/'custom_nodes/qwen21_native_tools';dst=comfy/'custom_nodes/qwen21_native_tools'
    if dst.exists():
        backup=user/'qwen21_native_backups'/str(time.time_ns())/'qwen21_native_tools'
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(dst,backup)
    shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    out=user/'default/workflows/Qwen21 Native v1.1.0';out.mkdir(parents=True,exist_ok=True)
    copied=0
    for row in selected_catalog(profile,include_bfs):
        path=out/row['file']
        if not path.exists():shutil.copy2(ROOT/'workflows'/row['file'],path);copied+=1
    print(f'Installed native helper at {dst}; added {copied} workflow files to {out}. Existing saved workflows were preserved.')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy-dir',type=Path,default=Path('/opt/ComfyUI'))
    p.add_argument('--user-dir',type=Path,required=True,help='Actual ComfyUI --user-directory, NOT its default subfolder')
    p.add_argument('--profile',choices=['identity','upscale'],default='identity')
    p.add_argument('--include-bfs',action='store_true');args=p.parse_args()
    install(args.comfy_dir,args.user_dir,args.profile,args.include_bfs)
    print('No models downloaded or core code changed. Restart ComfyUI to load the new helper. Existing models are reused through your current model paths.')

if __name__=='__main__':
    try:main()
    except (OSError,RuntimeError) as exc:print('INSTALL FAILED: '+str(exc),file=sys.stderr);sys.exit(1)
