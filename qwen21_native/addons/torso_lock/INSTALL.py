#!/usr/bin/env python3
"""Install Torso Lock alongside an existing Qwen 2.1 ComfyUI. No network/models."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent
PACK='qwen21_torso_tools'
FOLDER='Qwen21 Torso Lock v1.0.0'


def discover():
    """Prefer the actual running ComfyUI process over assumptions about /workspace."""
    found=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args=proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            main=next((x for x in args if x=='main.py' or x.endswith('/main.py')),None)
            if not main:continue
            cwd=proc.joinpath('cwd').resolve()
            comfy=(cwd/Path(main)).resolve().parent
            if not (comfy/'comfy_extras/nodes_qwen.py').exists():continue
            user=None
            for i,item in enumerate(args):
                if item=='--user-directory' and i+1<len(args):user=Path(args[i+1])
                if item.startswith('--user-directory='):user=Path(item.split('=',1)[1])
            if user is None:user=comfy/'user'
            if not user.is_absolute():user=cwd/user
            pair=(comfy,user.resolve())
            if pair not in found:found.append(pair)
        except (OSError,UnicodeError,StopIteration):continue
    return found


def install(comfy,user):
    comfy=Path(comfy).resolve();user=Path(user).resolve()
    native=comfy/'comfy_extras/nodes_qwen.py'
    if not native.is_file() or 'class TextEncodeQwenImage21' not in native.read_text():
        raise RuntimeError('Native Qwen 2.1 is absent. Use the existing dedicated Qwen image; no core update was attempted.')
    aus=comfy/'custom_nodes/ComfyUI-AusBoss/nodes/node_inpaint_crop_stitch.py'
    if not aus.is_file() or 'AUSBOSS_NODES_StitchInpaint' not in aus.read_text():
        raise RuntimeError('The existing AusBoss crop/stitch package is required. No extra install was attempted.')
    # The addon only uses SciPy/Torch/Pillow/NumPy already required by ComfyUI.
    src=ROOT/'custom_nodes'/PACK;dst=comfy/'custom_nodes'/PACK
    if dst.exists():
        source_hash=hashlib.sha256((src/'__init__.py').read_bytes()).hexdigest()
        target=dst/'__init__.py'
        target_hash=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else ''
        if source_hash!=target_hash:
            backup=user/'qwen21_torso_backups'/str(time.time_ns())/PACK
            backup.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(dst,backup)
    shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    out=user/'default/workflows'/FOLDER;out.mkdir(parents=True,exist_ok=True)
    count=0
    for row in json.loads((ROOT/'config/catalog.json').read_text())['workflows']:
        dest=out/row['file']
        if not dest.exists():shutil.copy2(ROOT/'workflows'/row['file'],dest);count+=1
    print(f'Installed {PACK} into {comfy}/custom_nodes')
    print(f'Added {count} workflows to {out}; existing user-edited workflows were preserved.')
    print('No weights downloaded, packages changed, old helpers overwritten, or server stopped.')
    print('Restart ComfyUI through its existing process/service control, then refresh the browser. Do not terminate a stateless Pod to reload nodes.')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy-dir',type=Path);p.add_argument('--user-dir',type=Path)
    args=p.parse_args()
    if args.comfy_dir and args.user_dir:
        install(args.comfy_dir,args.user_dir);return
    found=discover()
    if len(found)==1:
        comfy,user=found[0];install(args.comfy_dir or comfy,args.user_dir or user);return
    if len(found)>1:
        raise RuntimeError('Multiple ComfyUI servers found. Pass --comfy-dir and --user-dir explicitly.')
    # These kits expose DATA_ROOT when started from their supplied entrypoint.
    env_root=os.getenv('DATA_ROOT')
    available=[x for x in (Path('/opt/ComfyUI'),Path('/workspace/ComfyUI')) if (x/'comfy_extras/nodes_qwen.py').is_file()]
    if len(available)==1 and env_root:
        install(args.comfy_dir or available[0],args.user_dir or Path(env_root)/'user');return
    raise RuntimeError('Could not identify the real ComfyUI user directory. Pass --comfy-dir /opt/ComfyUI --user-dir "$DATA_ROOT/user" using the actual paths.')

if __name__=='__main__':
    try:main()
    except (RuntimeError,OSError) as exc:
        print('INSTALL FAILED: '+str(exc),file=sys.stderr);sys.exit(1)
