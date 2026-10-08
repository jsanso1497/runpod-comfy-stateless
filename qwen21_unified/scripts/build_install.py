#!/usr/bin/env python3
"""Build-time integration: AusBoss, Native helpers and Torso Lock in Photo ComfyUI.

Does not download model weights, start ComfyUI, or change CUDA/Torch.
"""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path


def run(cmd:list[str]):
    print('+ '+' '.join(map(str,cmd)),flush=True)
    subprocess.run([str(a) for a in cmd],check=True)


def checkout(url:str,revision:str,target:Path):
    if target.is_dir():
        current=subprocess.run(['git','-C',str(target),'rev-parse','HEAD'],text=True,capture_output=True)
        if current.returncode==0 and current.stdout.strip()==revision:return
        raise RuntimeError(f'Existing custom-node directory at {target} is not the reviewed AusBoss revision. Refusing to overwrite it.')
    target.parent.mkdir(parents=True,exist_ok=True)
    run(['git','init',str(target)])
    run(['git','-C',str(target),'remote','add','origin',url])
    run(['git','-C',str(target),'fetch','--depth','1','origin',revision])
    run(['git','-C',str(target),'checkout','--detach','FETCH_HEAD'])
    got=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
    if got!=revision:raise RuntimeError('AusBoss SHA mismatch')


def install_build(comfy:Path,native:Path,torch_constraints:Path):
    nodes=comfy/'comfy_extras/nodes_qwen.py'
    if not nodes.is_file() or 'class TextEncodeQwenImage21' not in nodes.read_text():
        raise RuntimeError('Photo ComfyUI revision lacks native Qwen Image 2.1. Refusing to build an image with broken native graphs.')
    pins=json.loads((native/'config/upstream.json').read_text())
    ausboss=comfy/'custom_nodes/ComfyUI-AusBoss'
    checkout(pins['ausboss']['url'],pins['ausboss']['commit'],ausboss)
    req=ausboss/'requirements.txt'
    if req.is_file():
        run([sys.executable,'-m','pip','install','--no-cache-dir','--constraint',str(torch_constraints),'-r',str(req)])
    # Install custom-node Python without overwriting the Qwen Photo node package.
    for rel in ('qwen21_native_tools','qwen21_torso_tools'):
        src=(native/'custom_nodes'/rel) if rel=='qwen21_native_tools' else (native/'addons/torso_lock/custom_nodes'/rel)
        dest=comfy/'custom_nodes'/rel
        if not (src/'__init__.py').is_file():raise RuntimeError(f'Missing bundled helper: {src}')
        if dest.exists():
            if (dest/'__init__.py').read_bytes() != (src/'__init__.py').read_bytes():
                raise RuntimeError(f'Existing helper differs: {dest}. Do not silently replace a third-party node.')
        else:shutil.copytree(src,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    user=comfy/'user/default/workflows'
    for label,src in [('Qwen21 Native v1.1.0',native/'workflows'),('Qwen21 Torso Lock v1.0.0',native/'addons/torso_lock/workflows')]:
        dest=user/label;dest.mkdir(parents=True,exist_ok=True)
        for f in src.glob('*.json'):
            out=dest/f.name
            if out.exists() and out.read_bytes()!=f.read_bytes():raise RuntimeError(f'Differs: {out}')
            if not out.exists():shutil.copy2(f,out)
    run([sys.executable,str(native/'scripts/validate_workflows.py'),'--profile','upscale','--include-bfs'])
    run([sys.executable,str(native/'addons/torso_lock/scripts/validate_workflows.py')])
    run([sys.executable,'-m','pip','check'])
    print('BUILD ADD-ON READY: native Qwen + AusBoss + Torso custom nodes; Photo SAM core retained. Weights remain runtime downloads.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--comfy',type=Path,default=Path('/opt/ComfyUI'))
    p.add_argument('--native',type=Path,default=Path('/opt/qwen21_native'))
    p.add_argument('--constraints',type=Path,default=Path('/tmp/pytorch-constraints.txt'))
    a=p.parse_args();install_build(a.comfy,a.native,a.constraints)

if __name__=='__main__':main()
