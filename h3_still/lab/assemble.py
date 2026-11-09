#!/usr/bin/env python3
"""Build-time only: pinned optional provider sources and isolated P-H3 core upgrade."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

P=Path(__file__).resolve().parent
M=json.loads((P/'manifest.json').read_text())

def checkout(url,revision,destination):
    if destination.exists():raise RuntimeError('Refusing existing checkout: '+str(destination))
    subprocess.run(['git','init',str(destination)],check=True)
    subprocess.run(['git','-C',str(destination),'remote','add','origin',url],check=True)
    subprocess.run(['git','-C',str(destination),'fetch','--depth','1','origin',revision],check=True)
    subprocess.run(['git','-C',str(destination),'checkout','--detach','FETCH_HEAD'],check=True)
    actual=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:raise RuntimeError('Pinned source mismatch.')
    shutil.rmtree(destination/'.git')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['stable','ph3'],default='stable');a=parser.parse_args()
    for name,row in M['providers'].items():checkout(row['url'],row['revision'],P/'vendor'/name)
    if a.profile=='ph3':
        with tempfile.TemporaryDirectory(prefix='ph3-core-') as temp:
            stage=Path(temp)/'ComfyUI';checkout('https://github.com/Comfy-Org/ComfyUI.git',M['ph3_comfy_commit'],stage)
            # An image-layer update only; no user volume or custom node is deleted.
            for item in stage.iterdir():
                if item.name in ('custom_nodes','models','input','output','user','temp','.git'):continue
                dest=Path('/opt/ComfyUI')/item.name
                if item.is_dir():shutil.copytree(item,dest,dirs_exist_ok=True)
                else:shutil.copy2(item,dest)
        subprocess.run([sys.executable,'-m','pip','install','-c','/opt/h3-still/constraints.txt','-r','/opt/ComfyUI/requirements.txt'],check=True)
    dest=Path('/opt/ComfyUI/custom_nodes/H3_Studio_Lab')
    shutil.copytree(P/'nodepack',dest)
    (P/'build-profile.txt').write_text(a.profile+'\n')
    subprocess.run([sys.executable,str(P/'build_workflows.py')],check=True)
    (P/'build-provenance.json').write_text(json.dumps({'profile':a.profile,'manifest':M,'gpu_tested':False},indent=2)+'\n')
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
if __name__=='__main__':main()
