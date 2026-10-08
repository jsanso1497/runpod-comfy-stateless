#!/usr/bin/env python3
"""Fail closed when Photo, Native or Torso workflows are missing in repo source."""
from __future__ import annotations
import argparse,json
from pathlib import Path

NATIVE=34
TORSO=4
PHOTO=('Qwen21_Photo_2K_SAM3_BF16.json','Qwen21_Photo_2K_SAM3_INT8.json')


def check(repo:Path) -> dict:
    repo=repo.resolve()
    photo=repo/'qwen21_photo_edit'
    root=repo/'qwen21_native'
    errors=[]
    ui_files={x.name for x in photo.rglob('*.json') if x.name in PHOTO}
    for name in PHOTO:
        if name not in ui_files:errors.append('Photo/SAM3 UI workflow missing: '+name)
    native=list((root/'workflows').glob('*.json'))
    torso=list((root/'addons/torso_lock/workflows').glob('*.json'))
    if len(native)!=38:errors.append('Native source should contain 38 workflow UI variants, found '+str(len(native)))
    if len(torso)!=4:errors.append('Torso Lock should contain 4 workflows, found '+str(len(torso)))
    for pack in ('qwen21_native_tools','qwen21_torso_tools'):
        source=(root/'custom_nodes'/pack) if pack=='qwen21_native_tools' else (root/'addons/torso_lock/custom_nodes'/pack)
        if not (source/'__init__.py').is_file():errors.append('Missing custom node package: '+str(source))
    for p in (root/'config/models.json',repo/'qwen21_photo_edit/Dockerfile.unified',repo/'qwen21_unified/scripts/entry.sh'):
        if not p.is_file():errors.append('Missing build prerequisite: '+str(p))
    assets=json.loads((root/'config/models.json').read_text()).get('assets',[]) if (root/'config/models.json').is_file() else []
    needed={'qwen_image_2.1_bf16.safetensors','qwen3vl_8b_bf16.safetensors','qwen_image_2.1_vae_bf16.safetensors'}
    actual={a['destination'].split('/')[-1] for a in assets}
    if not needed<=actual:errors.append('Native model manifest is missing a required BF16 Qwen weight: '+str(needed-actual))
    if list(repo.rglob('*.safetensors')):errors.append('Model weight(s) are in the GitHub source tree. Weights must be downloaded at RunPod startup.')
    if errors:raise RuntimeError('\n'.join(errors))
    result={'status':'SOURCE_VERIFIED','photo_workflows':2,'native_source_variants':len(native),
            'native_default_variants':NATIVE,'torso_workflows':len(torso),
            'default_accessible_workflows':2+NATIVE+TORSO}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'))
    a=p.parse_args();result=check(a.repo)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
