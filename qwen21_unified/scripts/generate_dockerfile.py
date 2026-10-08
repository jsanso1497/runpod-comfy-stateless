#!/usr/bin/env python3
"""Derive unified Dockerfile strictly from EXISTING pinned Qwen Photo Dockerfile.

Retains source checkout, SAM3, photo LoRA, and runtime install logic. Replaces
only optional ComfyUI gallery media dependency with on-image trimmed list,
adds Native/AusBoss/Torso files, and wraps the existing Photo startup.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path

OLD_REQ='python -m pip install --constraint /tmp/pytorch-constraints.txt -r /opt/ComfyUI/requirements.txt'
NEW_REQ=('python /opt/qwen21_unified/scripts/filter_requirements.py /opt/ComfyUI/requirements.txt /tmp/comfy-unified-requirements.txt '
         '&& python -m pip install --no-cache-dir --constraint /tmp/pytorch-constraints.txt -r /tmp/comfy-unified-requirements.txt')
INSTALL=("\n# Qwen Photo, Native Qwen and Torso Lock integration: reuse one ComfyUI/Torch installation.\n"
         "RUN python /opt/qwen21_unified/scripts/build_install.py --comfy /opt/ComfyUI "
         "&& python -m compileall -q /opt/qwen21_unified/scripts /opt/qwen21_native/custom_nodes /opt/qwen21_native/addons/torso_lock/custom_nodes "
         "&& chmod +x /opt/qwen21_unified/scripts/entry.sh\n")
COPY=('COPY qwen21_native/ /opt/qwen21_native/\n'
      'COPY qwen21_unified/ /opt/qwen21_unified/\n')

def generate(photo: str) -> str:
    if len(re.findall(r'(?m)^FROM\s+',photo)) != 1:raise ValueError('Expected a single-stage Photo Dockerfile; refusing to rewrite unfamiliar multi-stage build.')
    if photo.count(OLD_REQ)!=1:raise ValueError('Photo dependency-install command changed; cannot safely produce a unified build.')
    if 'qwen21_unified/scripts' in photo:raise ValueError('Photo Dockerfile already appears unified; use the original qwen21_photo_edit/Dockerfile as source.')
    if photo.count('COPY qwen21_photo_edit/ /opt/qwen21_photo_edit/')!=1:raise ValueError('Cannot find Photo package COPY anchor.')
    entries=re.findall(r'(?m)^ENTRYPOINT\s+(\[[^\n]+\])\s*$',photo)
    if len(entries)!=1:raise ValueError('Expected exactly one JSON ENTRYPOINT in Photo Dockerfile.')
    original=json.loads(entries[0])
    if '/opt/qwen21_photo_edit/start.sh' not in original:
        raise ValueError('Original Photo ENTRYPOINT differs from verified build. It must use /opt/qwen21_photo_edit/start.sh; check the current repository.')
    if len(re.findall(r'(?m)^WORKDIR\s+',photo))!=1:raise ValueError('Expected one WORKDIR anchor.')
    result=photo.replace('COPY qwen21_photo_edit/ /opt/qwen21_photo_edit/',
        'COPY qwen21_photo_edit/ /opt/qwen21_photo_edit/\n'+COPY.rstrip(),1)
    result=result.replace(OLD_REQ,NEW_REQ,1)
    result=result.replace('python -m pip install --constraint /tmp/pytorch-constraints.txt scipy huggingface-hub pillow',
                          'python -m pip install --no-cache-dir --constraint /tmp/pytorch-constraints.txt scipy huggingface-hub pillow',1)
    result=result.replace('WORKDIR ',INSTALL+'WORKDIR ',1)
    new_original=[('/opt/qwen21_unified/scripts/entry.sh' if part=='/opt/qwen21_photo_edit/start.sh' else part) for part in original]
    result=re.sub(r'(?m)^ENTRYPOINT\s+\[[^\n]+\]\s*$',
                  'ENTRYPOINT '+json.dumps(new_original),result,count=1)
    result=result.replace('FROM ', 'FROM ',1)
    # Prepended to all RUN pip invocations, without affecting the base pinned Torch.
    spot=result.index('\n',result.index('FROM '))
    result=result[:spot+1]+'ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1\n'+result[spot+1:]
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--check',action='store_true');a=p.parse_args()
    photo=a.repo/'qwen21_photo_edit/Dockerfile';out=a.repo/'qwen21_photo_edit/Dockerfile.unified'
    if not photo.exists():raise SystemExit('Missing original Photo Dockerfile '+str(photo))
    rendered=generate(photo.read_text())
    if a.check:
        if not out.exists() or out.read_text()!=rendered:raise SystemExit('Dockerfile.unified differs from generated version; rerun APPLY_QWEN_UNIFIED.py.')
        print('PASS generated unified Dockerfile matches current Photo source.')
    else:
        out.write_text(rendered)
        print('Generated:',out)

if __name__=='__main__':main()
