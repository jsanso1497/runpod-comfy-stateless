#!/usr/bin/env python3
"""Pinned, verified, resumable model downloads. No downloads during generation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,'/opt/h3-still/runtime')


def confirmed():
    if os.environ.get('H3V_LICENSE_CONFIRMED')!='1':
        raise RuntimeError('Before downloading weights, confirm your right to use them. The H3 community license excludes the US, EU, UK and South Korea without separate permission. After confirming applicable authorization, set H3V_LICENSE_CONFIRMED=1. Files/Jupyter remain available. See LICENSE_NOTES.md.')


def resolve_checksum(row,root):
    from bootstrap import atomic_json, safe_child
    if row.get('sha256'):return row
    resolved=safe_child(root,'.verified/h3v-resolved-'+row['id']+'.json')
    if resolved.is_file():
        existing=json.loads(resolved.read_text())
        if all(existing.get(k)==row[k] for k in ('id','repo','revision','file')) and re.fullmatch('[0-9a-f]{64}',existing.get('sha256','')):
            return existing
    from huggingface_hub import HfApi
    info=HfApi(token=os.environ.get('HF_TOKEN') or None).model_info(row['repo'],revision=row['revision'],files_metadata=True)
    if info.sha!=row['revision']:raise RuntimeError('Publisher revision mismatch while resolving checksum.')
    sibling=next((s for s in info.siblings if s.rfilename==row['file']),None)
    lfs=getattr(sibling,'lfs',None)
    sha=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
    if not sha or not re.fullmatch('[0-9a-f]{64}',sha):raise RuntimeError('No publisher LFS SHA-256 is available for '+row['file'])
    row=dict(row,sha256=sha);atomic_json(resolved,row)
    return row


def download(root,group='hq'):
    confirmed()
    from bootstrap import safe_child,verified
    from huggingface_hub import hf_hub_download
    import fcntl
    root=Path(root).resolve()
    if root==Path('/workspace') or not root.is_relative_to(Path('/workspace')):raise ValueError('Data root must be inside /workspace.')
    root.mkdir(parents=True,exist_ok=True)
    rows=json.loads((PACKAGE/'manifest.json').read_text())['models']
    groups={'hq'} if group=='hq' else {'bf16'} if group=='bf16' else {'hq','bf16'}
    lock=safe_child(root,'.h3v-download.lock')
    if lock.is_symlink():raise ValueError('Refusing a symlinked download lock.')
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        for original in rows:
            if original['group'] not in groups:continue
            row=resolve_checksum(original,root)
            dest=safe_child(root/'models',row['destination'])
            marker=safe_child(root,'.verified/h3v-'+row['id']+'.json')
            if verified(dest,row,marker,os.environ.get('H3_STILL_VERIFY_ALL')=='1'):
                print('Verified cache:',row['id'],flush=True);continue
            if dest.exists() or dest.is_symlink():raise RuntimeError('Existing file failed its checksum; it was not overwritten: '+str(dest))
            source=safe_child(root/'models','.sources/h3v-'+row['id'])
            source.mkdir(parents=True,exist_ok=True)
            completed=safe_child(source,row['file'])
            if not verified(completed,row,marker,False):
                # Conservative free-space guard. Existing partial downloads may require less.
                needed=int(row['approx_gb']*1e9)+2_000_000_000
                if shutil.disk_usage(root).free<needed:raise RuntimeError('Insufficient free persistent disk for '+row['id']+'. Use 250 GB for HQ or 500 GB for all precision sets.')
                print('Downloading pinned asset:',row['id'],flush=True)
                completed=Path(hf_hub_download(repo_id=row['repo'],filename=row['file'],revision=row['revision'],
                    local_dir=str(source),token=os.environ.get('HF_TOKEN') or None))
                if not completed.resolve().is_relative_to(source.resolve()) or not verified(completed,row,marker,True):
                    raise RuntimeError('Downloaded checksum verification failed for '+row['id'])
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(os.path.relpath(completed,dest.parent))
    print('H3 Video Still Studio model set ready:',group,flush=True)


def verify_remote_metadata():
    """Verify pinned filenames/checksums using metadata only; never fetch model weights."""
    from huggingface_hub import HfApi
    rows=json.loads((PACKAGE/'manifest.json').read_text())['models'];cache={};checks=[]
    api=HfApi(token=os.environ.get('HF_TOKEN') or None)
    for row in rows:
        key=(row['repo'],row['revision'])
        if key not in cache:
            info=api.model_info(row['repo'],revision=row['revision'],files_metadata=True)
            if info.sha!=row['revision']:raise RuntimeError('Publisher revision mismatch: '+row['id'])
            cache[key]={x.rfilename:x for x in info.siblings}
        file=cache[key].get(row['file'])
        lfs=getattr(file,'lfs',None)
        sha=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
        if not sha or not re.fullmatch('[0-9a-f]{64}',sha):raise RuntimeError('Missing pinned model file or LFS checksum: '+row['id'])
        if row.get('sha256') and sha!=row['sha256']:raise RuntimeError('Publisher checksum differs from package lock: '+row['id'])
        checks.append(dict(id=row['id'],revision=row['revision'],sha256=sha,status='verified publisher metadata'))
    report=dict(model_weights_downloaded=False,checks=checks)
    (PACKAGE/'remote-model-metadata-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['download','status','verify-metadata'])
    parser.add_argument('group',choices=['hq','bf16','all'],nargs='?',default='hq')
    parser.add_argument('--root',default=os.environ.get('H3_STILL_DATA_ROOT','/workspace/h3-still'))
    args=parser.parse_args()
    if args.action=='download':download(Path(args.root),args.group)
    elif args.action=='verify-metadata':verify_remote_metadata()
    else:
        for row in json.loads((PACKAGE/'manifest.json').read_text())['models']:
            path=Path(args.root)/'models'/row['destination']
            print(('PRESENT ' if path.is_file() else 'MISSING ')+row['id']+' '+str(path))
if __name__=='__main__':main()
