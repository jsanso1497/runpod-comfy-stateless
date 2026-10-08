#!/usr/bin/env python3
"""Merge self-contained Qwen Photo + Native + Torso bundle into an existing repo.

Run from the extracted bundle, passing the path to the ACTUAL Git checkout.
Never changes the original Photo Dockerfile or its existing build action.
"""
from __future__ import annotations
import argparse,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
HEX=re.compile(r'^[a-fA-F0-9]{64}$')

def sha(path:Path):return hashlib.sha256(path.read_bytes()).hexdigest()

def update_snapshot(repo:Path, changed:set[str], allowed_rehash:set[str]|None=None) -> None:
    manifest=repo/'SOURCE_SNAPSHOT.json'
    if not manifest.exists():
        print('No root SOURCE_SNAPSHOT.json. No snapshot changed.')
        return
    obj=json.loads(manifest.read_text())
    allowed_rehash = allowed_rehash or set()
    parent=None;member=None;mapping=None;mode=None
    # Conservative recognition of path->sha256 maps only; never rewrite source digests.
    for name in ('files','entries','inventory','source_files','file_hashes','sha256'):
        c=obj.get(name) if isinstance(obj,dict) else None
        if isinstance(c,dict) and len(c)>=10 and sum('/' in k or '.' in k for k in c)>=10:
            if all((isinstance(v,str) and HEX.fullmatch(v)) or (isinstance(v,dict) and isinstance(v.get('sha256'),str)) for v in c.values()):
                mapping=c;parent=obj;member=name;mode='dict';break
        if isinstance(c,list) and len(c)>=10 and isinstance(c[0],dict):
            if {'path','sha256'}<=set(c[0]):
                mapping=c;parent=obj;member=name;mode='list';break
    if mapping is None:
        raise RuntimeError('SOURCE_SNAPSHOT schema not recognized. Do not bypass tools/verify_snapshot.py. Regenerate snapshot with the repository’s own manifest creator.')
    if mode=='dict':
        template=next(iter(mapping.values()))
        for rel in sorted(changed):
            if rel=='SOURCE_SNAPSHOT.json':continue
            p=repo/rel
            if not p.is_file():continue
            digest=sha(p)
            if rel in mapping:
                old=mapping[rel] if isinstance(mapping[rel],str) else mapping[rel]['sha256']
                if old!=digest:
                    if rel not in allowed_rehash:
                        raise RuntimeError(f'Snapshot has a different digest for existing source {rel}; refuse to silently rewrite it. Use the repository manifest tool.')
                    mapping[rel]=digest if isinstance(mapping[rel],str) else {**mapping[rel], 'sha256':digest}
                continue
            mapping[rel]=digest if isinstance(template,str) else {'sha256':digest}
    else:
        existing={row['path']:row for row in mapping}
        for rel in sorted(changed):
            if rel=='SOURCE_SNAPSHOT.json':continue
            p=repo/rel
            if not p.is_file():continue
            digest=sha(p)
            if rel in existing:
                if existing[rel]['sha256']!=digest:
                    if rel not in allowed_rehash:
                        raise RuntimeError(f'Snapshot mismatch for {rel}; do not overwrite old digest')
                    existing[rel]['sha256']=digest
                continue
            mapping.append({'path':rel,'sha256':digest})
        mapping.sort(key=lambda x:x['path'])
    for key in ('count','file_count','files_count','total_files'):
        if key in obj and isinstance(obj[key],int):obj[key]=len(mapping)
    original=manifest.read_bytes()
    manifest.write_text(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+'\n')
    verifier=repo/'tools/verify_snapshot.py'
    if verifier.is_file():
        probe=subprocess.run([sys.executable,str(verifier)],cwd=repo,capture_output=True,text=True)
        if probe.returncode:
            manifest.write_bytes(original)
            raise RuntimeError('Snapshot verifier rejected new file inventory; original manifest restored. Output:\n'+(probe.stdout+'\n'+probe.stderr)[-3000:])
        print('PASS repository snapshot verifier after additive integration')
    else:print('Snapshot updated; repository had no tools/verify_snapshot.py to confirm it.')

def apply(repo:Path,force_native:bool=False,skip_snapshot:bool=False):
    repo=repo.resolve()
    required=[repo/'qwen21_photo_edit/Dockerfile',repo/'qwen21_photo_edit/start.sh',repo/'qwen21_photo_edit/ci_validate.py']
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise RuntimeError('This must target a checkout of the Qwen Photo 2K/SAM3 repository. Missing: '+', '.join(missing))
    sys.path.insert(0,str(ROOT/'qwen21_unified/scripts'))
    from generate_dockerfile import generate
    # Fail before writing anything when Photo source has changed unexpectedly.
    rendered=generate((repo/'qwen21_photo_edit/Dockerfile').read_text())
    target_items=[]
    for dirname in ('qwen21_native','qwen21_unified'):
        for source in (ROOT/dirname).rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts or source.suffix=='.pyc':continue
            rel=source.relative_to(ROOT)
            target_items.append((source,repo/rel))
    for source in (ROOT/'.github/workflows').glob('build-qwen21-unified.yml'):
        target_items.append((source,repo/source.relative_to(ROOT)))
    changed=set()
    for src,dst in target_items:
        rel=dst.relative_to(repo).as_posix()
        if dst.exists() and dst.read_bytes()!=src.read_bytes():
            if not force_native or not rel.startswith('qwen21_native/'):
                raise RuntimeError(f'Conflict: {rel}; will not overwrite existing source. For reviewed Native v1.1.0 files only, rerun with --force-native after reviewing the diff.')
        changed.add(rel)
    target_docker=repo/'qwen21_photo_edit/Dockerfile.unified'
    if target_docker.exists() and target_docker.read_text()!=rendered:
        raise RuntimeError('Existing unified Dockerfile differs from current Photo source. Review manual edits before overwriting.')
    # Apply transactionally: rollback created files and replaced bytes on any
    # validation/snapshot error. Snapshot has its own guarded recovery.
    previous={}
    written=[]
    created_dirs=set()
    def remember_missing_parents(dst:Path):
        parent=dst.parent
        while parent != repo and not parent.exists():
            created_dirs.add(parent)
            parent=parent.parent
    try:
        for src,dst in target_items:
            if dst.exists() and dst.read_bytes()==src.read_bytes():continue
            previous[dst]=dst.read_bytes() if dst.exists() else None
            remember_missing_parents(dst)
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
            written.append(dst)
        if not target_docker.exists() or target_docker.read_text()!=rendered:
            previous[target_docker]=target_docker.read_bytes() if target_docker.exists() else None
            remember_missing_parents(target_docker)
            target_docker.parent.mkdir(parents=True,exist_ok=True)
            target_docker.write_text(rendered)
            written.append(target_docker)
        changed.add(target_docker.relative_to(repo).as_posix())
        if not skip_snapshot:
            allowed={r for r in changed if r.startswith('qwen21_native/')} if force_native else set()
            update_snapshot(repo,changed,allowed_rehash=allowed)
    except Exception:
        for dst in reversed(written):
            former=previous[dst]
            if former is None:dst.unlink(missing_ok=True)
            else:dst.write_bytes(former)
        for directory in sorted(created_dirs,key=lambda path:len(path.parts),reverse=True):
            try:directory.rmdir()
            except OSError:pass
        print('ROLLED BACK: source files restored after failed integration.',file=sys.stderr)
        raise
    print(f'APPLIED: {len(changed)} files tracked; generated Photo-derived unified Dockerfile.')
    print('Existing Photo Dockerfile, native/Photo build workflows, model defaults and unrelated repository files remain unchanged.')
    print('Commit and run: Build Qwen Photo + Native + Torso (Unified). A Docker build/GPU test has NOT yet been run.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--force-native',action='store_true');p.add_argument('--skip-snapshot',action='store_true')
    a=p.parse_args();apply(a.repo,a.force_native,a.skip_snapshot)

if __name__=='__main__':
    try:main()
    except Exception as e:print('APPLY FAILED: '+str(e),file=sys.stderr);sys.exit(1)
