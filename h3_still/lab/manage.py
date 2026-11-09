#!/usr/bin/env python3
"""Opt-in downloads and architecture-specific setup, run inside the Pod."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

PACKAGE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('H3_STILL_DATA_ROOT','/workspace/h3-still'))
MANIFEST=json.loads((PACKAGE/'manifest.json').read_text())


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def safe_path(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative:raise ValueError('Unsafe path: '+relative)
    out=(root/p).resolve()
    if not out.is_relative_to(root.resolve()):raise ValueError('Path escapes '+str(root))
    return out


def verified(path,row):
    if not path.is_file():return False
    marker=ROOT/'lab/verified'/(row['id']+'.json')
    stat=path.stat()
    signature={'size':stat.st_size,'mtime':stat.st_mtime_ns,'sha256':row['sha256'],'path':str(path.resolve())}
    try:
        if json.loads(marker.read_text())==signature:return True
    except (OSError,ValueError):pass
    print('Checking SHA-256:',path.name,flush=True)
    if sha256(path)!=row['sha256']:return False
    marker.parent.mkdir(parents=True,exist_ok=True)
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(signature));tmp.replace(marker)
    return True


def download(group):
    rows=[row for row in MANIFEST['downloads'] if row['group']==group or group=='all']
    if not rows:raise ValueError('Choose sam, swap, ph3 or all.')
    for row in rows:
        target=ROOT/'models'/row['destination']
        if target.is_file() and verified(target,row):
            print('Already verified:',target);continue
        if target.exists() or target.is_symlink():
            raise RuntimeError('Existing file does not match the locked checksum; not overwritten: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        expected=int(row['approx_gb']*1e9)
        if shutil.disk_usage(ROOT).free<expected+1024**3:
            raise RuntimeError('Insufficient disk space for '+row['id']+'. Keep at least an extra 1 GB free.')
        from huggingface_hub import hf_hub_download
        cache=ROOT/'lab/model_sources'/row['id']
        cache.mkdir(parents=True,exist_ok=True)
        print('Downloading optional model:',row['id'],flush=True)
        source=Path(hf_hub_download(repo_id=row['repo'],revision=row['revision'],filename=row['file'],
                    local_dir=str(cache),token=os.environ.get('HF_TOKEN') or None))
        if not source.resolve().is_relative_to(cache.resolve()) or not verified(source,row):
            raise RuntimeError('Model SHA-256 verification failed; model was not registered.')
        target.symlink_to(source.resolve())
        print('Registered:',target,flush=True)
    print('Done. Refresh ComfyUI model definitions. No service restart normally needed.')


def safe_extract(archive,destination):
    with zipfile.ZipFile(archive) as z:
        if sum(x.file_size for x in z.infolist())>512*1024**2 or len(z.infolist())>20000:
            raise ValueError('Unexpected Spark archive size.')
        for member in z.infolist():
            safe_path(destination,member.filename)
            if stat.S_ISLNK(member.external_attr>>16):raise ValueError('Archive contains a symbolic link.')
        z.extractall(destination)


def spark_setup(restore=False):
    import torch
    if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability())!=(8,6):
        if restore:return
        raise RuntimeError('Spark package is SM86 only (for example A40). A100 is SM80; use Spectrum or baseline there.')
    root=ROOT/'lab/providers/spark';node=root/'ComfyUI-Spark-H3';spec=MANIFEST['spark']
    archive=ROOT/'lab/downloads/spark-sm86-v020.zip'
    if restore:
        if not (root/'enabled.json').is_file():return
        if not archive.is_file() or sha256(archive)!=spec['sha256']:
            raise RuntimeError('Cached Spark archive missing/changed. Run manage.py setup spark again.')
    else:
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():
            temp=archive.with_suffix('.part')
            req=urllib.request.Request(spec['url'],headers={'User-Agent':'H3-Studio-Lab/1.0'})
            with urllib.request.urlopen(req,timeout=90) as response,temp.open('wb') as f:
                shutil.copyfileobj(response,f,1024*1024)
            if temp.stat().st_size!=spec['size_bytes'] or sha256(temp)!=spec['sha256']:
                raise RuntimeError('Spark archive checksum mismatch. It was not executed.')
            temp.replace(archive)
        if sha256(archive)!=spec['sha256']:raise RuntimeError('Spark archive has changed; not executed.')
        if not node.exists():
            root.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='spark-stage-',dir=str(root.parent)) as t:
                stage=Path(t);safe_extract(archive,stage)
                matches=[p.parent for p in stage.rglob('install.py') if (p.parent/'__init__.py').is_file()]
                if len(matches)!=1:raise RuntimeError('Cannot identify the Spark node folder in the published archive.')
                root.mkdir(parents=True,exist_ok=True);shutil.copytree(matches[0],node)
        if not (node/'install.py').is_file():raise RuntimeError('Incomplete Spark package. Nothing was installed.')
    # Installation is restricted to the isolated backend. Original comfy-kitchen and torch stay intact.
    old={n:importlib.metadata.version(n) for n in ('torch','comfy-kitchen')}
    wheels=sorted(node.rglob('spark_h3_kitchen-0.2.37+spark.h3.sm86.cu130.1-*.whl'))
    if not wheels:raise RuntimeError('Pinned SM86 0.2.37 wheel not found; no source-build fallback is attempted.')
    try:present=importlib.metadata.version('spark-h3-kitchen')
    except importlib.metadata.PackageNotFoundError:present=''
    if present!='0.2.37+spark.h3.sm86.cu130.1':
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index',str(wheels[0])],check=True)
    if {n:importlib.metadata.version(n) for n in old}!=old:
        raise RuntimeError('Unexpected change to base dependencies.')
    if not restore:
        # Official installer probes both CUDA paths. It should reuse the installed wheel.
        subprocess.run([sys.executable,str(node/'install.py'),'--wheel',str(wheels[0])],cwd=node,check=True)
        if {n:importlib.metadata.version(n) for n in old}!=old:raise RuntimeError('Provider installer changed a protected dependency.')
        (root/'enabled.json').write_text(json.dumps({'archive_sha256':spec['sha256'],'implementation':spec['implementation']}))
        print('Spark SM86 backend installed. Finish/clear the queue and restart the Pod once before testing.')


def status():
    print('H3 Studio Lab',MANIFEST['version'])
    print('Build profile:',(PACKAGE/'build-profile.txt').read_text().strip() if (PACKAGE/'build-profile.txt').exists() else 'source-only')
    for row in MANIFEST['downloads']:
        p=ROOT/'models'/row['destination'];print(row['id']+': '+('present' if p.is_file() else 'not downloaded'))
    try:
        import torch
        print('CUDA:',torch.cuda.is_available())
        if torch.cuda.is_available():print('GPU:',torch.cuda.get_device_name(),torch.cuda.get_device_capability())
    except ImportError:pass
    try:
        with urllib.request.urlopen('http://127.0.0.1:8189/object_info',timeout=5) as r:info=json.load(r)
        names=sorted(n for n in info if n.startswith('H3Lab'))
        print('Registered Lab nodes:',len(names));print('P-H3 native node:', 'PH3ImageConditioning' in info)
    except Exception as e:print('ComfyUI node query not ready:',type(e).__name__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    d=subs.add_parser('download');d.add_argument('group',choices=['sam','swap','ph3','all'])
    s=subs.add_parser('setup');s.add_argument('feature',choices=['spark'])
    subs.add_parser('status');subs.add_parser('restore-spark')
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.command=='download':download(args.group)
    elif args.command=='setup':spark_setup()
    elif args.command=='restore-spark':spark_setup(True)
    else:status()
if __name__=='__main__':
    try:main()
    except Exception as error:print('STOPPED:',str(error),file=sys.stderr);sys.exit(1)
