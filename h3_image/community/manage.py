#!/usr/bin/env python3
"""Optional Pruna decoder download and live node preflight for Native H3 Image.

No package upgrades, process restarts, prompts, API keys, or GitHub writes.
Run on the Pod. Existing model files are verified, never overwritten.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.request

MODEL = {
    'repo': 'PrunaAI/P-H3-Image-Comfy',
    'revision': '83e4891dd80a33e639e6e75b466fdccecbcad4a9',
    'file': 'vae/p-h3-image_vae_int8_convrot.safetensors',
    'name': 'p-h3-image_vae_int8_convrot.safetensors',
    'size_bytes': 3146996440,
    'sha256': '583e0d43354b6b833aa9a0815a5fac831b1ef6d0c19a7eaa0616980a3dab5ffe',
}
CORE_INPUTS = {
    'BasicScheduler': {'model','scheduler','steps','denoise'},
    'KSamplerSelect': {'sampler_name'},
    'SamplerLCM': {'s_noise','s_noise_end','noise_clip_std'},
    'ImageScaleToMaxDimension': {'image','upscale_method','largest_size'},
    'ImageStitch': {'image1','image2','direction','match_image_size','spacing_width','spacing_color'},
}
REQUIRED_HELPERS = [
    'MiniMaxH3ImageConditioning','H3ImageOptionalReference','H3ImageManualConditioning',
    'H3ImageTimedSampler','H3ImageTimedDecode','H3ImageReport','H3ImageCompare',
    'UNETLoader','CLIPLoader','VAELoader','BasicGuider','RandomNoise','SaveImage',
]


def under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def data_root():
    root = Path(os.environ.get('H3_IMAGE_ROOT','/workspace/h3-image')).resolve()
    if root == Path('/workspace').resolve() or not under(root, '/workspace'):
        raise RuntimeError('H3_IMAGE_ROOT must be a subdirectory of /workspace.')
    return root


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def verify(path, model=MODEL):
    path=Path(path)
    return path.is_file() and path.stat().st_size==model['size_bytes'] and digest(path)==model['sha256']


def safe_dir(root, relative):
    root=Path(root).resolve()
    p=root/relative
    if not under(p,root):
        raise RuntimeError('Refusing a path outside the requested workspace: '+str(p))
    p.mkdir(parents=True,exist_ok=True)
    return p


def register(source,target,model=MODEL):
    source=Path(source);target=Path(target)
    if target.exists() or target.is_symlink():
        raise RuntimeError('Refusing to overwrite an existing model path: '+str(target))
    if not verify(source,model):
        raise RuntimeError('Pruna VAE checksum verification failed. It was not registered.')
    target.symlink_to(source.resolve())
    return target


@contextmanager
def download_lock(path):
    import fcntl
    flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(str(path), flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another optional decoder download is running.') from None
        yield
    finally:
        os.close(descriptor)


def download_vae(root, downloader=None, reuse=None, model=MODEL):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with download_lock(root/'.community-vae-download.lock'):
        target=safe_dir(root,'models/vae')/model['name']
        if verify(target,model):
            print('Already verified: '+str(target));return target
        if target.exists() or target.is_symlink():
            raise RuntimeError('Existing decoder does not match the published checksum; nothing overwritten: '+str(target))
        candidates = [Path('/workspace/h3-still/models/vae')/model['name']] if reuse is None else list(map(Path,reuse))
        for previous in candidates:
            if verify(previous,model):
                # Reading and linking a valid previous model never changes it.
                register(previous,target,model)
                print('Reused previous P-H3 decoder: '+str(previous))
                print('Registered: '+str(target));return target
        cache=safe_dir(root,'community/downloads/pruna_vae')
        source=cache/model['file']
        if source.exists() or source.is_symlink():
            if not verify(source,model):
                raise RuntimeError('An unverified complete cached decoder exists. It was not overwritten: '+str(source))
        else:
            required=model['size_bytes']+4*1024**3
            if shutil.disk_usage(root).free<required:
                raise RuntimeError('Allow the 3.15 GB decoder plus at least 4 GiB free headroom before downloading.')
            if downloader is None:
                from huggingface_hub import hf_hub_download
                downloader=hf_hub_download
            print('Downloading OPTIONAL Pruna INT8 VAE only (3.15 GB).',flush=True)
            got=Path(downloader(repo_id=model['repo'],filename=model['file'],revision=model['revision'],
                local_dir=str(cache),token=os.environ.get('HF_TOKEN') or None))
            if got.resolve()!=source.resolve() or not under(source,cache):
                raise RuntimeError('The downloader returned an unexpected path. Nothing registered.')
        print('Verifying SHA-256 before registration.',flush=True)
        register(source,target,model)
        print('Registered: '+str(target))
        print('Refresh ComfyUI model definitions. No restart or P-H3 build is needed.')
        return target


def validate_live_info(info):
    errors=[]
    for name in REQUIRED_HELPERS:
        if name not in info:errors.append('Missing node: '+name)
    for name,expected in CORE_INPUTS.items():
        if name not in info:
            errors.append('Missing node: '+name);continue
        inputs=info[name].get('input',{})
        actual=set(inputs.get('required',{}))|set(inputs.get('optional',{}))
        if expected-actual:errors.append('%s missing inputs: %s'%(name,', '.join(sorted(expected-actual))))
    sampler=info.get('KSamplerSelect',{}).get('input',{}).get('required',{}).get('sampler_name',[[]])[0]
    if isinstance(sampler,list):
        for name in ('euler','er_sde'):
            if name not in sampler:errors.append('Sampler not available: '+name)
    sched=info.get('BasicScheduler',{}).get('input',{}).get('required',{}).get('scheduler',[[]])[0]
    if isinstance(sched,list) and 'simple' not in sched:errors.append('simple scheduler unavailable')
    return errors


def status(root, info_getter=None):
    if info_getter is None:
        def info_getter():
            with urllib.request.urlopen('http://127.0.0.1:8189/object_info',timeout=15) as response:
                return json.load(response)
    info=info_getter();errors=validate_live_info(info)
    print('Community workflow nodes: '+('READY' if not errors else 'NOT READY'))
    for error in errors:print('  '+error)
    vae=Path(root)/'models/vae'/MODEL['name']
    print('Optional Pruna VAE file: '+('present (presence only, not a checksum test)' if vae.is_file() else 'not downloaded; only workflow 18 needs it'))
    print('Core model/encoder unchanged. GPU inference and 16 MP capacity are not tested by this check.')
    return not errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['status','download-vae'])
    args=p.parse_args()
    if not Path('/opt/ComfyUI/main.py').is_file():
        raise RuntimeError('Run this inside your Native H3 Image RunPod, not in the Mac repository terminal.')
    root=data_root()
    if args.command=='download-vae':download_vae(root)
    elif not status(root):return 2
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        text=re.sub(r'hf_[A-Za-z0-9]+','[redacted]',str(error))
        print('STOPPED: '+text,file=sys.stderr);sys.exit(1)
