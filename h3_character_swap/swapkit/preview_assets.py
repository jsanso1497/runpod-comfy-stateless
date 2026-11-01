"""Pinned tiny preview decoder only; no replacement of generation weights.

Git blob SHA is verified against the inspected immutable GitHub tree. A SHA256
receipt is also written after download. Runtime never silently replaces a
conflicting file on the persistent volume.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from .config import ROOT, atomic_json, data_root

COMMIT = '011dfc2112197741c540e0bdd5b7b67bcc930771'
BLOB = '9eff5f6da1f0baa60ede7f097507119b1488219f'
SIZE = 22709752
NAME = 'taeh3.safetensors'
URL = f'https://raw.githubusercontent.com/madebyollin/taehv/{COMMIT}/safetensors/{NAME}'


def valid(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    digest = hashlib.sha1(f'blob {SIZE}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest() == BLOB


def download():
    import requests
    folder = ROOT / 'preview_assets'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / NAME
    if not valid(path):
        temp = path.with_suffix('.partial')
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.get(URL, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    total = 0
                    with temp.open('wb') as stream:
                        for block in response.iter_content(1024*1024):
                            total += len(block)
                            if total > SIZE:
                                raise ValueError('Unexpected tiny-preview asset size')
                            stream.write(block)
            if not valid(temp):
                raise ValueError('Tiny-preview decoder Git blob verification failed')
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    atomic_json(folder / 'receipt.json', {'url': URL, 'git_commit': COMMIT, 'git_blob': BLOB,
        'bytes': SIZE, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'purpose': 'approximate sampling preview only; never the final decoder'})
    return path


def install_workspace():
    source = ROOT / 'preview_assets' / NAME
    if not valid(source):
        raise ValueError('Preview decoder is missing from the rebuilt image')
    target = data_root() / 'models/vae_approx' / NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError('Refusing a symlinked preview decoder')
    if target.exists() and not valid(target):
        raise ValueError('Different taeh3.safetensors on volume; not overwritten. '
                         'Rename it or set H3_SWAP_PREVIEW_METHOD=latent2rgb.')
    if not target.exists():
        temporary = target.with_suffix('.partial')
        shutil.copy2(source, temporary)
        if not valid(temporary):
            raise ValueError('Copied preview decoder failed verification')
        os.replace(temporary, target)
    return target


def method():
    value = os.environ.get('H3_SWAP_PREVIEW_METHOD', 'taesd')
    if value not in ('taesd', 'latent2rgb', 'none'):
        raise ValueError('H3_SWAP_PREVIEW_METHOD must be taesd, latent2rgb or none')
    return value


def smoke():
    """CI only: actual pinned ComfyUI + actual tiny decoder, CPU, no H3 weights."""
    path = ROOT / 'preview_assets' / NAME
    if not valid(path):
        raise ValueError('Preview asset verification failed')
    sys.path.insert(0, '/opt/ComfyUI')
    sys.argv = [sys.argv[0], '--cpu', '--preview-method', 'taesd']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    import comfy.latent_formats
    import latent_preview
    folder_paths.folder_names_and_paths['vae_approx'] = ([str(path.parent)], {'.safetensors'})
    fmt = comfy.latent_formats.MiniMaxH3AV()
    fmt.compile_preview = False
    previewer = latent_preview.get_previewer(torch.device('cpu'), fmt)
    if not isinstance(previewer, latent_preview.TAEHVPreviewerImpl):
        raise ValueError('Pinned ComfyUI did not select the H3 tiny preview decoder')
    with torch.inference_mode():
        image = previewer.decode_latent_to_preview(torch.zeros((1, 24, 1, 4, 4)))
    if image.mode != 'RGB' or min(image.size) < 32:
        raise ValueError('Preview smoke produced invalid image')
    print('H3 tiny-preview CPU decode passed:', image.size, flush=True)
    # Exercise real core preview and video metadata IO as well as imports.
    import tempfile
    import subprocess
    from fractions import Fraction
    from comfy_api.latest._input_impl.video_types import VideoFromComponents
    from comfy_api.latest._util import VideoComponents
    from .preview_nodes import H3SwapWindowPreview
    from .nodes import H3SwapSaveVideo
    with tempfile.TemporaryDirectory(prefix='h3-preview-io-smoke-') as folder:
        old_output=folder_paths.get_output_directory()
        old_temp=folder_paths.get_temp_directory()
        try:
            folder_paths.set_output_directory(folder)
            folder_paths.set_temp_directory(str(Path(folder)/'temp'))
            frames=torch.zeros((5,64,64,3))
            metadata={'workflow':{'version':0.4,'nodes':[],'links':[]}}
            graph={'test':{'class_type':'PreviewAny','inputs':{'source':'CPU smoke'}}}
            output=H3SwapWindowPreview().inspect(frames,hold_seconds=0,
                    prompt=graph,extra_pnginfo=metadata)
            if output['result'][0] is not frames or not output['ui'].get('images'):
                raise ValueError('Decoded preview failed its unchanged-pixel contract')
            video=VideoFromComponents(VideoComponents(images=frames,frame_rate=Fraction(24)))
            H3SwapSaveVideo().save(video,'cpu_smoke',16,graph,metadata)
            saved=next(Path(folder).glob('*.mp4'))
            tags=json.loads(subprocess.check_output(['ffprobe','-v','error',
                '-show_entries','format_tags','-of','json',str(saved)]))['format']['tags']
            if json.loads(tags['workflow'])!=metadata['workflow'] or json.loads(tags['prompt'])!=graph:
                raise ValueError('Native MP4 workflow metadata round trip failed')
            if not Path(str(saved)+'.workflow.json').is_file():
                raise ValueError('Native video workflow sidecar is missing')
            print('Actual ComfyUI decoded-preview and MP4 metadata IO smoke passed',flush=True)
        finally:
            folder_paths.set_output_directory(old_output)
            folder_paths.set_temp_directory(old_temp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    if options.download:
        print(download())
    elif options.smoke:
        smoke()
    else:
        parser.error('Choose --download or --smoke')

if __name__ == '__main__':
    main()
