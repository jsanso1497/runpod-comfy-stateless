#!/usr/bin/env python3
"""Download official ComfyUI model packages; never overwrite valid existing weights."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import struct
import time

REPOS = {
    'bf16': ('Comfy-Org/Qwen-Image-2.1', [
        'diffusion_models/qwen_image_2.1_bf16.safetensors',
        'text_encoders/qwen3vl_8b_bf16.safetensors',
        'vae/qwen_image_2.1_vae_bf16.safetensors']),
    'int8': ('Comfy-Org/Qwen-Image-2.1', [
        'diffusion_models/qwen_image_2.1_int8_convrot.safetensors',
        'text_encoders/qwen3vl_8b_int8_convrot.safetensors',
        'vae/qwen_image_2.1_vae_bf16.safetensors']),
    'sam': ('Comfy-Org/sam3.1', ['checkpoints/sam3.1_multiplex_fp16.safetensors']),
}


def valid_safetensors(path: Path) -> bool:
    """Structural completeness check, not a cryptographic provenance assertion."""
    try:
        size = path.stat().st_size
        with path.open('rb') as handle:
            length, = struct.unpack('<Q', handle.read(8))
            if not 2 <= length <= min(64 * 1024 * 1024, size - 8):
                return False
            header = json.loads(handle.read(length))
        rows = [row for name, row in header.items() if name != '__metadata__']
        if not rows:
            return False
        spans = sorted(tuple(row['data_offsets']) for row in rows)
        last = 0
        for start, end in spans:
            if start != last or end < start:
                return False
            last = end
        return 8 + length + last == size
    except (OSError, ValueError, TypeError, KeyError, struct.error, UnicodeDecodeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfy-home', type=Path, default=Path('/workspace/ComfyUI'))
    parser.add_argument('--models', choices=['none', 'bf16', 'int8', 'both'], default='bf16')
    parser.add_argument('--sam', action='store_true')
    args = parser.parse_args()
    selected = [] if args.models == 'none' else ['bf16', 'int8'] if args.models == 'both' else [args.models]
    if args.sam: selected.append('sam')
    if not selected: return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit('Install the downloader first: python -m pip install huggingface-hub') from exc
    target = args.comfy_home.resolve() / 'models'
    target.mkdir(parents=True, exist_ok=True)
    seen = set()
    for key in selected:
        repo, files = REPOS[key]
        for filename in files:
            if filename in seen: continue
            seen.add(filename)
            destination = target / filename
            if valid_safetensors(destination):
                print('Already present, structurally complete:', filename, flush=True)
                continue
            if destination.exists():
                backup = destination.with_name(destination.name + f'.invalid-{time.time_ns()}')
                destination.rename(backup)
                print('Preserved incomplete/invalid file as:', backup.name, flush=True)
            print('Downloading:', repo, filename, flush=True)
            hf_hub_download(repo_id=repo, filename=filename, local_dir=str(target),
                            token=os.environ.get('HF_TOKEN') or None)
            if not valid_safetensors(destination):
                raise SystemExit('Downloaded file failed the safetensors completeness check: ' + filename)
    print('Selected downloads verified for structural completeness. Restart ComfyUI after node installation.')

if __name__ == '__main__': main()
