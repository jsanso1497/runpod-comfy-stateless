#!/usr/bin/env python3
"""Keep PyTorch/CUDA stack shipped in the base image pinned during dependency installation."""
from importlib import metadata
from pathlib import Path
import sys


def pins():
    lines = []
    for name in ('torch', 'torchvision', 'torchaudio'):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            if name == 'torch':
                raise RuntimeError('Missing base-image torch; will not replace the CUDA stack.')
            continue
        lines.append(f'{name}=={version}')
    return lines


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/qwen-photo-torch-constraints.txt')
    path.write_text('\n'.join(pins()) + '\n')
    print('Pinned base image CUDA packages:', ', '.join(pins()), flush=True)
