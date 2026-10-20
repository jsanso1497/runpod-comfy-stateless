"""Small H3 still utilities. No model downloads, global patches or extra packages."""
from __future__ import annotations
import math
from pathlib import Path
import struct
import zlib

ASPECTS = ('source', '16:9', '9:16', '3:2', '2:3', '1:1', '4:5', '5:4', '3:4', '4:3')


def still_dimensions(megapixels: float, aspect: str = 'source', source_size=None):
    """Round to H3's 32-pixel grid; never exceed the pinned still node's 4096 limit."""
    if not math.isfinite(megapixels) or not 0.25 <= megapixels <= 16.777216:
        raise ValueError('Target megapixels must be between 0.25 and 16.777216.')
    if aspect not in ASPECTS:
        raise ValueError('Unsupported aspect ratio.')
    if aspect == 'source':
        if source_size is None:
            raise ValueError('Choose an explicit aspect ratio when no reference image is connected.')
        width, height = source_size
        if min(width, height) <= 0:
            raise ValueError('Reference dimensions must be positive.')
        ratio = width / height
    else:
        width, height = (int(n) for n in aspect.split(':'))
        ratio = width / height
    if not 0.25 <= ratio <= 4:
        raise ValueError('Use a reference crop with an aspect ratio between 1:4 and 4:1.')
    width = math.sqrt(megapixels * 1_000_000 * ratio)
    height = width / ratio
    scale = min(1.0, 4096 / max(width, height))
    width, height = [max(64, min(4096, round(v * scale / 32) * 32)) for v in (width, height)]
    note = (f'{width} x {height}; {width * height / 1e6:.3f} MP. '
            'Dimensions are rounded to multiples of 32. Reference pixels are not edited by this node. '
            '4096-pixel maximum per side; higher resolution is not a guarantee of better identity.')
    return width, height, note


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)


def encode_rgb16_png(image) -> bytes:
    """Lossless, true 16-bit RGB PNG from one HWC Comfy image. No HDR claim."""
    import numpy as np
    import torch
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected one HWC RGB image tensor.')
    if not torch.isfinite(image).all():
        raise ValueError('Image contains non-finite pixels; refusing a corrupt master.')
    height, width, _ = image.shape
    if min(width, height) < 1 or max(width, height) > 4096:
        raise ValueError('Still masters must have positive dimensions no greater than 4096.')
    pixels = (image.detach().to(device='cpu', dtype=torch.float32).clamp(0, 1) * 65535).round().numpy().astype('>u2')
    # PNG filter 0 for each row, in network byte order. Lossless zlib compression.
    scanlines = b''.join(b'\x00' + pixels[y].tobytes() for y in range(height))
    return (b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 16, 2, 0, 0, 0))
            + _chunk(b'sRGB', b'\x00') + _chunk(b'IDAT', zlib.compress(scanlines, 6)) + _chunk(b'IEND', b''))


def write_rgb16_png(image, path):
    """Create only. Never overwrite an existing image or follow a destination symlink."""
    path = Path(path)
    data = encode_rgb16_png(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(data)
    return path
