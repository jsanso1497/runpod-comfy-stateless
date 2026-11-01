"""Inspect actual decoded frames; pass the SAME IMAGE tensor to native assembly.

This is not a denoising preview. Core's TAEH3 sampler preview is separate and
approximate. This node adds a bounded, interruptible post-decode inspection hold.
"""
from __future__ import annotations
import time


class H3SwapWindowPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',),
            'iteration_index': ('INT', {'default': 0, 'min': 0}),
            'hold_seconds': ('INT', {'default': 10, 'min': 0, 'max': 60}),
            'hold_first_window_only': ('BOOLEAN', {'default': True})},
            'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('unchanged_frames',)
    FUNCTION = 'inspect'
    CATEGORY = 'H3 Swap/Preview and recovery'
    DESCRIPTION = ('Actual decoded first/middle/last frames. During the hold use ComfyUI Cancel '
                   'to stop before the next window. This never changes generated pixels.')

    def inspect(self, images, iteration_index=0, hold_seconds=10, hold_first_window_only=True,
                prompt=None, extra_pnginfo=None):
        import torch
        from PIL import Image, ImageDraw
        from nodes import PreviewImage
        import comfy.model_management as mm
        import comfy.utils
        from .recovery import media_metadata
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[-1] != 3:
            raise ValueError('Expected a nonempty RGB frame batch')
        if not 0 <= hold_seconds <= 60:
            raise ValueError('Preview hold must be 0..60 seconds')
        mm.throw_exception_if_processing_interrupted()
        indices = sorted(set([0, len(images)//2, len(images)-1]))
        tiles = []
        for i in indices:
            frame = images[i].detach().to(device='cpu', dtype=torch.float32)
            arr = (frame.clamp(0, 1).numpy()*255).round().astype('uint8')
            im = Image.fromarray(arr)
            im.thumbnail((512, 512), Image.Resampling.LANCZOS)
            tile = Image.new('RGB', (im.width, im.height + 24), 'black')
            tile.paste(im, (0, 24))
            ImageDraw.Draw(tile).text((6, 5), 'Window '+str(iteration_index+1)+' / frame '+str(i), fill='white')
            tiles.append(tile)
        montage = Image.new('RGB', (sum(x.width for x in tiles), max(x.height for x in tiles)), 'black')
        x = 0
        for tile in tiles:
            montage.paste(tile, (x, 0)); x += tile.width
        import numpy as np
        batch = torch.from_numpy(np.asarray(montage).copy().astype(np.float32)/255)[None]
        metadata = media_metadata(prompt, extra_pnginfo)
        saved = PreviewImage().save_images(batch, filename_prefix='H3_decoded_window',
                    prompt=metadata.get('prompt'),
                    extra_pnginfo={'workflow': metadata['workflow']} if 'workflow' in metadata else None)
        delay = hold_seconds if not hold_first_window_only or iteration_index == 0 else 0
        print('[H3 DECODED PREVIEW] Window '+str(iteration_index+1)+
              '; actual decoded pixels; inspection hold '+str(delay)+'s. Cancel to stop.', flush=True)
        # Core's binary progress event arrives while this node is executing, so
        # its preview is visible before the hold ends. No custom browser JS.
        progress = comfy.utils.ProgressBar(max(1, delay))
        progress.update_absolute(0, max(1, delay), ('JPEG', montage, 1536))
        end = time.monotonic() + delay
        last = -1
        while time.monotonic() < end:
            mm.throw_exception_if_processing_interrupted()
            elapsed = int(delay-(end-time.monotonic()))
            if elapsed != last:
                progress.update_absolute(elapsed, max(1, delay)); last = elapsed
            time.sleep(min(0.1, max(0, end-time.monotonic())))
        mm.throw_exception_if_processing_interrupted()
        progress.update_absolute(max(1, delay), max(1, delay))
        return {'ui': saved['ui'], 'result': (images,)}


NODE_CLASS_MAPPINGS = {'H3SwapWindowPreview': H3SwapWindowPreview}
NODE_DISPLAY_NAME_MAPPINGS = {'H3SwapWindowPreview': 'H3 Decoded Window Preview + Cancel Hold'}
