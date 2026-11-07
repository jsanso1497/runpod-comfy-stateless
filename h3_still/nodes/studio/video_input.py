"""Bounded, local-file-only FFmpeg preparation for H3 video references."""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from .logic import NONE
from .media_rules import aligned_frames, video_dimensions

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.m4v', '.avi'}


def checked_video_path(name):
    import folder_paths
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('Select a video uploaded into the ComfyUI input folder.')
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError('Upload an MP4/MOV/MKV/WebM video into the input folder and select it again.')
    if path.stat().st_size > 4 * 1024 ** 3:
        raise ValueError('Video input exceeds 4 GB. Trim it before uploading.')
    return path


def decode_section(path, start_seconds, duration_seconds, cap_megapixels):
    """Normalize time before downsampling; no whole-clip decoding or audio model."""
    import numpy as np
    import torch
    for value in (start_seconds, duration_seconds, cap_megapixels):
        if not math.isfinite(float(value)):
            raise ValueError('Video controls must be finite numbers.')
    if start_seconds < 0 or not .25 <= duration_seconds <= 15:
        raise ValueError('Use a nonnegative start and a section between 0.25 and 15 seconds.')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,duration:stream_side_data=rotation:format=duration', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True, timeout=30)
    meta = json.loads(probe.stdout)
    if not meta.get('streams'):
        raise ValueError('No video stream was found.')
    stream = meta['streams'][0]
    width, height = int(stream['width']), int(stream['height'])
    rotation = next((float(x['rotation']) for x in stream.get('side_data_list', []) if 'rotation' in x), 0.)
    if round(rotation / 90) % 2:
        width, height = height, width
    w, h = video_dimensions(width, height, cap_megapixels)
    limit = min(360, max(5, int(duration_seconds * 24 + 1e-6)))
    if limit * w * h * 3 * 4 > 2 * 1024 ** 3:
        raise ValueError('Selected reference would exceed 2 GiB of decoded pixels. Reduce duration or the video MP cap.')
    # A temporary raw file bounds memory and avoids a second giant captured stdout buffer.
    with tempfile.TemporaryDirectory(prefix='h3-video-') as temp:
        raw = Path(temp) / 'frames.rgb'
        filters = 'fps=24,scale=%s:%s:flags=lanczos,setsar=1' % (w, h)
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-filter_threads', '1',
            '-protocol_whitelist', 'file,pipe', '-ss', str(float(start_seconds)), '-i', str(path),
            '-t', str(float(duration_seconds)), '-map', '0:v:0', '-an', '-sn', '-dn',
            '-vf', filters, '-frames:v', str(limit), '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-y', str(raw)]
        result = subprocess.run(command, capture_output=True, timeout=180)
        if result.returncode:
            raise ValueError('Could not decode the selected video section: ' +
                             result.stderr.decode('utf-8', 'replace')[-1500:])
        available = raw.stat().st_size // (w * h * 3)
        kept = aligned_frames(available)
        values = np.fromfile(raw, dtype=np.uint8, count=kept * w * h * 3).reshape(kept, h, w, 3)
        frames = torch.from_numpy(values).to(dtype=torch.float32).div_(255.)
    info = dict(file=path.name, start_seconds=float(start_seconds), requested_seconds=float(duration_seconds),
                fps=24, decoded_frames=available, reference_frames=kept, used_seconds=round(kept / 24., 4),
                input_dimensions=[width, height], reference_dimensions=[w, h], audio='ignored',
                note='Tail trimmed to the H3 17*n+5 frame grid. This is reference duration, not output duration.')
    return frames, info


class H3StudioOptionalVideo:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        root = Path(folder_paths.get_input_directory())
        names = sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        return {'required': {
            'file': ([NONE] + names, {'video_upload': True}),
            'enabled': ('BOOLEAN', {'default': True}),
            'start_seconds': ('FLOAT', {'default': 0., 'min': 0., 'max': 86400., 'step': .1}),
            'duration_seconds': ('FLOAT', {'default': 3.1, 'min': .25, 'max': 15., 'step': .1}),
            'video_cap_megapixels': ('FLOAT', {'default': .5, 'min': .1, 'max': 1., 'step': .1}),
        }}
    RETURN_TYPES = ('H3_STILL_VIDEO',)
    RETURN_NAMES = ('video_reference',)
    FUNCTION = 'load'
    CATEGORY = 'H3 Still Studio'
    DESCRIPTION = 'Optional silent reference video. Bounded clip, 24 fps, H3 frame grid. Output remains one still.'

    @classmethod
    def VALIDATE_INPUTS(cls, file, enabled=True, **kwargs):
        if not enabled or file == NONE:
            return True
        try:
            checked_video_path(file)
            return True
        except (ValueError, OSError) as error:
            return str(error)

    @classmethod
    def IS_CHANGED(cls, file, enabled=True, **kwargs):
        if not enabled or file == NONE:
            return 'empty'
        path = checked_video_path(file)
        stat = path.stat()
        return '%s:%s:%s' % (path, stat.st_size, stat.st_mtime_ns)

    def load(self, file, enabled=True, start_seconds=0., duration_seconds=3.1, video_cap_megapixels=.5):
        if not enabled or file == NONE:
            return ({'frames': None, 'summary': None},)
        frames, summary = decode_section(checked_video_path(file), start_seconds, duration_seconds, video_cap_megapixels)
        return ({'frames': frames, 'summary': summary},)
