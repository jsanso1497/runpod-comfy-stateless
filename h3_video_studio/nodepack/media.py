"""Local FFmpeg reference decoding, including source-resolution keyframe extraction."""
from __future__ import annotations
import json
import math
from pathlib import Path
import subprocess
import tempfile
from .rules import crop_box, dimensions, frame_times

EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.m4v', '.avi'}


def checked_path(name, root):
    relative = Path(name)
    root = Path(root).resolve()
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('Choose a video inside the ComfyUI input directory.')
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        raise ValueError('Upload a supported video and refresh the file list.')
    if path.stat().st_size > 4 * 1024 ** 3:
        raise ValueError('Trim videos larger than 4 GiB before uploading.')
    return path


def probe_video(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,duration:stream_side_data=rotation:format=duration', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True, timeout=30)
    info = json.loads(result.stdout)
    if not info.get('streams'):
        raise ValueError('The selected file has no video stream.')
    stream = info['streams'][0]
    width, height = int(stream['width']), int(stream['height'])
    rotation = next((float(x['rotation']) for x in stream.get('side_data_list', []) if 'rotation' in x), 0.)
    if round(rotation / 90) % 2:
        width, height = height, width
    duration = stream.get('duration') or info.get('format', {}).get('duration')
    return width, height, float(duration) if duration not in (None, 'N/A') else None


def extract(path, start, duration, as_keyframes, native_mp=1., keyframe_mp=2.5,
            count=4, explicit_times='', crop=(0., 0., 0., 0.)):
    import numpy as np
    import torch
    if not math.isfinite(start) or start < 0 or not math.isfinite(duration) or not .25 <= duration <= 15:
        raise ValueError('Use a nonnegative start and a video section of 0.25 to 15 seconds.')
    width, height, total = probe_video(path)
    if total is not None:
        duration = min(duration, total - start)
        if duration < .25:
            raise ValueError('The selected section is too close to or past the end of this video.')
    x0, y0, x1, y1 = crop_box(width, height, *crop)
    area_mp = keyframe_mp if as_keyframes else min(native_mp, (768 * 1344) / (1024 ** 2))
    w, h = dimensions(x1 - x0, y1 - y0, area_mp, down_only=True)
    filters = 'crop=%d:%d:%d:%d,scale=%d:%d:flags=lanczos,setsar=1' % (x1-x0, y1-y0, x0, y0, w, h)
    base = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '2', '-filter_threads', '1',
            '-protocol_whitelist', 'file,pipe']
    tail = ['-map', '0:v:0', '-an', '-sn', '-dn']
    if as_keyframes:
        times = frame_times(duration, count, explicit_times)
        images = []
        for t in times:
            cmd = base + ['-ss', str(start+t), '-i', str(path)] + tail + ['-vf', filters,
                          '-frames:v', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
            result = subprocess.run(cmd, capture_output=True, timeout=90)
            if result.returncode or len(result.stdout) != w*h*3:
                raise ValueError('Cannot decode keyframe at %.3f seconds: %s' %
                                 (start+t, result.stderr.decode('utf-8', 'replace')[-800:]))
            array = np.frombuffer(result.stdout, np.uint8).copy().reshape(1, h, w, 3)
            images.append(torch.from_numpy(array).float().div_(255.))
        return dict(frames=None, keyframes=images, summary=dict(file=path.name,
            representation='keyframes', source_size=[width,height], crop=[x0,y0,x1,y1],
            decoded_size=[w,h], start_seconds=start, section_seconds=duration,
            timestamps_seconds=[round(start+t,6) for t in times], keyframe_count=len(times),
            source='Decoded directly from the original file, not from a reduced-resolution video preview.', audio='ignored'))
    limit = min(360, int(duration * 24))
    if limit*w*h*3*4 > 2*1024**3:
        raise ValueError('Native clip would exceed 2 GiB of decoded pixels. Reduce the duration or video MP cap.')
    with tempfile.TemporaryDirectory(prefix='h3v-') as directory:
        raw = Path(directory)/'clip.rgb'
        cmd = base + ['-ss', str(start), '-i', str(path), '-t', str(duration)] + tail + [
            '-vf', 'fps=24,'+filters, '-frames:v', str(limit), '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-y', str(raw)]
        result = subprocess.run(cmd, capture_output=True, timeout=180)
        if result.returncode:
            raise ValueError(result.stderr.decode('utf-8','replace')[-1500:])
        available = raw.stat().st_size//(w*h*3)
        if available < 5:
            raise ValueError('At least five frames are required for a native H3 video reference.')
        kept = 5 + 17*((available-5)//17)
        array = np.fromfile(raw, np.uint8, count=kept*w*h*3).reshape(kept,h,w,3)
        frames = torch.from_numpy(array).float().div_(255.)
    return dict(frames=frames, keyframes=[], summary=dict(file=path.name, representation='native video',
        source_size=[width,height], crop=[x0,y0,x1,y1], decoded_size=[w,h], fps=24,
        start_seconds=start, requested_seconds=duration, reference_frames=kept,
        used_seconds=kept/24., audio='ignored', note='Tail trimmed to the 17*n+5 frame grid; output is a single image.'))
