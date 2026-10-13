"""Bounded local MP4 demuxing. Never decode a whole movie to get its voice.

Uses existing ffmpeg/ffprobe, original audio decoded to float PCM, and 24 fps
video frames. No external inference, transcription, voice training or denoising.
"""
from __future__ import annotations
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

FPS = 24
MAX_SECONDS = 15.0
MAX_FRAME_PIXELS = 1_100_000


def number(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f'{name} must be a finite number from {lo} to {hi}.')
    return float(value)


def _run(command, timeout=120):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError('ffmpeg and ffprobe must be installed in the container.') from exc
    if result.returncode:
        raise ValueError('Cannot decode the selected local media: ' + result.stderr[-1400:])
    return result


def local_video_source(video):
    getter = getattr(video, 'get_stream_source', None)
    if not callable(getter):
        raise ValueError('Connect the native LoadVideo node directly. This decoder requires its file-backed VIDEO input.')
    source = getter()
    if not isinstance(source, (str, Path)) or '://' in str(source):
        raise ValueError('Use a locally uploaded video file, not a network URL, in-memory clip, or generated VIDEO.')
    path = Path(source).resolve()
    if not path.is_file():
        raise ValueError('The uploaded video file is missing.')
    # LoadVideo can carry an active trim. Honor it instead of silently decoding
    # a different part of the file. Direct LoadVideo has start=duration=0.
    window = getattr(video, 'get_active_trim_window', None)
    start, duration = window() if callable(window) else (0., 0.)
    return path, number(start, 'Input trim start', 0, 86400*30), number(duration, 'Input trim duration', 0, 86400*30)


def probe(path):
    result = _run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], timeout=30)
    info = json.loads(result.stdout)
    if not isinstance(info.get('streams'), list):
        raise ValueError('No readable streams in the reference file.')
    return info


def resolve_window(info, start, duration, input_start=0., input_duration=0.):
    start = number(start, 'Start seconds', 0, 86400*30)
    duration = number(duration, 'Duration seconds', .25, MAX_SECONDS)
    total = info.get('format', {}).get('duration')
    try: total = float(total)
    except (TypeError, ValueError):
        candidates = []
        for s in info['streams']:
            try: candidates.append(float(s.get('duration', 0)))
            except (TypeError, ValueError): pass
        total = max(candidates, default=0.)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('The file has no reliable duration. Remux it to a normal MP4 first.')
    available = min(total-input_start, input_duration) if input_duration > 0 else total-input_start
    remaining = available-start
    if remaining < .25:
        raise ValueError('Selected trim starts at or beyond the end of the clip.')
    return input_start+start, min(duration, remaining)


def displayed_dimensions(stream):
    w, h = int(stream['width']), int(stream['height'])
    try: sar = float(Fraction(stream.get('sample_aspect_ratio', '1:1').replace(':','/')))
    except (ValueError, ZeroDivisionError): sar = 1.
    if not math.isfinite(sar) or sar <= 0: sar = 1.
    w = max(1, round(w*sar))
    rotation = 0
    for row in stream.get('side_data_list', []):
        if 'rotation' in row: rotation = float(row['rotation'])
    if not rotation:
        try: rotation = float(stream.get('tags', {}).get('rotate', 0))
        except (ValueError, TypeError): rotation = 0
    if round(rotation/90) % 2: w,h = h,w
    if min(w,h)<16 or max(w/h,h/w)>5 or w*h>100_000_000:
        raise ValueError('Reference video geometry is outside the supported range.')
    if stream.get('color_transfer') in ('smpte2084','arib-std-b67'):
        raise ValueError('This is HDR footage. Convert it to SDR first to avoid incorrect reference colors.')
    return w,h


def _base(path, start):
    return ['ffmpeg','-v','error','-nostdin','-y','-threads','2',
            '-protocol_whitelist','file,pipe','-ss',f'{start:.8f}','-i',str(path)]


def read_frames(path, stream, start, duration, single_frame=False):
    w,h = displayed_dimensions(stream)
    scale = min(1., 2048/min(w,h), math.sqrt(12_000_000/(w*h))) if single_frame else min(1., math.sqrt(MAX_FRAME_PIXELS/(w*h)))
    ow,oh = max(16,round(w*scale)),max(16,round(h*scale))
    filters = f'scale={ow}:{oh}:flags=lanczos,setsar=1'
    if not single_frame: filters += ',fps=24'
    max_frames = 1 if single_frame else min(360, max(1, math.ceil(duration*FPS)))
    with tempfile.TemporaryDirectory(prefix='h3-ref-') as temp:
        raw = Path(temp)/'frames.rgb'
        _run(_base(path,start)+['-t',f'{duration:.8f}','-map',f"0:{stream['index']}",
            '-an','-sn','-dn','-vf',filters,'-frames:v',str(max_frames),
            '-pix_fmt','rgb24','-f','rawvideo',str(raw)])
        size = ow*oh*3
        byte_count = raw.stat().st_size
        if not byte_count or byte_count % size or byte_count > max_frames*size:
            raise ValueError('Reference video did not decode into complete bounded frames.')
        n = byte_count//size
        if not single_frame:
            if n<5: raise ValueError('Motion reference needs at least 5 decoded frames.')
            n = 5 + ((n-5)//17)*17
        array = np.fromfile(raw,dtype=np.uint8,count=n*size).reshape(n,oh,ow,3)
        frames = torch.from_numpy(array).to(torch.float32).div_(255.)
    return frames


def read_audio(path, stream, start, duration):
    # Keep the source sample rate (within a conservative bound), rather than
    # compressing to MP3, normalizing, enhancing or claiming better source quality.
    rate = int(stream.get('sample_rate', 0))
    if not 8000 <= rate <= 192000:
        raise ValueError('Reference audio has an unsupported sample rate.')
    channels = min(2, max(1, int(stream.get('channels',1))))
    with tempfile.TemporaryDirectory(prefix='h3-voice-') as temp:
        raw = Path(temp)/'audio.f32'
        _run(_base(path,start)+['-t',f'{duration:.8f}','-map',f"0:{stream['index']}",
            '-vn','-sn','-dn','-ac',str(channels),'-ar',str(rate),
            '-c:a','pcm_f32le','-f','f32le',str(raw)])
        data = np.fromfile(raw,dtype='<f4')
        if data.size % channels or data.size > math.ceil(duration*rate+8192)*channels:
            raise ValueError('Invalid or unexpectedly long decoded audio.')
        data = data.reshape(-1,channels).T.copy()
        if data.shape[1] < rate//4 or not np.isfinite(data).all():
            raise ValueError('Reference audio is missing, invalid or shorter than 0.25 seconds.')
        if np.max(np.abs(data)) < 1e-6:
            raise ValueError('Selected audio is silent. Select a section with clear speech.')
        return {'waveform':torch.from_numpy(data)[None], 'sample_rate':rate}


def decode_reference(video, start, duration, visual_role, audio_role, identity_position, audio_track):
    path, in_start, in_duration = local_video_source(video)
    info = probe(path)
    start, duration = resolve_window(info,start,duration,in_start,in_duration)
    videos = [s for s in info['streams'] if s.get('codec_type')=='video' and not s.get('disposition',{}).get('attached_pic')]
    audios = [s for s in info['streams'] if s.get('codec_type')=='audio']
    identity = motion = audio = None
    if visual_role != 'Ignore video (voice only)':
        if not videos: raise ValueError('This file has no usable video stream.')
        stream = videos[0]
        if visual_role in ('Identity - selected frame','Identity + action'):
            pos = number(identity_position,'Identity frame position',0,1)
            timestamp = start + min(pos*duration, max(0,duration-1/FPS))
            identity = read_frames(path,stream,timestamp,min(.25,start+duration-timestamp),single_frame=True)
        if visual_role in ('Action / motion','Identity + action'):
            motion = read_frames(path,stream,start,duration)
    if audio_role != 'Ignore audio':
        if type(audio_track) is not int or not 0 <= audio_track < len(audios):
            raise ValueError('Selected audio track does not exist. Choose another track or Ignore audio.')
        audio = read_audio(path,audios[audio_track],start,duration)
    meta = {'filename':path.name,'start_seconds':start,'selected_seconds':duration,
            'audio_track':audio_track if audio is not None else None,
            'video_frames':len(motion) if motion is not None else 0,
            'video_fps':FPS if motion is not None else None,
            'video_seconds':len(motion)/FPS if motion is not None else 0,
            'identity_frame_seconds':timestamp if identity is not None else None,
            'audio_seconds':audio['waveform'].shape[-1]/audio['sample_rate'] if audio is not None else 0,
            'source_audio_is_final_output':False}
    return identity,motion,audio,meta
