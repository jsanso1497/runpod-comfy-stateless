"""Local, 16-bit RGB FFV1 masters. No network calls or generative processing."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from fractions import Fraction
from typing import Callable


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f'{name} is required. Rebuild the H3 image with ffmpeg installed.')
    return path


def _positive_fps(value: float) -> Fraction:
    value = float(value)
    if not math.isfinite(value) or not 1 <= value <= 120:
        raise ValueError('FPS must be a finite value from 1 to 120.')
    return Fraction(str(value)).limit_denominator(100000)


def _array(value):
    if hasattr(value, 'detach'):
        return value.detach().to(device='cpu').float().numpy()
    return value


def save_master(images, destination: Path, fps: float = 24.0,
                audio=None, interrupt: Callable[[], None] = lambda: None) -> dict:
    """Losslessly encode RGB rounded to 16 bits, not losslessly store float tensors."""
    import numpy as np

    destination = Path(destination)
    if destination.suffix.lower() != '.mkv':
        raise ValueError('A lossless H3 master must have an .mkv extension.')
    if len(images.shape) != 4 or images.shape[0] < 1 or images.shape[-1] != 3:
        raise ValueError('Expected a nonempty RGB image batch [frames, height, width, 3].')
    count, height, width, _ = map(int, images.shape)
    if height < 1 or width < 1:
        raise ValueError('Image dimensions must be positive.')
    rate = _positive_fps(fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Refusing to overwrite an existing master.')

    with tempfile.TemporaryDirectory(prefix='.h3-master-', dir=destination.parent) as temp:
        temp = Path(temp)
        staged = temp / 'master.mkv'
        command = [_tool('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin',
                   '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s:v', f'{width}x{height}',
                   '-r', f'{rate.numerator}/{rate.denominator}', '-i', 'pipe:0']
        has_audio = audio is not None
        if has_audio:
            waveform = np.asarray(_array(audio['waveform']), dtype=np.float32)
            if waveform.ndim != 3 or waveform.shape[0] != 1 or not 1 <= waveform.shape[1] <= 8:
                raise ValueError('Audio must have shape [1, channels, samples], with 1 to 8 channels.')
            if not np.isfinite(waveform).all() or waveform.shape[2] < 1:
                raise ValueError('Audio is empty or contains non-finite values.')
            sample_rate = int(audio['sample_rate'])
            if not 8000 <= sample_rate <= 384000:
                raise ValueError('Unsupported audio sample rate.')
            raw_audio = temp / 'audio.f32le'
            np.asarray(waveform[0].T, dtype='<f4').tofile(raw_audio)
            command += ['-f', 'f32le', '-ar', str(sample_rate), '-ac', str(waveform.shape[1]),
                        '-i', str(raw_audio)]
        command += ['-map', '0:v:0']
        if has_audio:
            command += ['-map', '1:a:0', '-c:a', 'pcm_f32le']
        command += ['-c:v', 'ffv1', '-level', '3', '-coder', '1', '-context', '1',
                    '-pix_fmt', 'gbrp16le', '-color_range', 'pc',
                    '-color_primaries', 'bt709', '-color_trc', 'iec61966-2-1',
                    '-colorspace', 'rgb', '-metadata', 'title=H3 RGB16 lossless master',
                    '-y', str(staged)]
        with (temp / 'ffmpeg.log').open('w+b') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for frame in images:
                    interrupt()
                    array = np.asarray(_array(frame), dtype=np.float32)
                    if not np.isfinite(array).all():
                        raise ValueError('Video contains non-finite pixels.')
                    packed = np.ascontiguousarray(np.rint(np.clip(array, 0, 1) * 65535), dtype='<u2')
                    process.stdin.write(packed.tobytes())
                process.stdin.close()
                code = process.wait()
                if code:
                    log.seek(0)
                    raise RuntimeError('FFV1 encoding failed: ' + log.read().decode(errors='replace')[-4000:])
                interrupt()
                # Same-filesystem move; destination is unique per execution.
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError('The master destination was created during encoding.')
                os.replace(staged, destination)
            except BaseException as exc:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if isinstance(exc, BrokenPipeError):
                    log.seek(0)
                    raise RuntimeError('FFV1 encoding stopped: ' + log.read().decode(errors='replace')[-4000:]) from exc
                raise
            finally:
                if process.stdin and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
    return {'path': str(destination), 'width': width, 'height': height,
            'frames': count, 'fps': float(rate), 'audio': has_audio,
            'video_encoding': 'FFV1 RGB16; float RGB rounded to 16-bit before lossless encoding'}


def read_native_master(path: Path, interrupt: Callable[[], None] = lambda: None):
    """Decode an uncompressed-quality native master without an 8-bit intermediate."""
    import numpy as np
    import torch

    path = Path(path)
    if not path.is_file() or path.suffix.lower() != '.mkv':
        raise ValueError('Select an existing native .mkv master, not the MP4 preview.')
    probe = subprocess.run([_tool('ffprobe'), '-v', 'error', '-show_streams', '-of', 'json', str(path)],
                           check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout)['streams']
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    if not video or video.get('codec_name') != 'ffv1':
        raise ValueError('This finishing workflow expects the FFV1 native master created by HFL.')
    width, height = int(video['width']), int(video['height'])
    if width * height > 2_100_000:
        raise ValueError('Select the native H3 master, not an already upscaled 4K master.')
    fps_string = video.get('avg_frame_rate', '0/0')
    if fps_string in ('0/0', '0'):
        fps_string = video.get('r_frame_rate', '24/1')
    fps = float(Fraction(fps_string))
    _positive_fps(fps)
    frame_bytes = width * height * 3 * 2
    frames = []
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([_tool('ffmpeg'), '-v', 'error', '-nostdin', '-i', str(path),
                                    '-map', '0:v:0', '-vsync', '0', '-f', 'rawvideo',
                                    '-pix_fmt', 'rgb48le', 'pipe:1'],
                                   stdout=subprocess.PIPE, stderr=log)
        try:
            while True:
                interrupt()
                chunks = bytearray()
                while len(chunks) < frame_bytes:
                    chunk = process.stdout.read(frame_bytes - len(chunks))
                    if not chunk:
                        break
                    chunks.extend(chunk)
                if not chunks:
                    break
                if len(chunks) != frame_bytes:
                    raise RuntimeError('The master ended in the middle of a video frame.')
                if len(frames) >= 362:
                    raise ValueError('HFL finishing supports native H3 clips of at most 362 frames.')
                frame = np.frombuffer(chunks, dtype='<u2').reshape(height, width, 3)
                frames.append(frame.astype(np.float32) / 65535.0)
            code = process.wait()
            if code:
                log.seek(0)
                raise RuntimeError('Master decode failed: ' + log.read().decode(errors='replace')[-4000:])
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if not frames:
        raise ValueError('The master contains no decodable frames.')
    images = torch.from_numpy(np.stack(frames))
    audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)
    audio = None
    if audio_stream:
        interrupt()
        sample_rate = int(audio_stream['sample_rate'])
        channels = int(audio_stream['channels'])
        raw = subprocess.run([_tool('ffmpeg'), '-v', 'error', '-nostdin', '-i', str(path),
                              '-map', '0:a:0', '-f', 'f32le', '-c:a', 'pcm_f32le', 'pipe:1'],
                             check=True, capture_output=True).stdout
        samples = np.frombuffer(raw, dtype='<f4').reshape(-1, channels).T.copy()
        audio = {'waveform': torch.from_numpy(samples).unsqueeze(0), 'sample_rate': sample_rate}
    return images, audio, fps
