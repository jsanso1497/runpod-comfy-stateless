"""Bounded-memory CPU media preparation and finishing for the isolated worker."""
from __future__ import annotations
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
from fractions import Fraction
from .video_process import run_logged

MODES = ('1080p', '4k', '2x3-source')


def probe(path: Path, count: bool = False) -> dict:
    cmd = ['ffprobe', '-v', 'error'] + (['-count_frames'] if count else [])
    cmd += ['-show_streams', '-show_format', '-of', 'json', str(path)]
    return json.loads(subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=120))


def video_stream(info: dict) -> dict:
    return next(s for s in info['streams'] if s.get('codec_type') == 'video')


def source_plan(source: Path, mode: str, start: float, duration: float, info: dict | None = None) -> dict:
    if mode not in MODES or not all(math.isfinite(x) for x in (start, duration)) or start < 0 or duration < 0:
        raise ValueError('Invalid mode or trim. Use finite nonnegative times.')
    info = probe(source) if info is None else info
    v = video_stream(info)
    if v.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('This workflow is SDR. Convert HDR/log explicitly before restoration.')
    if v.get('sample_aspect_ratio', '1:1') not in ('1:1', '0:1', 'N/A'):
        raise ValueError('Non-square pixels require explicit normalization.')
    w, h = int(v['width']), int(v['height'])
    rotation = float(v.get('tags', {}).get('rotate', 0))
    for side in v.get('side_data_list', []):
        rotation = float(side.get('rotation', rotation))
    if rotation % 90:
        raise ValueError('Unsupported rotation; normalize this source first.')
    if round(rotation) % 180:
        w, h = h, w
    rate = v.get('avg_frame_rate', '0/0')
    if rate in ('0/0', '0', 'N/A'):
        rate = v.get('r_frame_rate', '0/0')
    try:
        fps = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        raise ValueError('No usable frame rate') from None
    if not 1 <= fps <= 120:
        raise ValueError('Expected source frame rate from 1 through 120 FPS')
    length = float(v.get('duration') or info.get('format', {}).get('duration') or 0)
    if length <= start:
        raise ValueError('Start time is outside the source video')
    duration = duration or (length - start)
    if start + duration > length + 0.5 / float(fps):
        raise ValueError(f'Requested {duration:g}s at {start:g}s exceeds the {length:g}s source')
    if mode == '2x3-source':
        pw, ph = 4 * (h // 6), 6 * (h // 6)
        if ph < 6 or w < pw:
            raise ValueError('Source is too narrow for a right-side-only 2:3 crop')
        final, restored = [pw, ph], [1080, 1620]
    else:
        if w % 2 or h % 2:
            raise ValueError('Full-frame H.264 requires even dimensions; no silent crop is applied')
        pw, ph = w, h
        short, long = (2160, 3840) if mode == '4k' else (1080, 1920)
        scale = min(short / min(w, h), long / max(w, h))
        restored = [max(2, 2 * round(w * scale / 2)), max(2, 2 * round(h * scale / 2))]
        final = list(restored)
    return dict(source=str(source), mode=mode, start=start, duration=duration,
                source_size=[w, h], prepared_size=[pw, ph], final_size=final,
                restored_size=restored, fps=str(fps),
                source_audio=any(s.get('codec_type') == 'audio' for s in info['streams']),
                frames_estimate=math.ceil(duration * float(fps)),
                timing_note='CFR output at source average FPS; VFR timestamps are normalized')


def storage_required(plan: dict) -> int:
    pixels = math.prod(plan['prepared_size']) + math.prod(plan['restored_size'])
    # Lossless prepared video, PNGs, and one bounded retry. A disk estimate,
    # not a RAM requirement. Failed attempt frames are removed before retry.
    return int(plan['frames_estimate'] * pixels * 4 * 1.25 + 2 * 1024 ** 3)


def prepare(source: Path, job: Path, plan: dict) -> tuple[Path, int]:
    dest = job / 'prepared.mkv'
    filters = ['setsar=1']
    if plan['mode'] == '2x3-source':
        w, h = plan['prepared_size']
        filters.insert(0, f'crop={w}:{h}:0:0')
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'warning', '-y',
           '-threads', '2', '-filter_threads', '1', '-i', str(source),
           '-ss', str(plan['start']), '-t', str(plan['duration']),
           '-map', '0:v:0', '-map', '0:a:0?', '-vf', ','.join(filters),
           '-r', plan['fps'], '-fps_mode', 'cfr', '-c:v', 'ffv1', '-level', '3',
           '-pix_fmt', 'bgr0', '-threads', '2', '-c:a', 'pcm_s16le', str(dest)]
    if run_logged(cmd, job / 'prepare.log'):
        raise RuntimeError('Source preparation failed; see prepare.log')
    info = probe(dest, count=True)
    v = video_stream(info)
    count = int(v['nb_read_frames'])
    if count < 1 or tuple(plan['prepared_size']) != (v['width'], v['height']):
        raise RuntimeError('Prepared video geometry/frame verification failed')
    if abs(count / float(Fraction(plan['fps'])) - plan['duration']) > 1.5 / float(Fraction(plan['fps'])):
        raise RuntimeError('Prepared video duration does not match requested trim')
    return dest, count


def png_size(path: Path) -> tuple[int, int]:
    with path.open('rb') as handle:
        data = handle.read(24)
    if len(data) != 24 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
        raise ValueError('Incomplete PNG: ' + str(path))
    return struct.unpack('>II', data[16:24])


def find_frames(root: Path, count: int) -> tuple[Path, str, int, tuple[int, int]]:
    groups = {}
    for p in root.rglob('*.png'):
        match = re.fullmatch(r'(.*?)(\d+)\.png', p.name)
        if match:
            key = (p.parent, match[1], len(match[2]))
            groups.setdefault(key, []).append((int(match[2]), p))
    valid = []
    for (folder, prefix, digits), entries in groups.items():
        entries.sort()
        if len(entries) != count or [i for i, _ in entries] != list(range(entries[0][0], entries[0][0] + count)):
            continue
        if '%' in prefix:
            raise ValueError('Unsupported sequence filename')
        sizes = {png_size(p) for _, p in entries}
        if len(sizes) != 1:
            raise ValueError('Restored frames have different dimensions')
        valid.append((folder, prefix + '%0' + str(digits) + 'd.png', entries[0][0], sizes.pop()))
    if len(valid) != 1:
        raise ValueError(f'Expected one complete sequence of {count} restored frames; found {len(valid)}')
    return valid[0]


def finish(job: Path, output: Path, plan: dict, count: int) -> Path:
    folder, pattern, first, dimensions = find_frames(job / 'restored_frames', count)
    if dimensions != tuple(plan['restored_size']):
        raise ValueError(f'Restored size {dimensions} differs from expected {plan["restored_size"]}')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Completed output exists; not overwriting it')
    partial = output.with_suffix('.partial.mp4')
    w, h = plan['final_size']
    filters = []
    if plan['mode'] == '2x3-source':
        filters.append(f'scale={w}:{h}:flags=lanczos')
    # The input frames are RGB; explicitly choose the BT.709 YUV matrix.
    filters += ['scale=in_range=full:out_range=tv:out_color_matrix=bt709', 'setsar=1']
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'warning', '-y',
           '-threads', '2', '-filter_threads', '1', '-framerate', plan['fps'],
           '-start_number', str(first), '-i', str(folder / pattern), '-i', str(job / 'prepared.mkv'),
           '-map', '0:v:0', '-map', '1:a:0?', '-vf', ','.join(filters), '-frames:v', str(count),
           '-c:v', 'libx264', '-preset', 'slow', '-crf', str(plan.get('crf', 16)),
           '-pix_fmt', 'yuv420p', '-threads', '2', '-color_primaries', 'bt709',
           '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv',
           '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(partial)]
    if run_logged(cmd, job / 'encode.log'):
        raise RuntimeError('Final encoding failed. Restored frames retained; see encode.log')
    info = probe(partial, count=True)
    v = video_stream(info)
    if (int(v['width']), int(v['height']), int(v['nb_read_frames'])) != (w, h, count):
        raise RuntimeError('Final size/frame-count verification failed')
    if abs(float(Fraction(v['avg_frame_rate']) - Fraction(plan['fps']))) > 1e-5:
        raise RuntimeError('Final FPS verification failed')
    if plan['source_audio'] and not any(s.get('codec_type') == 'audio' for s in info['streams']):
        raise RuntimeError('Final audio verification failed')
    partial.replace(output)
    return output
