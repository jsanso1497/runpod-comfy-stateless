"""CPU-only, bounded FFmpeg exports: exactly 124 frames per H3 source clip.

No H3, Ollama, torch, full-video tensor, or remote downloads are used here.
Each source interval is independently sought and transcoded, never stream-copied.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile

FPS = 24
FRAMES = 124
DURATION = Fraction(FRAMES, FPS)
AUDIO_RATE = 48000
AUDIO_SAMPLES = FRAMES * AUDIO_RATE // FPS
MAX_CLIPS = 1000
RESERVE_BYTES = 1024**3
MODES = ('Selected chunk', 'Consecutive chunks')
TAIL_POLICIES = ('Skip incomplete tail', 'Hold last frame to 124')
AUDIO_MODES = ('Keep source audio', 'Silent')
RELEASE = '2026-10-22.2'


def upload_cli_args(environ=None):
    env = os.environ if environ is None else environ
    raw = env.get('WB_H3_MAX_UPLOAD_MB', '1024')
    if not isinstance(raw, str) or not re.fullmatch(r'[0-9]{1,4}', raw) or not 1 <= int(raw) <= 1900:
        raise ValueError('WB_H3_MAX_UPLOAD_MB must be an integer from 1 to 1900. Default: 1024 MiB. The gateway is capped at 2 GiB.')
    return ['--max-upload-size', str(int(raw))]


def rational(value, name, maximum=2592000):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Fraction)):
        raise ValueError(name + ' must be a finite nonnegative number.')
    try:
        f = Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(name + ' must be a finite nonnegative number.') from exc
    if not 0 <= f <= maximum:
        raise ValueError(f'{name} must be between 0 and {maximum}.')
    return f


def seconds_arg(value):
    # Only the ffmpeg command uses a decimal; all segment arithmetic is rational.
    return f'{float(value):.12f}'


def safe_child(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or '\\' in relative or '\x00' in relative:
        raise ValueError('Invalid relative storage path.')
    parts = relative.split('/')
    if any(p in ('', '.', '..') for p in parts):
        raise ValueError('Storage path must remain inside its root.')
    target = root
    for part in parts:
        target = target / part
        if target.is_symlink():
            raise ValueError('Symlinked media/export paths are not accepted.')
    if not target.resolve().is_relative_to(root):
        raise ValueError('Storage path escapes its root.')
    return target


def resolve_video(video, input_root):
    getter = getattr(video, 'get_stream_source', None)
    if not callable(getter):
        raise ValueError('Connect the native Load Video node directly, not a decoded image batch.')
    raw = getter()
    if not isinstance(raw, (str, Path)) or '://' in str(raw):
        raise ValueError('Use an uploaded local video, not a URL or in-memory video.')
    root = Path(input_root).resolve()
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError('Source must be inside the ComfyUI input directory.') from exc
    path = safe_child(root, relative)
    if not path.is_file():
        raise ValueError('The source video is missing from input storage.')
    active = getattr(video, 'get_active_trim_window', None)
    start, duration = active() if callable(active) else (0, 0)
    return path, rational(start, 'Input trim start'), rational(duration, 'Input trim duration')


def run_media(command, *, check_cancel=None, timeout=1200, disk_root=None, capture=False):
    """Poll for cancellation; stop the child cleanly. stderr never fills a pipe."""
    if check_cancel:
        check_cancel()
    with tempfile.TemporaryFile() as errors, tempfile.TemporaryFile() as output:
        try:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                    stdout=output if capture else subprocess.DEVNULL, stderr=errors)
        except FileNotFoundError as exc:
            raise RuntimeError('FFmpeg and ffprobe must be present in the H3 container.') from exc
        start = time.monotonic()
        try:
            while proc.poll() is None:
                if check_cancel:
                    check_cancel()
                if time.monotonic() - start > timeout:
                    raise TimeoutError('Media processing timed out. Completed clips remain available; inspect run_status.json.')
                if disk_root is not None and shutil.disk_usage(disk_root).free < RESERVE_BYTES:
                    raise RuntimeError('Export stopped to preserve 1 GiB of free storage. Source and completed clips are unchanged.')
                time.sleep(.1)
            if proc.returncode:
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 3000))
                raise RuntimeError('FFmpeg/ffprobe failed: ' + errors.read().decode('utf-8', 'replace'))
            if capture:
                output.seek(0)
                return output.read().decode('utf-8')
            return ''
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            raise


def probe(path, *, count_frames=False, check_cancel=None):
    command = ['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe']
    if count_frames:
        command.append('-count_frames')
    command += ['-show_streams', '-show_format', '-of', 'json', str(path)]
    return json.loads(run_media(command, timeout=120, check_cancel=check_cancel, capture=True))


def describe_source(metadata):
    videos = [s for s in metadata.get('streams', [])
              if s.get('codec_type') == 'video' and not s.get('disposition', {}).get('attached_pic')]
    if not videos:
        raise ValueError('No video stream was found.')
    video = videos[0]
    if video.get('color_transfer') in ('smpte2084', 'arib-std-b67') or video.get('color_primaries') == 'bt2020':
        raise ValueError('HDR/wide-gamut source detected. Convert to SDR before this H3 preparation step; no silent HDR-to-SDR conversion is performed.')
    try:
        width, height = int(video['width']), int(video['height'])
        sar = Fraction(str(video.get('sample_aspect_ratio', '1:1')).replace(':', '/'))
    except (ValueError, ZeroDivisionError):
        width, height = int(video['width']), int(video['height'])
        sar = Fraction(1)
    if min(width, height) < 16 or width * height > 100_000_000 or not 0 < sar <= 10:
        raise ValueError('Unsupported video dimensions or pixel aspect ratio.')
    width = round(width * sar)
    rotation = video.get('tags', {}).get('rotate', 0)
    for side in video.get('side_data_list', []):
        if 'rotation' in side:
            rotation = side['rotation']
    rotation = float(rotation)
    if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > .001:
        raise ValueError('Only normal right-angle video rotation is supported.')
    if round(rotation / 90) % 2:
        width, height = height, width
    width, height = max(2, round(width / 2) * 2), max(2, round(height / 2) * 2)
    if max(width, height) > 16384:
        raise ValueError('Video dimensions exceed the export safety limit.')
    duration = None
    # Prefer the video stream duration: audio can end earlier or later.
    candidates = [video.get('duration'), metadata.get('format', {}).get('duration')]
    for candidate in candidates:
        try:
            parsed = rational(candidate, 'Source duration')
            if parsed > 0:
                duration = parsed
                break
        except ValueError:
            pass
    if duration is None:
        raise ValueError('The file has no usable duration. Remux it to a regular local MP4 first.')
    audios = [s for s in metadata.get('streams', []) if s.get('codec_type') == 'audio']
    return {'video_index': int(video['index']), 'audio_index': int(audios[0]['index']) if audios else None,
            'duration': duration, 'width': width, 'height': height, 'rotation': rotation,
            'source_fps': video.get('avg_frame_rate', 'unknown')}


@dataclass(frozen=True)
class Clip:
    number: int
    start: Fraction
    available: Fraction
    padding: bool


def plan_clips(source_duration, start_seconds=0, first_chunk=1, chunk_count=0,
               mode=MODES[0], final_chunk=TAIL_POLICIES[0], input_start=0, input_duration=0):
    if mode not in MODES or final_chunk not in TAIL_POLICIES:
        raise ValueError('Invalid export mode or final-chunk policy.')
    if type(first_chunk) is not int or first_chunk < 1 or type(chunk_count) is not int or not 0 <= chunk_count <= MAX_CLIPS:
        raise ValueError('First chunk is 1-based; chunk count must be 0 (all remaining) through 1000.')
    duration = rational(source_duration, 'Source duration')
    trim_start = rational(input_start, 'Input trim start')
    trim_duration = rational(input_duration, 'Input trim duration')
    end = min(duration, trim_start + trim_duration) if trim_duration else duration
    origin = trim_start + rational(start_seconds, 'Start seconds')
    begin = origin + (first_chunk - 1) * DURATION
    remaining = end - begin
    if remaining <= 0:
        raise ValueError('The selected start/chunk is at or past the end of the video.')
    # Probe duration is commonly rounded to six decimals. Sub-microsecond fuzz
    # prevents an exact 124-frame input being rejected as 123.99999 frames.
    full = int((remaining + Fraction(1, 1_000_000)) // DURATION)
    leftover = max(Fraction(0), remaining - full * DURATION)
    tail_available = leftover >= Fraction(1, FPS)
    total = full + int(final_chunk == TAIL_POLICIES[1] and tail_available)
    wanted = 1 if mode == MODES[0] else (chunk_count or total)
    count = min(wanted, total)
    if count == 0:
        raise ValueError('Less than one complete 124-frame chunk remains. Choose Hold last frame to 124 explicitly or select an earlier start.')
    if count > MAX_CLIPS:
        raise ValueError('This run would export over 1000 clips. Set a count of 1000 or fewer, then continue with a later first_chunk.')
    clips = []
    for i in range(count):
        start = begin + i * DURATION
        avail = min(DURATION, end - start)
        padded = avail + Fraction(1, 1_000_000) < DURATION
        clips.append(Clip(first_chunk + i, start, avail, padded))
    return clips, {'available_full_chunks': full, 'requested_count': wanted,
                   'export_count': count, 'unexported_source_seconds': float(max(Fraction(0), end - clips[-1].start - clips[-1].available)),
                   'incomplete_tail_seconds': float(leftover), 'origin_seconds': float(origin),
                   'final_chunk_policy': final_chunk}


def export_command(source, destination, info, clip, keep_audio, crf=16):
    if type(crf) is not int or not 0 <= crf <= 30:
        raise ValueError('CRF must be an integer from 0 through 30.')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
               '-threads', '2', '-filter_threads', '2', '-protocol_whitelist', 'file,pipe',
               '-ss', seconds_arg(clip.start), '-t', seconds_arg(clip.available), '-i', str(source),
               '-map', f"0:{info['video_index']}"]
    filters = (f"fps=fps=24:start_time=0:round=near,scale={info['width']}:{info['height']}:flags=lanczos,setsar=1")
    if clip.padding:
        filters += ',tpad=stop_mode=clone:stop_duration=' + seconds_arg(DURATION)
    filters += ',trim=end_frame=124,setpts=N/(24*TB)'
    command += ['-vf', filters, '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
                '-pix_fmt', 'yuv420p', '-threads', '2', '-r', '24', '-vsync', 'cfr',
                '-frames:v', '124', '-t', seconds_arg(DURATION)]
    if keep_audio and info['audio_index'] is not None:
        command += ['-map', f"0:{info['audio_index']}", '-af',
                    f'aresample=48000:async=1:first_pts=0,apad,atrim=end_sample={AUDIO_SAMPLES}',
                    '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2']
    else:
        command += ['-an']
    command += ['-sn', '-dn', '-map_metadata', '-1', '-map_chapters', '-1',
                '-metadata:s:v:0', 'rotate=0', '-movflags', '+faststart', '-f', 'mp4', str(destination)]
    return command


def verify_clip(path, *, check_cancel=None):
    data = probe(path, count_frames=True, check_cancel=check_cancel)
    videos = [s for s in data['streams'] if s.get('codec_type') == 'video']
    if len(videos) != 1:
        raise ValueError('Export must contain exactly one video stream.')
    stream = videos[0]
    if int(stream.get('nb_read_frames', 0)) != FRAMES or Fraction(stream.get('avg_frame_rate', '0/1')) != FPS:
        raise ValueError('Export did not produce exactly 124 frames at 24 fps. It was not published as an H3-ready clip.')
    if abs(float(stream.get('duration', 0)) - float(DURATION)) > .002:
        raise ValueError('Export has an unexpected video duration.')
    return {'frames': FRAMES, 'fps': FPS, 'video_seconds': float(DURATION),
            'width': int(stream['width']), 'height': int(stream['height']),
            'audio_present': any(s.get('codec_type') == 'audio' for s in data['streams'])}


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    os.replace(temp, path)


def publish_input(source, input_root, name):
    target = safe_child(input_root, name)
    if target.exists():
        raise ValueError('Refusing to overwrite an existing input file.')
    try:
        os.link(source, target)
        return name, 'hardlink'
    except OSError:
        # Different filesystems: copy, never symlink. Native LoadVideo lists
        # top-level input files; use a unique top-level name for compatibility.
        if shutil.disk_usage(input_root).free < source.stat().st_size + RESERVE_BYTES:
            raise RuntimeError('Insufficient input storage to publish the completed clip.')
        with source.open('rb') as src, target.open('xb') as dest:
            try:
                shutil.copyfileobj(src, dest, 1024**2)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
        return name, 'copy'


def export_clips(source, output_root, input_root, *, start_seconds=0.0, first_chunk=1,
                 chunk_count=0, mode=MODES[0], final_chunk=TAIL_POLICIES[0],
                 audio=AUDIO_MODES[0], output_prefix='H3_clips', make_loadable=True,
                 make_zip=True, crf=16, input_start=0, input_duration=0,
                 check_cancel=None, progress=None):
    if audio not in AUDIO_MODES:
        raise ValueError('Choose Keep source audio or Silent.')
    if not isinstance(output_prefix, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', output_prefix):
        raise ValueError('Output prefix: 1-48 letters, digits, underscores or hyphens; begin with a letter or digit.')
    if type(make_loadable) is not bool or type(make_zip) is not bool:
        raise ValueError('Export switches must be boolean.')
    source, output_root, input_root = Path(source), Path(output_root).resolve(), Path(input_root).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError('Source must be an existing regular file.')
    info = describe_source(probe(source, check_cancel=check_cancel))
    clips, planning = plan_clips(info['duration'], start_seconds, first_chunk, chunk_count, mode,
                                 final_chunk, input_start, input_duration)
    root = safe_child(output_root, 'h3_clips')
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < RESERVE_BYTES:
        raise RuntimeError('At least 1 GiB of free output storage is required.')
    run_id = output_prefix + '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    folder = safe_child(root, run_id)
    folder.mkdir()
    report = {'release': RELEASE, 'state': 'exporting', 'source_file': source.name,
              'source_seconds': float(info['duration']), 'source_fps': info['source_fps'],
              'output_size': [info['width'], info['height']], 'frames_per_clip': FRAMES,
              'fps': FPS, 'seconds_per_clip_exact': '31/6', 'seconds_per_clip': float(DURATION),
              'quality': f'H.264 CRF {crf}, medium preset, 8-bit SDR; re-encoded, not lossless',
              'audio': audio, 'directory': str(folder.relative_to(output_root)),
              'planning': planning, 'clips': [], 'zip': None,
              'source_unchanged': True, 'gpu_models_loaded_by_this_tool': False}
    status = folder / 'run_status.json'
    atomic_json(status, report)
    if progress:
        progress(0, len(clips))
    try:
        for i, clip in enumerate(clips, 1):
            if check_cancel:
                check_cancel()
            name = f'chunk_{clip.number:05d}_124f.mp4'
            final = folder / name
            partial = folder / ('.' + name + '.part')
            print(f'[H3 CLIPS] {i}/{len(clips)}: source {float(clip.start):.6f}s; 124 frames; tail_pad={clip.padding}', flush=True)
            run_media(export_command(source, partial, info, clip, audio == AUDIO_MODES[0], crf),
                      check_cancel=check_cancel, disk_root=folder)
            verified = verify_clip(partial, check_cancel=check_cancel)
            os.replace(partial, final)
            row = {'chunk': clip.number, 'file': name, 'source_start_seconds': float(clip.start),
                   'source_start_exact': str(clip.start), 'source_end_seconds': float(clip.start + clip.available),
                   'hold_last_frame_seconds': float(max(Fraction(0), DURATION - clip.available)),
                   'tail_padded': clip.padding, 'bytes': final.stat().st_size, **verified}
            report['clips'].append(row)
            if make_loadable:
                alias = f'h3clip_{run_id}_{clip.number:05d}.mp4'
                row['input_file'], row['input_storage'] = publish_input(final, input_root, alias)
            atomic_json(status, report)
            if progress:
                progress(i, len(clips))
        csv_name = folder / 'clip_manifest.csv'
        with csv_name.open('w', newline='', encoding='utf-8') as handle:
            columns = ['chunk', 'file', 'source_start_seconds', 'source_start_exact', 'source_end_seconds',
                       'frames', 'fps', 'video_seconds', 'tail_padded', 'hold_last_frame_seconds', 'input_file']
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report['clips'])
        report['state'] = 'complete'
        report['manifest'] = str(csv_name.relative_to(output_root))
        atomic_json(status, report)
        if make_zip and len(clips) > 1:
            total_bytes = sum((folder / row['file']).stat().st_size for row in report['clips'])
            if shutil.disk_usage(folder).free < total_bytes + RESERVE_BYTES:
                raise RuntimeError('Clips are complete, but there is not enough extra space for the download ZIP. Use the individual MP4s.')
            archive = folder / (run_id + '.zip')
            temp_zip = archive.with_suffix('.zip.part')
            with zipfile.ZipFile(temp_zip, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
                for row in report['clips']:
                    if check_cancel:
                        check_cancel()
                    bundle.write(folder / row['file'], row['file'])
                bundle.write(csv_name, csv_name.name)
                bundle.write(status, 'run_status.json')
            os.replace(temp_zip, archive)
            report['zip'] = str(archive.relative_to(output_root))
        atomic_json(status, report)
        print(f"[H3 CLIPS] COMPLETE: {len(report['clips'])} verified clips in {folder}", flush=True)
        return report
    except BaseException as exc:
        report['state'] = 'incomplete'
        report['failure'] = str(exc) or type(exc).__name__
        atomic_json(status, report)
        for part in folder.glob('*.part'):
            part.unlink(missing_ok=True)
        # Includes hidden .mp4.part files; completed MP4s and inputs are preserved.
        for part in folder.glob('.*.part'):
            part.unlink(missing_ok=True)
        raise
