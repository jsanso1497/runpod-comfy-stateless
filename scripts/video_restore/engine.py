"""Video-restoration comparison runner. All media operations are local to the Pod."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from fractions import Fraction
from typing import Callable

APP = Path(__file__).resolve().parent
SEED = Path(os.environ.get('SEEDVR2_HOME', '/opt/seedvr2'))
HOME = Path(os.environ.get('RESTORE_HOME', '/workspace/video-restore'))
MANIFEST = Path(os.environ.get('RESTORE_MANIFEST', '/opt/video-restore/models.json'))
SEED_REF = '4490bd1f482e026674543386bb2a4d176da245b9'
ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.m4v', '.avi', '.webm'}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ensure_models(model: str, progress: Callable[[str], None] = print) -> dict:
    import requests
    models = json.loads(MANIFEST.read_text())['models']
    if model not in {'7b', '3b'}:
        raise ValueError('RESTORE_MODEL must be 7b or 3b.')
    target = HOME / 'models' / 'SEEDVR2'
    target.mkdir(parents=True, exist_ok=True)
    for key in (model, 'vae'):
        info = models[key]
        path = target / info['filename']
        progress(f'Checking {path.name}')
        if path.exists() and sha256_file(path) == info['sha256']:
            continue
        if path.exists():
            path.unlink()
        url = f'https://huggingface.co/{info["repo"]}/resolve/main/{info["filename"]}'
        partial = path.with_suffix(path.suffix + '.partial')
        for attempt in range(1, 5):
            try:
                size = partial.stat().st_size if partial.exists() else 0
                headers = {'Range': f'bytes={size}-'} if size else {}
                # These models are public; no access token is required or sent.
                with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    append = size > 0 and response.status_code == 206
                    if append and not response.headers.get('Content-Range', '').startswith(f'bytes {size}-'):
                        raise RuntimeError('Server returned an unexpected download range.')
                    mode = 'ab' if append else 'wb'
                    downloaded = size if append else 0
                    remaining = int(response.headers.get('Content-Length', '0'))
                    if shutil.disk_usage(target).free < remaining + 8 * 2**30:
                        raise RuntimeError('Insufficient free container disk for weights. Use at least 150 GB for this template.')
                    total = downloaded + remaining
                    last = 0.0
                    with partial.open(mode) as output:
                        for chunk in response.iter_content(8 * 1024 * 1024):
                            if not chunk:
                                continue
                            output.write(chunk)
                            downloaded += len(chunk)
                            now = time.monotonic()
                            if now - last > 2:
                                suffix = f' / {total / 2**30:.1f} GiB' if total else ''
                                progress(f'{path.name}: {downloaded / 2**30:.1f} GiB{suffix}')
                                last = now
                progress(f'Verifying SHA-256: {path.name}')
                if sha256_file(partial) != info['sha256']:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError(f'Hash verification failed for {path.name}.')
                partial.replace(path)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt * 3)
    return {'directory': str(target), 'dit': models[model], 'vae': models['vae']}


def probe(path: Path, count: bool = False) -> dict:
    cmd = ['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe']
    if count:
        cmd += ['-count_frames']
    cmd += ['-show_streams', '-show_format', '-of', 'json', str(path)]
    return json.loads(subprocess.check_output(cmd, text=True, timeout=300))


def video_stream(info: dict) -> dict:
    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video' and not stream.get('disposition', {}).get('attached_pic'):
            return stream
    raise ValueError('No video stream was found.')


def frame_count(path: Path) -> int:
    stream = video_stream(probe(path, count=True))
    value = stream.get('nb_read_frames') or stream.get('nb_frames')
    if not str(value).isdigit():
        raise ValueError(f'Cannot verify frame count for {path.name}.')
    return int(value)


def validate_source(path: Path) -> dict:
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError('Use an MP4, MOV, MKV, M4V, AVI or WebM video.')
    info = probe(path)
    s = video_stream(info)
    if (s.get('width'), s.get('height')) != (1920, 1080):
        raise ValueError('This first-test profile expects a 1920 x 1080 landscape clip. Export a 1080p copy first.')
    if s.get('sample_aspect_ratio', '1:1') not in {'1:1', '0:1', 'N/A'}:
        raise ValueError('Anamorphic input is not supported in this test; export square-pixel 1920 x 1080.')
    if s.get('field_order') not in {None, 'unknown', 'progressive'}:
        raise ValueError('Interlaced input detected. Deinterlace before using this progressive-video test.')
    for d in s.get('side_data_list', []):
        if abs(float(d.get('rotation', 0))) > 0.1:
            raise ValueError('Rotated input detected. Export a landscape copy with rotation applied.')
    if s.get('color_transfer') in {'smpte2084', 'arib-std-b67'} or s.get('color_primaries') == 'bt2020':
        raise ValueError('HDR/wide-gamut input detected. This first profile is SDR Rec.709 only; do not silently tone-map it.')
    fps = Fraction(s.get('avg_frame_rate') or s.get('r_frame_rate') or '0/1')
    if not 1 <= float(fps) <= 120:
        raise ValueError('Could not establish a valid source frame rate between 1 and 120 fps.')
    rate = Fraction(s.get('r_frame_rate') or str(fps))
    if abs(float(fps) - float(rate)) > max(0.015, float(fps) * 0.0005):
        raise ValueError('Likely variable-frame-rate input. Export a constant-frame-rate copy at its intended frame rate first.')
    duration = float(s.get('duration') or info.get('format', {}).get('duration', 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Could not read the input duration.')
    warnings = []
    if s.get('color_space') not in {'bt709', 'smpte170m', 'bt470bg', 'gbr', 'rgb'}:
        warnings.append('Input color matrix is not tagged; treating this 1080p SDR clip as Rec.709.')
    if s.get('color_range') not in {'tv', 'pc'}:
        warnings.append('Input range is not tagged; assuming limited-range YUV unless RGB.')
    if '10' in s.get('pix_fmt', '') or '12' in s.get('pix_fmt', ''):
        warnings.append('SeedVR2 standalone video I/O uses 8-bit RGB. This test is not a 10/12-bit mastering pipeline.')
    return {'raw': info, 'stream': s, 'fps': str(fps), 'duration': duration,
            'audio': any(x.get('codec_type') == 'audio' for x in info.get('streams', [])),
            'warnings': warnings}


def verify_cadence(path: Path, start: float, duration: float, fps: Fraction) -> None:
    text = subprocess.check_output([
        'ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-select_streams', 'v:0',
        '-read_intervals', f'{start:.9f}%{start + duration:.9f}', '-show_frames',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'csv=p=0', str(path)
    ], text=True, timeout=180)
    points = []
    for line in text.splitlines():
        try:
            value = float(line.split(',')[0])
            if math.isfinite(value):
                points.append(value)
        except ValueError:
            continue
    diffs = [b - a for a, b in zip(points, points[1:])]
    expected = 1 / float(fps)
    if len(diffs) > 2 and any(abs(d - expected) > max(0.002, expected * 0.04) for d in diffs):
        raise ValueError('Variable or irregular frame timestamps detected in this segment. Use a CFR export first.')


class Cancelled(RuntimeError):
    pass


class Runner:
    def __init__(self, job: Path, cancel: threading.Event, progress: Callable[[str], None]):
        self.job, self.cancel, self.progress = job, cancel, progress
        self.log = job / 'run.log'
        self.commands = []
        self.started = time.monotonic()

    def run(self, cmd: list[str], cwd: Path | None = None) -> None:
        self.commands.append(cmd)
        if self.cancel.is_set():
            raise Cancelled('Job cancelled. The Pod itself is still running and billable.')
        with self.log.open('a') as log:
            log.write('\nCOMMAND: ' + json.dumps(cmd) + '\n')
            log.flush()
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while proc.poll() is None:
                    if self.cancel.is_set():
                        raise Cancelled('Job cancelled. The Pod itself is still running and billable.')
                    if time.monotonic() - self.started > int(os.environ.get('RESTORE_MAX_JOB_SECONDS', '7200')):
                        raise RuntimeError('Job exceeded the configured two-hour process limit. The Pod is still billable.')
                    if shutil.disk_usage(self.job).free < 2 * 2**30:
                        raise RuntimeError('Stopping before the container disk fills. Download results, clear jobs or use a larger disk.')
                    time.sleep(0.2)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                raise
            if proc.returncode:
                tail = self.log.read_text(errors='replace')[-6000:]
                raise RuntimeError(f'Command failed ({proc.returncode}).\n{tail}')

    def ffmpeg(self, args: list[str]) -> None:
        self.run(['ffmpeg', '-hide_banner', '-nostdin', '-y', '-threads', '4',
                  '-filter_threads', '2', '-filter_complex_threads', '2'] + args)


def video_encoding() -> list[str]:
    return ['-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
            '-color_range', 'tv', '-movflags', '+faststart', '-map_metadata', '-1']


def select_profile(hardware: dict, model: str, conservative: bool = False) -> dict:
    vram = hardware.get('vram_gib', 48)
    if conservative:
        return {'batch': 5, 'chunk': 17, 'overlap': 3, 'tile': 512,
                'blocks': 32 if model == '7b' else 28}
    if vram >= 70:
        return {'batch': 9, 'chunk': 33, 'overlap': 3, 'tile': 1024, 'blocks': 0}
    if vram >= 40:
        return {'batch': 5, 'chunk': 25, 'overlap': 3, 'tile': 768, 'blocks': 16 if model == '7b' else 0}
    return {'batch': 5, 'chunk': 17, 'overlap': 3, 'tile': 512,
            'blocks': 32 if model == '7b' else 24}


def seed_command(source: Path, dest: Path, weights: dict, hardware: dict,
                 mode: str, profile: dict, attention: str) -> list[str]:
    resolution = 2160 if mode == 'compare_4k' else 1080
    return [sys.executable, str(SEED / 'inference_cli.py'), str(source),
            '--output', str(dest), '--output_format', 'png',
            '--model_dir', weights['directory'], '--dit_model', weights['dit']['filename'],
            '--resolution', str(resolution), '--max_resolution', str(resolution * 16 // 9),
            '--cuda_device', '0', '--batch_size', str(profile['batch']), '--uniform_batch_size',
            '--chunk_size', str(profile['chunk']), '--temporal_overlap', str(profile['overlap']),
            '--prepend_frames', '0', '--seed', '42', '--color_correction', 'lab',
            '--input_noise_scale', '0', '--latent_noise_scale', '0',
            '--attention_mode', attention, '--dit_offload_device', 'cpu',
            '--vae_offload_device', 'cpu', '--tensor_offload_device', 'cpu',
            '--blocks_to_swap', str(profile['blocks']), '--cache_dit', '--cache_vae',
            '--vae_encode_tiled', '--vae_decode_tiled',
            '--vae_encode_tile_size', str(profile['tile']), '--vae_decode_tile_size', str(profile['tile']),
            '--vae_encode_tile_overlap', '128', '--vae_decode_tile_overlap', '128', '--debug']


def encode_clip(r: Runner, source: Path, output: Path, fps: str, frames: int,
                audio: Path | None, filters: str = '', sequence: bool = False) -> None:
    args = ['-framerate', fps, '-start_number', '0'] if sequence else []
    args += ['-i', str(source)]
    if audio:
        args += ['-i', str(audio)]
    args += ['-map', '0:v:0']
    if audio:
        args += ['-map', '1:a:0', '-c:a', 'aac', '-b:a', '192k']
    else:
        args += ['-an']
    vf = (filters + ',' if filters else '') + 'scale=out_color_matrix=bt709:out_range=tv,setsar=1,format=yuv420p'
    args += ['-vf', vf, '-r', fps, '-frames:v', str(frames),
             '-t', f'{frames / float(Fraction(fps)):.9f}']
    args += video_encoding() + [str(output)]
    r.ffmpeg(args)


def run_test(source: Path, job: Path, options: dict, weights: dict, hardware: dict,
             cancel: threading.Event | None = None, progress: Callable[[str], None] = print) -> dict:
    job.mkdir(parents=True, exist_ok=True)
    r = Runner(job, cancel or threading.Event(), progress)
    mode = options.get('mode', 'compare_4k')
    if mode not in {'compare_4k', 'native_1080', 'cleanup_only'}:
        raise ValueError('Unknown processing mode.')
    shadow = options.get('shadow', 'none')
    if shadow not in {'none', 'gentle'}:
        raise ValueError('Shadow setting must be none or gentle.')
    start = float(options.get('start', 0))
    length = float(options.get('duration', 3))
    if not math.isfinite(start) or not math.isfinite(length) or start < 0 or length <= 0 or length > 120:
        raise ValueError('Use a start >= 0 and a duration greater than 0 and at most 120 seconds.')
    progress('Inspecting input, timing and color metadata')
    meta = validate_source(source)
    fps = Fraction(meta['fps'])
    start_frame = round(start * float(fps))
    start = start_frame / float(fps)
    frames = min(round(length * float(fps)), math.floor((meta['duration'] - start) * float(fps) + 0.01))
    if frames < 5:
        raise ValueError('Select a segment containing at least five frames.')
    length = frames / float(fps)
    verify_cadence(source, start, length, fps)
    # Worst-case uncompressed RGB image-sequence budget, plus intermediates and reserve.
    width, height = (3840, 2160) if mode == 'compare_4k' else (1920, 1080)
    required = frames * width * height * 4 + frames * 1920 * 1080 * 4 + 8 * 2**30
    if shutil.disk_usage(job).free < required:
        raise ValueError(f'Use a shorter segment or more container disk; this job reserves about {required / 2**30:.1f} GiB.')
    report = {'status': 'running', 'engine_commit': SEED_REF, 'options': options,
              'source': meta, 'effective_start_seconds': start, 'frames': frames,
              'fps': str(fps), 'duration_seconds': length, 'hardware': hardware,
              'weights': weights, 'warnings': list(meta['warnings']),
              'precision_note': '8-bit RGB media I/O; FP16 weights, BF16 compute where supported. Not HDR mastering.',
              'outputs': [], 'checks': {}}
    (job / 'report.json').write_text(json.dumps(report, indent=2))
    progress(f'Preparing {frames} frames at {fps} fps without resizing or frame interpolation')
    original = job / 'source.mkv'
    s = meta['stream']
    matrix = 'bt601' if s.get('color_space') in {'smpte170m', 'bt470bg'} else 'bt709'
    rgb = s.get('pix_fmt', '').startswith(('rgb', 'bgr', 'gbr'))
    in_range = s.get('color_range') if s.get('color_range') in {'tv', 'pc'} else ('pc' if rgb else 'tv')
    normalize = f'scale=in_color_matrix={matrix}:in_range={in_range}:out_range=pc,setsar=1,format=bgr0'
    r.ffmpeg(['-ss', f'{start:.9f}', '-protocol_whitelist', 'file,pipe', '-i', str(source),
              '-map', '0:v:0', '-an', '-sn', '-dn', '-frames:v', str(frames), '-r', str(fps),
              '-vf', normalize, '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', str(original)])
    if frame_count(original) != frames:
        raise RuntimeError('Input extraction did not preserve the requested frame count. No AI render was started.')
    audio = job / 'audio.wav' if meta['audio'] else None
    if audio:
        r.ffmpeg(['-ss', f'{start:.9f}', '-protocol_whitelist', 'file,pipe', '-i', str(source),
                  '-t', f'{length:.9f}', '-map', '0:a:0', '-vn',
                  '-af', 'aresample=async=1:first_pts=0', '-c:a', 'pcm_s24le', str(audio)])
    input_file = job / 'input.mkv'
    if shadow == 'gentle':
        r.ffmpeg(['-i', str(original), '-vf', 'eq=gamma=1.06:gamma_weight=0.5,format=bgr0',
                  '-an', '-c:v', 'ffv1', '-level', '3', str(input_file)])
        report['warnings'].append('Gentle gamma lift is an intentional global tonal change, not recovered exposure data.')
    else:
        os.link(original, input_file)
    encode_clip(r, original, job / '01_Original_1080p.mp4', str(fps), frames, audio)
    progress('Rendering the non-generative cleanup reference')
    encode_clip(r, input_file, job / '02_Cleanup_1080p.mp4', str(fps), frames, audio,
                'hqdn3d=0.8:0.8:1.2:1.2,unsharp=5:5:0.25:5:5:0.0')
    outputs = ['01_Original_1080p.mp4', '02_Cleanup_1080p.mp4']
    if mode != 'cleanup_only':
        model = os.environ.get('RESTORE_MODEL', '7b')
        profile = select_profile(hardware, model)
        attention = hardware.get('attention', 'sdpa')
        raw = job / 'raw_seedvr2'
        progress(f'Running SeedVR2 {model.upper()}, {width} x {height}, {attention}; this is the GPU-intensive step')
        command = seed_command(input_file, raw, weights, hardware, mode, profile, attention)
        try:
            r.run(command, cwd=SEED)
        except RuntimeError as exc:
            if isinstance(exc, Cancelled):
                raise
            message = str(exc).lower()
            oom = any(x in message for x in ['cuda out of memory', 'outofmemoryerror', 'cuda error: out of memory'])
            kernel = attention != 'sdpa' and any(x in message for x in ['sageattn', 'sageattention', 'no kernel image', 'invalid device function'])
            if not oom and not kernel:
                raise
            # One clean retry only. Never conceal which settings actually succeeded.
            shutil.rmtree(raw, ignore_errors=True)
            if oom:
                profile = select_profile(hardware, model, conservative=True)
                report['warnings'].append('CUDA OOM: retried once with smaller batches/tiles and additional CPU offload.')
            if kernel:
                attention = 'sdpa'
                report['warnings'].append('Sage runtime error: retried once with PyTorch SDPA.')
            progress(report['warnings'][-1])
            r.run(seed_command(input_file, raw, weights, hardware, mode, profile, attention), cwd=SEED)
        report['effective_profile'] = profile
        report['effective_attention'] = attention
        files = sorted(raw.rglob('*.png'))
        if len(files) != frames:
            raise RuntimeError(f'SeedVR2 wrote {len(files)} frames; expected {frames}. Refusing to hide a dropped/added-frame error.')
        from PIL import Image
        for image_path in files:
            with Image.open(image_path) as image:
                if image.size != (width, height):
                    raise RuntimeError('SeedVR2 output resolution does not match the requested dimensions.')
        ordered = job / 'ordered'
        ordered.mkdir(exist_ok=True)
        for i, path in enumerate(files):
            os.link(path, ordered / f'{i:06d}.png')
        sequence = ordered / '%06d.png'
        progress('Encoding deliverables, restoring audio and checking output metadata')
        if mode == 'compare_4k':
            encode_clip(r, sequence, job / '04_AI_4K.mp4', str(fps), frames, audio, sequence=True)
            ai_name = '03_AI_1080p_from_4K.mp4'
            encode_clip(r, sequence, job / ai_name, str(fps), frames, audio,
                        'scale=1920:1080:flags=lanczos', sequence=True)
            outputs += [ai_name, '04_AI_4K.mp4']
            report['note'] = 'The 1080p AI file is downsampled from the same 4K restoration, not a second independent AI pass.'
        else:
            ai_name = '03_AI_Native_1080p.mp4'
            encode_clip(r, sequence, job / ai_name, str(fps), frames, audio, sequence=True)
            outputs += [ai_name]
        # Full-resolution side-by-side: each half retains the original 1920x1080 display size.
        compare = job / '05_Original_vs_AI_3840x1080.mp4'
        args = ['-i', str(job / outputs[0]), '-i', str(job / ai_name),
                '-filter_complex', '[0:v][1:v]hstack=inputs=2,setsar=1[v]', '-map', '[v]']
        args += ['-map', '0:a:0?', '-c:a', 'copy', '-frames:v', str(frames)]
        r.ffmpeg(args + video_encoding() + [str(compare)])
        outputs.append(compare.name)
        # Lossless detail comparison before final delivery compression.
        mid = frames // 2
        detail = job / '_original_detail.png'
        r.ffmpeg(['-i', str(original), '-vf', f'select=eq(n\\,{mid}),crop=640:360',
                  '-frames:v', '1', str(detail)])
        before = Image.open(detail).convert('RGB')
        after = Image.open(files[mid]).convert('RGB')
        if mode == 'compare_4k':
            after = after.resize((1920, 1080), Image.Resampling.LANCZOS)
        after = after.crop((640, 360, 1280, 720))
        from PIL import ImageDraw
        canvas = Image.new('RGB', (1280, 392), (22, 22, 22))
        canvas.paste(before, (0, 32)); canvas.paste(after, (640, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), 'ORIGINAL / center crop at 100%', fill='white')
        draw.text((652, 10), 'AI / matched 1080p crop at 100%', fill='white')
        canvas.save(job / '06_Detail_comparison.png')
        outputs.append('06_Detail_comparison.png')
        detail.unlink(missing_ok=True)
    progress('Verifying frame counts, rate, dimensions, square pixels and audio')
    for name in outputs:
        if not name.endswith('.mp4'):
            continue
        path = job / name
        data = probe(path, count=True)
        stream = video_stream(data)
        actual_frames = int(stream.get('nb_read_frames', '0'))
        rate = Fraction(stream.get('avg_frame_rate', '0/1'))
        audio_present = any(x.get('codec_type') == 'audio' for x in data['streams'])
        checks = {'frames': actual_frames, 'fps': str(rate),
                  'width': stream['width'], 'height': stream['height'],
                  'sample_aspect_ratio': stream.get('sample_aspect_ratio'),
                  'color_space': stream.get('color_space'), 'audio_present': audio_present}
        expected_size = (3840, 2160) if name == '04_AI_4K.mp4' else ((3840, 1080) if name.startswith('05_') else (1920, 1080))
        if (stream['width'], stream['height']) != expected_size:
            raise RuntimeError(f'Output resolution validation failed: {name}')
        report['checks'][name] = checks
        if actual_frames != frames or abs(float(rate - fps)) > 1e-5:
            raise RuntimeError(f'Frame count or frame rate validation failed: {name}')
        if stream.get('sample_aspect_ratio') != '1:1' or audio_present != meta['audio']:
            raise RuntimeError(f'Pixel-aspect or audio-presence validation failed: {name}')
    report.update(status='complete', elapsed_seconds=round(time.monotonic() - r.started, 2),
                  commands=r.commands, outputs=outputs + ['report.json', 'run.log'])
    (job / 'report.json').write_text(json.dumps(report, indent=2))
    # Intermediate PNGs and lossless files are temporary. Keep logs and deliverables only.
    for name in ['raw_seedvr2', 'ordered']:
        shutil.rmtree(job / name, ignore_errors=True)
    for path in [original, input_file, audio]:
        if path:
            path.unlink(missing_ok=True)
    progress('Complete. Download the results before terminating the Pod.')
    return report
