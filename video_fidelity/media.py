"""Explicit SDR media contracts. Never silently crop, stretch, or normalize VFR."""
from __future__ import annotations
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import subprocess
from fv_common import run_logged
from ffmpeg_compat import video_sync_args


def probe(path: Path, count: bool = False) -> dict:
    command = ['ffprobe', '-v', 'error'] + (['-count_frames'] if count else [])
    command += ['-show_streams', '-show_format', '-of', 'json', str(path)]
    return json.loads(subprocess.check_output(command, timeout=600))


def video(info: dict) -> dict:
    return next(s for s in info['streams'] if s['codec_type'] == 'video')


def make_plan(source: Path, request: dict, info: dict | None = None) -> dict:
    info = probe(source, count=True) if info is None else info
    v = video(info)
    start, duration = float(request['start']), float(request['duration'])
    strength = float(request['strength'])
    if not all(math.isfinite(x) for x in (start, duration, strength)) or min(start, duration) < 0:
        raise ValueError('Trim times and strength must be finite and nonnegative')
    if not 0 <= strength <= 1 or request['scale'] not in (1, 2):
        raise ValueError('Strength must be 0..1; scale must be 1 or 2')
    batch = int(request['batch'])
    if batch < 5 or batch > 33 or batch % 4 != 1:
        raise ValueError('Temporal batch must be 5, 9, 13, 17, 21, 25, 29 or 33')
    if v.get('color_transfer') in ('smpte2084', 'arib-std-b67') or v.get('color_primaries') == 'bt2020':
        raise ValueError('HDR/wide-gamut input is not supported. Color-manage a separate SDR copy first.')
    if v.get('field_order', 'unknown') not in ('unknown', 'progressive'):
        raise ValueError('Interlaced source: make a deliberately deinterlaced copy first.')
    # The pinned model input path is 8-bit RGB. Refuse silent high-bit-depth loss.
    pixel = v.get('pix_fmt', '')
    supported = {'yuv420p','yuv422p','yuv444p','yuvj420p','yuvj422p','yuvj444p',
                 'rgb24','bgr24','bgr0','rgb0','0rgb','0bgr','nv12','nv21','gray','yuyv422','uyvy422'}
    if pixel not in supported:
        raise ValueError('Unsupported pixel format: this workflow accepts common 8-bit SDR formats, not high-bit-depth, alpha, RAW/log/HDR masters.')
    if v.get('sample_aspect_ratio', '1:1') not in ('1:1', '0:1', 'N/A'):
        raise ValueError('Non-square pixels require explicit normalization before use.')
    rotation = float(v.get('tags', {}).get('rotate', 0))
    for side in v.get('side_data_list', []):
        rotation = float(side.get('rotation', rotation))
    if rotation % 90:
        raise ValueError('Only right-angle display rotation is supported.')
    w, h = int(v['width']), int(v['height'])
    if round(rotation) % 180:
        w, h = h, w
    if w % 2 or h % 2:
        raise ValueError('Odd source dimensions: normalize deliberately. No silent crop will be applied.')
    if max(w, h) * request['scale'] > 4096:
        raise ValueError('This bounded preset supports output up to a 4096-pixel long edge.')
    rate = v.get('avg_frame_rate', '0/0')
    if rate in ('0/0', '0', 'N/A'):
        rate = v.get('r_frame_rate', '0/0')
    fps = Fraction(rate)
    if not 1 <= fps <= 120:
        raise ValueError('Unsupported or unknown source frame rate.')
    total = int(v.get('nb_read_frames') or v.get('nb_frames') or 0)
    if total < 1:
        raise ValueError('Could not count the source frames.')
    first = round(start * fps)
    count = round(duration * fps) if duration else total - first
    if first < 0 or count < 1 or first + count > total:
        raise ValueError(f'Requested trim exceeds source ({total} frames at {fps} FPS). Use duration 0 for the remainder.')
    # Do not silently remove an intentional audio delay.
    audios = [x for x in info['streams'] if x['codec_type'] == 'audio']
    if audios:
        offset = float(audios[0].get('start_time', 0)) - float(v.get('start_time', 0))
        if abs(offset) > max(0.05, 1.0 / float(fps)):
            raise ValueError('Source audio/video has a nontrivial start offset. Normalize sync explicitly first.')
    matrix = v.get('color_space', 'unknown')
    matrix = {'smpte170m':'bt601', 'bt470bg':'bt601', 'bt709':'bt709'}.get(matrix, 'bt709')
    source_range = 'full' if v.get('color_range') == 'pc' or pixel.startswith(('rgb','bgr','gbr')) else 'tv'
    return dict(source=str(source), source_size=[w,h], output_size=[w*request['scale'],h*request['scale']],
                fps=str(fps), first=first, count=count, duration=float(count/fps),
                start=float(first/fps), source_audio=bool(audios), input_matrix=matrix,
                input_range=source_range, metadata_assumption='Untagged HD SDR is interpreted as BT.709.',
                color_note='RGB working frames; BT.709-limited H.264 delivery; no white balance or relighting.')


def prepare(source: Path, job: Path, plan: dict) -> Path:
    dest = job/'prepared.mkv'
    filters = [f'trim=start_frame={plan["first"]}:end_frame={plan["first"]+plan["count"]}',
               'setpts=PTS-STARTPTS',
               f'scale=in_color_matrix={plan["input_matrix"]}:in_range={plan["input_range"]}:out_range=full',
               'format=bgr0', 'setsar=1']
    command = ['ffmpeg','-nostdin','-hide_banner','-v','warning','-y','-threads','2','-filter_threads','1',
               '-i',str(source),'-map','0:v:0','-map','0:a:0?','-vf',','.join(filters),
               *video_sync_args(),'-c:v','ffv1','-level','3','-pix_fmt','bgr0','-threads','2']
    if plan['source_audio']:
        command += ['-af',f'atrim=start={plan["start"]}:duration={plan["duration"]},asetpts=PTS-STARTPTS',
                    '-c:a','pcm_s24le']
    command += [str(dest)]
    run_logged(command, job/'prepare.log')
    v = video(probe(dest, True))
    if [v['width'],v['height']] != plan['source_size'] or int(v['nb_read_frames']) != plan['count']:
        raise RuntimeError('Prepared geometry or frame count differs from the source plan.')
    check_cfr(dest, Fraction(plan['fps']), plan['count'], job)
    return dest


def check_cfr(path: Path, fps: Fraction, count: int, job: Path) -> None:
    # Check actual timestamps rather than treating r_frame_rate as proof of CFR.
    with (job/'timestamp_check.log').open('wb') as err:
        p = subprocess.Popen(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
                              '-show_entries','frame=best_effort_timestamp_time',
                              '-of','csv=p=0',str(path)], stdout=subprocess.PIPE, stderr=err, text=True)
        first = None
        seen = 0
        try:
            for row in p.stdout:
                field = row.strip().split(',')[0]
                if not field:
                    continue
                stamp = float(field)
                if first is None:
                    first = stamp
                expected = first + seen / float(fps)
                if abs(stamp - expected) > max(0.002, 0.05 / float(fps)):
                    raise ValueError('Variable or discontinuous frame timing detected. No silent CFR conversion was performed.')
                seen += 1
            if p.wait() or seen != count:
                raise RuntimeError('Source timestamp/frame count verification failed.')
        finally:
            from fv_common import terminate
            terminate(p)


def detect_cuts(path: Path, job: Path, count: int) -> list[int]:
    # Only a small downscaled prepass is analyzed. Rendered frames are NOT resized here.
    command = ['ffmpeg','-nostdin','-hide_banner','-v','info','-threads','2','-filter_threads','1',
               '-i',str(path),'-an','-vf',
               'scale=320:-2,scdet=t=10,metadata=mode=print:key=lavfi.scd.time',
               '-f','null','-']
    log = job/'cuts.log'
    run_logged(command, log)
    result = []
    last = None
    for line in log.read_text(errors='replace').splitlines():
        match = re.search(r'frame:\s*(\d+)\s+pts:',line)
        if match:
            last = int(match[1])
        if 'lavfi.scd.time=' in line and last is not None and 0 < last < count:
            result.append(last)
    return sorted(set(result))


def finish(job: Path, output: Path, plan: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a completed output.')
    # Native 16-bit PNGs avoid another 8-bit quantization before the delivery encode.
    partial = output.with_suffix('.partial.mp4')
    command = ['ffmpeg','-nostdin','-hide_banner','-v','warning','-y','-threads','2','-filter_threads','1',
               '-framerate',plan['fps'],'-start_number','0','-i',str(job/'frames/frame_%08d.png'),
               '-i',str(job/'prepared.mkv'),'-map','0:v:0','-map','1:a:0?',
               '-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,setsar=1',
               '-frames:v',str(plan['count']),'-c:v','libx264','-preset','slow','-crf','14',
               '-pix_fmt','yuv420p','-threads','2','-color_primaries','bt709',
               '-color_trc','bt709','-colorspace','bt709','-color_range','tv',
               '-c:a','aac','-b:a','256k','-movflags','+faststart',str(partial)]
    run_logged(command, job/'encode.log')
    info = probe(partial, True)
    v = video(info)
    if [v['width'],v['height']] != plan['output_size'] or int(v['nb_read_frames']) != plan['count']:
        raise RuntimeError('Final geometry/frame count check failed; intermediate frames retained.')
    if Fraction(v['avg_frame_rate']) != Fraction(plan['fps']):
        raise RuntimeError('Output FPS changed; intermediate frames retained.')
    if abs(float(v['duration']) - plan['duration']) > 1.0 / float(Fraction(plan['fps'])):
        raise RuntimeError('Output duration check failed.')
    if plan['source_audio'] and not any(x['codec_type']=='audio' for x in info['streams']):
        raise RuntimeError('Audio was lost during encoding.')
    partial.replace(output)
