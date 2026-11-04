"""Exercise the installed FFmpeg with actual media, before installing model code.

The tests intentionally include the exact short 30000/1001 audio regression.
No torch, NumPy, weights, network, or GPU are needed. On failure, print the
media probes and FFmpeg diagnostics before TemporaryDirectory removes them.
"""
from __future__ import annotations
from array import array
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from ffmpeg_compat import video_sync_args
from media import make_plan, prepare, detect_cuts, finish, probe, video


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f'Media preflight failed: {args}\n{result.stderr[-6000:]}')


def audio_signature(path: Path) -> tuple[int, float]:
    """Decode small SYNTHETIC test clips, checking real samples, not a track label.

    Normalize test audio to mono 48 kHz for sample-duration/RMS comparisons.
    This helper is only used on short test fixtures, never on user clips.
    """
    result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
                             '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '48000',
                             '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'],
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Audio decode failed: ' + result.stderr.decode('utf-8', 'replace')[-4000:])
    samples = array('h')
    samples.frombytes(result.stdout)
    if sys.byteorder != 'little':
        samples.byteswap()
    if not samples:
        raise RuntimeError('Audio track contains no decoded samples.')
    rms = math.sqrt(sum(float(value)**2 for value in samples) / len(samples)) / 32768.0
    return len(samples), rms


def print_diagnostics(job: Path) -> None:
    print('FAILED media case: ' + job.name, flush=True)
    for name in ('prepare.log', 'timestamp_check.log', 'encode.log'):
        path = job / name
        if path.exists():
            print('\n--- ' + name + ' ---\n' + path.read_text(errors='replace')[-6000:], flush=True)
    for name in ('source.mkv', 'prepared.mkv', 'enhanced.partial.mp4', 'enhanced.mp4'):
        path = job / name
        if not path.exists():
            continue
        try:
            fields = {'index', 'codec_type', 'codec_name', 'sample_rate', 'channels',
                      'width', 'height', 'avg_frame_rate', 'time_base', 'start_time',
                      'duration', 'nb_read_frames', 'tags'}
            rows = [{k: v for k, v in row.items() if k in fields}
                    for row in probe(path, True)['streams']]
            print('\n--- probe ' + name + ' ---\n' + json.dumps(rows, indent=2), flush=True)
        except Exception as exc:
            print('Probe failed for ' + name + ': ' + str(exc), flush=True)


def media_case(job: Path, rate: str, start: float, duration: float,
               audio: bool = True, audio_seconds: float = 1.0) -> dict:
    job.mkdir()
    source = job / 'source.mkv'
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '2', '-filter_threads', '1',
               '-f', 'lavfi', '-i', f'testsrc2=s=96x64:r={rate}:d=1']
    if audio:
        command += ['-f', 'lavfi', '-i', f'sine=f=440:r=48000:d={audio_seconds}']
    command += ['-c:v', 'ffv1', '-threads', '2']
    if audio:
        command += ['-c:a', 'pcm_s16le']
    command += [str(source)]
    _run(command)
    request = dict(start=start, duration=duration, strength=0.0, scale=1, batch=9)
    plan = make_plan(source, request)
    prepared = prepare(source, job, plan)
    detect_cuts(prepared, job, plan['count'])
    frames = job / 'frames'
    frames.mkdir()
    _run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '2', '-filter_threads', '1',
          '-i', str(prepared), '-an', *video_sync_args(), '-frames:v', str(plan['count']),
          '-pix_fmt', 'rgb48be', '-threads', '2', '-start_number', '0',
          str(frames / 'frame_%08d.png')])
    # -frames:v above is safe because that command has NO audio output.
    output = job / 'enhanced.mp4'
    finish(job, output, plan)
    info = probe(output, True)
    stream = video(info)
    if (stream['width'], stream['height'], int(stream['nb_read_frames'])) != (96, 64, plan['count']):
        raise RuntimeError('Media preflight changed output size or frame count.')
    if Fraction(stream['avg_frame_rate']) != Fraction(plan['fps']):
        raise RuntimeError('Media preflight changed output frame rate.')
    audios = [row for row in info['streams'] if row['codec_type'] == 'audio']
    if bool(audios) != audio:
        raise RuntimeError('Media preflight lost audio or added unwanted audio.')
    summary = dict(case=job.name, frames=plan['count'], fps=plan['fps'], requested_fixture_rate=rate, audio=audio)
    if audio:
        before, before_rms = audio_signature(prepared)
        after, after_rms = audio_signature(output)
        # AAC packets can include end padding. Permit two AAC frames, but
        # catch the absent track, empty stream, silent replacement or major cut.
        if abs(after - before) > 2048:
            raise RuntimeError(f'Audio duration changed: {before} input samples, {after} output samples.')
        if before_rms < 0.01 or after_rms < before_rms * 0.70:
            raise RuntimeError('The test tone was lost or replaced by silence.')
        if abs(float(audios[0].get('start_time', 0))) > 0.003:
            raise RuntimeError('Encoded audio gained an unexpected start offset.')
        summary.update(source_samples=before, output_samples=after, output_rms=round(after_rms, 6))
    print('PASS: ' + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def run_preflight() -> None:
    version = subprocess.check_output(['ffmpeg', '-version'], text=True, timeout=10).splitlines()[0]
    print(version, flush=True)
    print('Frame timing mode: ' + ' '.join(video_sync_args()), flush=True)
    with tempfile.TemporaryDirectory(prefix='faithful-media-preflight-') as folder:
        root = Path(folder)
        for label, rate, start, duration, audio, audio_seconds in (
            ('integer', '10', 0.0, 0.0, True, 1.0),
            ('fractional_trim', '30000/1001', 0.2, 0.5, True, 1.0),
            ('film_trim', '24000/1001', 0.125, 0.5, True, 1.0),
            ('high_rate_trim', '60000/1001', 0.2, 0.5, True, 1.0),
            ('single_frame', '30000/1001', 0.2, 1001/30000, True, 1.0),
            ('video_only', '10', 0.0, 0.5, False, 1.0),
            ('short_audio', '30', 0.0, 0.0, True, 0.4),
        ):
            job = root / label
            try:
                media_case(job, rate, start, duration, audio, audio_seconds)
            except Exception:
                print_diagnostics(job)
                raise

        # Keep the existing strict VFR regression; do not normalize away gaps.
        job = root / 'vfr'
        job.mkdir()
        try:
            source = job / 'source.mkv'
            _run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '2', '-filter_threads', '1',
                  '-f', 'lavfi', '-i', 'testsrc2=s=96x64:r=10:d=2',
                  '-vf', "select='not(eq(n,7))'", *video_sync_args('vfr'),
                  '-c:v', 'ffv1', '-threads', '2', str(source)])
            plan = make_plan(source, dict(start=0.0, duration=0.0, strength=0.0, scale=1, batch=9))
            try:
                prepare(source, job, plan)
            except ValueError as exc:
                if 'Variable or discontinuous' not in str(exc):
                    raise
            else:
                raise RuntimeError('VFR was silently normalized instead of rejected.')
        except Exception:
            print_diagnostics(job)
            raise
        print('PASS: discontinuous timestamps preserved and rejected, not normalized.', flush=True)
    print('PASS: container-native FFmpeg CPU media preflight. GPU restoration is not tested.', flush=True)


if __name__ == '__main__':
    run_preflight()
