"""Real CPU media gate using the FFmpeg installed INSIDE the container.

Runs before model libraries/source clones are installed. No torch, weights,
network, or GPU required. Exercises the real preparation/finishing functions.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import subprocess
import tempfile

from ffmpeg_compat import video_sync_args
from media import make_plan, prepare, detect_cuts, finish, probe, video


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f'Media preflight failed: {args}\n{result.stderr[-6000:]}')


def run_preflight() -> None:
    version = subprocess.check_output(['ffmpeg', '-version'], text=True, timeout=10).splitlines()[0]
    print(version, flush=True)
    print('Frame timing mode: ' + ' '.join(video_sync_args()), flush=True)
    with tempfile.TemporaryDirectory(prefix='faithful-media-preflight-') as folder:
        root = Path(folder)
        for label, rate, start, duration in (
            ('integer', '10', 0.0, 0.0),
            ('fractional_trim', '30000/1001', 0.2, 0.5),
        ):
            job = root / label
            job.mkdir()
            source = job / 'source.mkv'
            _run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '2', '-filter_threads', '1',
                  '-f', 'lavfi', '-i', f'testsrc2=s=96x64:r={rate}:d=1',
                  '-f', 'lavfi', '-i', 'sine=f=440:r=48000:d=1',
                  '-c:v', 'ffv1', '-threads', '2', '-c:a', 'pcm_s16le', str(source)])
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
            output = job / 'enhanced.mp4'
            finish(job, output, plan)
            info = probe(output, True)
            stream = video(info)
            if (stream['width'], stream['height'], int(stream['nb_read_frames'])) != (96, 64, plan['count']):
                raise RuntimeError('Media preflight changed output size or frame count.')
            if Fraction(stream['avg_frame_rate']) != Fraction(rate):
                raise RuntimeError('Media preflight changed output frame rate.')
            if not any(s['codec_type'] == 'audio' for s in info['streams']):
                raise RuntimeError('Media preflight lost audio.')
            print(f'PASS: {label}, {plan["count"]} frames, {rate} FPS, audio present.', flush=True)

        # An intentional timestamp gap must remain a gap, then be rejected by
        # the workflow. A fallback that silently makes CFR would fail this test.
        job = root / 'vfr'
        job.mkdir()
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
        print('PASS: discontinuous timestamps preserved and rejected, not normalized.', flush=True)
    print('PASS: container-native FFmpeg CPU media preflight. GPU restoration is not tested.', flush=True)


if __name__ == '__main__':
    run_preflight()
