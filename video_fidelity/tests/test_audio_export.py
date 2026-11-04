"""Regression coverage for FFmpeg 4.4's shared-output frame-limit hazard.

Actual media runs use the locally installed FFmpeg. Testing legacy flags on a
modern executable is not represented as testing the FFmpeg 4.4 binary itself.
The same media preflight also runs in the real build image before publishing.
"""
from __future__ import annotations
import inspect
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import media
import media_preflight


CASES = [
    ('integer', '10', 0.0, 0.0, True, 1.0),
    ('fractional_trim', '30000/1001', 0.2, 0.5, True, 1.0),
    ('film_trim', '24000/1001', 0.125, 0.5, True, 1.0),
    ('high_rate_trim', '60000/1001', 0.2, 0.5, True, 1.0),
    ('single_frame', '30000/1001', 0.2, 1001/30000, True, 1.0),
    ('video_only', '10', 0.0, 0.5, False, 1.0),
    ('short_audio', '30', 0.0, 0.0, True, 0.4),
]


@pytest.mark.parametrize('legacy_flags', [False, True], ids=['native_flags', 'legacy_flags'])
@pytest.mark.parametrize('case', CASES, ids=[row[0] for row in CASES])
def test_real_audio_and_timing(tmp_path, monkeypatch, case, legacy_flags):
    if legacy_flags:
        monkeypatch.setattr(media, 'video_sync_args', lambda: ['-vsync', '0'])
        monkeypatch.setattr(media_preflight, 'video_sync_args', lambda: ['-vsync', '0'])
    label, rate, start, duration, audio, audio_seconds = case
    result = media_preflight.media_case(tmp_path / label, rate, start, duration, audio, audio_seconds)
    assert result['audio'] is audio
    if audio:
        assert result['output_samples'] > 0 and result['output_rms'] > 0.01


def header(w=96, h=64):
    return b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', w, h)


def fake_sequence(tmp_path, count=3):
    folder = tmp_path / 'frames'
    folder.mkdir()
    for i in range(count):
        (folder / f'frame_{i:08d}.png').write_bytes(header())
    return folder, {'count': count, 'output_size': [96, 64]}


def test_sequence_validation_exact_complete(tmp_path):
    folder, plan = fake_sequence(tmp_path)
    media.validate_frame_sequence(folder, plan)


@pytest.mark.parametrize('change', ['gap', 'extra', 'wrong_name', 'wrong_size', 'bad_header', 'symlink'])
def test_sequence_rejects_invalid_before_encoding(tmp_path, change):
    folder, plan = fake_sequence(tmp_path)
    path = folder / 'frame_00000001.png'
    if change == 'gap':
        path.unlink()
    elif change == 'extra':
        (folder / 'frame_00000003.png').write_bytes(header())
    elif change == 'wrong_name':
        path.rename(folder / 'wrong.png')
    elif change == 'wrong_size':
        path.write_bytes(header(192, 128))
    elif change == 'bad_header':
        path.write_bytes(b'not a PNG')
    elif change == 'symlink':
        path.unlink()
        path.symlink_to(folder / 'frame_00000000.png')
    with pytest.raises(RuntimeError):
        media.validate_frame_sequence(folder, plan)


@pytest.mark.parametrize('value', [0, '0', None, 'N/A'])
def test_empty_audio_is_not_accepted(value):
    with pytest.raises(RuntimeError, match='no decodable'):
        media.require_audio({'streams': [{'codec_type': 'audio', 'nb_read_frames': value}]}, 'Test')


def test_missing_audio_is_not_accepted():
    with pytest.raises(RuntimeError, match='expected one audio'):
        media.require_audio({'streams': []}, 'Test')


def test_real_silent_audio_is_not_rejected():
    # Real silence is valid audio. Production checks do not require nonzero RMS.
    media.require_audio({'streams': [{'codec_type': 'audio', 'nb_read_frames': '12'}]}, 'Test')


@pytest.mark.parametrize('with_audio', [False, True])
def test_mixed_output_has_no_global_early_stop(tmp_path, monkeypatch, with_audio):
    _, plan = fake_sequence(tmp_path)
    plan.update(fps='30000/1001', source_audio=with_audio, duration=0.1001)
    monkeypatch.setattr(media, 'probe', lambda *a, **k: {
        'streams': [{'codec_type': 'audio', 'nb_read_frames': '5'}]})
    captured = []

    class CaptureDone(Exception):
        pass

    def capture(command, log):
        captured.extend(command)
        raise CaptureDone()

    monkeypatch.setattr(media, 'run_logged', capture)
    with pytest.raises(CaptureDone):
        media.finish(tmp_path, tmp_path/'out/enhanced.mp4', plan)
    for forbidden in ('-frames:v', '-vframes', '-frames:a', '-t', '-to', '-shortest', '-r'):
        assert forbidden not in captured
    assert captured[captured.index('-crf')+1] == '14'
    assert captured[captured.index('-preset')+1] == 'slow'
    assert captured[captured.index('-pix_fmt')+1] == 'yuv420p'
    assert 'empty_output_stream' in captured
    assert '-fps_mode' in captured or '-vsync' in captured
    if with_audio:
        assert '1:a:0' in captured and '1:a:0?' not in captured
        assert captured[captured.index('-c:a')+1] == 'aac'
        assert captured[captured.index('-b:a')+1] == '256k'
        assert '-an' not in captured
    else:
        assert '-an' in captured and str(tmp_path/'prepared.mkv') not in captured


def test_output_missing_audio_is_never_published(tmp_path, monkeypatch):
    real_probe = media.probe

    def remove_audio_from_report(path, *args, **kwargs):
        result = real_probe(path, *args, **kwargs)
        if str(path).endswith('enhanced.partial.mp4'):
            result['streams'] = [row for row in result['streams'] if row['codec_type'] != 'audio']
        return result

    monkeypatch.setattr(media, 'probe', remove_audio_from_report)
    job = tmp_path/'missing_audio'
    with pytest.raises(RuntimeError, match='expected one audio'):
        media_preflight.media_case(job, '30000/1001', 0.2, 0.5)
    assert not (job/'enhanced.mp4').exists()
    assert (job/'enhanced.partial.mp4').is_file()
    assert (job/'frames/frame_00000000.png').is_file()


def test_missing_prepared_audio_fails_before_render(tmp_path):
    source = tmp_path/'source.mkv'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-f','lavfi','-i',
                    'testsrc2=s=96x64:r=10:d=1','-c:v','ffv1',str(source)], check=True)
    plan = media.make_plan(source, dict(start=0, duration=0, scale=1, strength=0, batch=9))
    plan['source_audio'] = True
    job = tmp_path/'job'
    job.mkdir()
    with pytest.raises(RuntimeError, match='expected one audio'):
        media.prepare(source, job, plan)


def test_completed_video_is_not_overwritten(tmp_path):
    output = tmp_path/'enhanced.mp4'
    output.write_bytes(b'original')
    with pytest.raises(FileExistsError):
        media.finish(tmp_path, output, {})
    assert output.read_bytes() == b'original'


def test_preflight_diagnostics_are_printed_before_cleanup(tmp_path, capsys):
    (tmp_path/'encode.log').write_text('deliberate diagnostic marker')
    media_preflight.print_diagnostics(tmp_path)
    assert 'deliberate diagnostic marker' in capsys.readouterr().out


def test_original_failing_regression_remains_in_build_preflight():
    text = inspect.getsource(media_preflight.run_preflight)
    assert "'fractional_trim', '30000/1001', 0.2, 0.5" in text
    assert 'print_diagnostics(job)' in text
