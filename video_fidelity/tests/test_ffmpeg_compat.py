"""Regression coverage for the image's FFmpeg 4.4 -fps_mode failure."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
from unittest import mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ffmpeg_compat as compat
import media
from media_preflight import run_preflight


@pytest.mark.parametrize('help_text', [
    '-fps_mode mode      set framerate mode\n-vsync method',
    '  -fps_mode[:stream_specifier] mode\n',
    '-fps_mode:v mode\n',
])
def test_prefers_modern_per_stream_option(help_text):
    assert compat.sync_args_from_help(help_text) == ['-fps_mode', 'passthrough']
    assert compat.sync_args_from_help(help_text, 'vfr') == ['-fps_mode', 'vfr']


@pytest.mark.parametrize('mode,number', [('passthrough', '0'), ('vfr', '2')])
def test_legacy_44_uses_equivalent_mode(mode, number):
    assert compat.sync_args_from_help('-vsync              video sync method\n', mode) == ['-vsync', number]


@pytest.mark.parametrize('text', ['', 'use -fps_mode someday', '-fps_mode_other parameter', '-vsync_other parameter'])
def test_does_not_guess_or_omit_mode_when_unavailable(text):
    with pytest.raises(RuntimeError, match='refusing'):
        compat.sync_args_from_help(text)


@pytest.mark.parametrize('mode', ['cfr', 'auto', 'drop', 'unknown'])
def test_not_a_silent_cfr_fallback(mode):
    with pytest.raises(ValueError):
        compat.sync_args_from_help('-vsync method', mode)


def test_missing_binary_is_clear():
    with mock.patch.object(compat.shutil, 'which', return_value=None):
        with pytest.raises(RuntimeError, match='not found'):
            compat.video_sync_args()


@pytest.mark.parametrize('failure', [FileNotFoundError('missing'), subprocess.TimeoutExpired('ffmpeg', 20),
                                    subprocess.CalledProcessError(1, 'ffmpeg')])
def test_probe_failure_is_not_treated_as_legacy_support(failure):
    compat._full_help.cache_clear()
    with mock.patch.object(compat.subprocess, 'check_output', side_effect=failure):
        with pytest.raises(RuntimeError, match='Cannot inspect'):
            compat._full_help('/probe/failure/ffmpeg')
    compat._full_help.cache_clear()


def test_help_is_cached_per_resolved_executable():
    compat._full_help.cache_clear()
    with mock.patch.object(compat.shutil, 'which', return_value='/test/ffmpeg'), \
         mock.patch.object(compat.subprocess, 'check_output', return_value='-vsync method') as call:
        assert compat.video_sync_args() == ['-vsync', '0']
        assert compat.video_sync_args('vfr') == ['-vsync', '2']
        assert call.call_count == 1
    compat._full_help.cache_clear()


def test_real_container_media_preflight():
    run_preflight()


def test_real_legacy_prepare_path_including_vfr_rejection():
    # This exercises real FFmpeg with legacy -vsync flags. It does not claim
    # the host binary is version 4.4; the actual Docker gate tests that binary.
    executable = compat.shutil.which('ffmpeg')
    if executable is None or not compat.sync_args_from_help(compat._full_help(executable)):
        pytest.skip('No FFmpeg available')
    if '-vsync' not in compat._full_help(executable):
        pytest.skip('This binary removed legacy vsync; unit tests still cover selection.')
    with mock.patch.object(media, 'video_sync_args', return_value=['-vsync', '0']):
        run_preflight()


def test_preflight_runs_before_heavy_dependencies_in_docker():
    docker = (ROOT / 'Dockerfile').read_text()
    assert docker.index('RUN python media_preflight.py') < docker.index('git init /opt/ComfyUI')
    assert 'python validate_package.py' in docker
