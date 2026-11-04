"""Select equivalent timestamp-preserving flags from the installed FFmpeg's help.

FFmpeg 4.4 uses -vsync 0. Newer builds support -fps_mode passthrough.
Only one video output exists in this pipeline, so the legacy global scope is
unambiguous. Never omit synchronization flags or silently convert to CFR.
"""
from __future__ import annotations
from functools import lru_cache
import re
import shutil
import subprocess


LEGACY_MODES = {'passthrough': '0', 'vfr': '2'}


def sync_args_from_help(help_text: str, mode: str = 'passthrough') -> list[str]:
    """Use actual advertised options, not a guessed version threshold."""
    if mode not in LEGACY_MODES:
        raise ValueError('Only passthrough and vfr modes are supported here.')
    if re.search(r'(?m)^\s*-fps_mode(?:\s|\[|:|$)', help_text):
        return ['-fps_mode', mode]
    if re.search(r'(?m)^\s*-vsync(?:\s|$)', help_text):
        return ['-vsync', LEGACY_MODES[mode]]
    raise RuntimeError('FFmpeg advertises neither -fps_mode nor -vsync; refusing to change frame timing.')


@lru_cache(maxsize=4)
def _full_help(executable: str) -> str:
    try:
        return subprocess.check_output(
            [executable, '-hide_banner', '-h', 'full'],
            stderr=subprocess.STDOUT, text=True, errors='replace', timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'Cannot inspect FFmpeg capabilities at {executable}: {exc}') from exc


def video_sync_args(mode: str = 'passthrough', executable: str = 'ffmpeg') -> list[str]:
    """Return output flags for the binary currently resolved through PATH."""
    if mode not in LEGACY_MODES:
        raise ValueError('Only passthrough and vfr modes are supported here.')
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f'FFmpeg executable not found: {executable}')
    return sync_args_from_help(_full_help(resolved), mode)
