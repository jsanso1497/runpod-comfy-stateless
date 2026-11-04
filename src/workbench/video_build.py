"""Fail-closed patch of the pinned legacy cancellation handler, plus source contracts."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path

UPSTREAM = '159ca2dcaf5ee77a80bd33928626d772ce4b8c75'
OLD = '''        except Exception as e:
            cleanup(dit_cache=dit_cache, vae_cache=vae_cache)
            raise e'''
NEW = '''        except BaseException:
            # ComfyUI cancellation derives from BaseException, not Exception.
            # Never preserve cache entries on an aborted run. Do not mask the
            # original exception if best-effort cleanup itself fails.
            try:
                cleanup(dit_cache=False, vae_cache=False)
            except Exception:
                pass
            raise'''


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def patch(comfy: Path = Path('/opt/ComfyUI')) -> None:
    root = comfy / 'custom_nodes/ComfyUI-SeedVR2_VideoUpscaler'
    target = root / 'src/interfaces/video_upscaler.py'
    data = target.read_bytes()
    if NEW in data.decode():
        return
    if git_blob(data) != UPSTREAM or data.decode().count(OLD) != 1:
        raise RuntimeError('SeedVR2 source changed. Refusing to apply an unreviewed cancellation patch.')
    modified = data.decode().replace(OLD, NEW)
    ast.parse(modified)
    target.write_text(modified)
    cli = (root / 'inference_cli.py').read_text()
    for marker in ('--chunk_size', '--output_format', '--model_dir', '--swap_io_components'):
        if marker not in cli:
            raise RuntimeError('SeedVR2 streaming CLI contract missing: ' + marker)
    core = (comfy / 'comfy_api/latest/_input_impl/video_types.py').read_text()
    if 'def get_stream_source(' not in core:
        raise RuntimeError('Core VIDEO does not expose a file-backed source')

if __name__ == '__main__':
    patch()
