"""Prepare persistent assets and run ComfyUI on loopback. The gateway is separate."""
import json
import os
from pathlib import Path
import sys
from bootstrap import prepare, verified, safe_child
from recovery_launcher import terminal_environment

PACKAGE = Path(__file__).resolve().parents[1]


def prepare_turbo(root):
    """Reuse the user's previously downloaded REF2VA adapter; download only if absent."""
    import shutil
    row = json.loads((PACKAGE / 'extra-models.lock.json').read_text())['models'][0]
    path = safe_child(root / 'models', row['destination'])
    marker = root / '.verified' / (row['id'] + '.json')
    if verified(path, row, marker, os.environ.get('H3_STILL_VERIFY_ALL') == '1'):
        print('Using existing verified REF2VA Turbo adapter.', flush=True)
        return
    if path.exists() or path.is_symlink():
        raise RuntimeError('Existing REF2VA Turbo file failed verification. It was not overwritten: ' + str(path))
    if shutil.disk_usage(root).free < row['size_bytes'] + 512 * 1024 * 1024:
        raise RuntimeError('Allow 2.5 GB free for the REF2VA Turbo adapter.')
    from huggingface_hub import hf_hub_download
    path.parent.mkdir(parents=True, exist_ok=True)
    print('Preparing REF2VA Turbo adapter (1.96 GB only when missing).', flush=True)
    downloaded = Path(hf_hub_download(repo_id=row['repo'], filename=row['file'], revision=row['revision'],
                                     local_dir=str(path.parent), token=os.environ.get('HF_TOKEN') or None))
    if downloaded.resolve() != path.resolve() or not verified(downloaded, row, marker, True):
        raise RuntimeError('REF2VA Turbo adapter checksum verification failed.')


def main():
    root = Path(os.environ.get('H3_STILL_DATA_ROOT', '/workspace/h3-still')).resolve()
    if root == Path('/workspace') or not root.is_relative_to(Path('/workspace')):
        raise ValueError('Data root must be inside /workspace.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Files/Jupyter remain available on 8888; use a compatible GPU host.')
    prepare(root)
    prepare_turbo(root)
    env = terminal_environment()
    command = [sys.executable, '/opt/ComfyUI/main.py', '--listen', '127.0.0.1', '--port', '8189',
        '--input-directory', str(root / 'input'), '--output-directory', str(root / 'output'),
        '--user-directory', str(root / 'user'), '--temp-directory', str(root / 'temp'),
        '--extra-model-paths-config', str(root / 'extra_model_paths.yaml'),
        '--preview-method', 'none', '--disable-auto-launch']
    os.chdir('/opt/ComfyUI')
    os.execve(sys.executable, command, env)

if __name__ == '__main__':
    main()
