"""Prepare only the new H3 Image workspace; launch native ComfyUI on loopback."""
import json
import os
from pathlib import Path
import shutil
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from assets import data_root, download
from helpers import terminal_environment


def prepare(root):
    import yaml
    for name in ('input', 'output', 'temp', 'user', 'logs', 'custom_nodes', 'models'):
        (root / name).mkdir(parents=True, exist_ok=True)
    paths = {'h3_image': {'base_path': str(root), 'is_default': True,
                         'diffusion_models': 'models/diffusion_models', 'text_encoders': 'models/text_encoders',
                         'vae': 'models/vae', 'loras': 'models/loras', 'checkpoints': 'models/checkpoints',
                         'custom_nodes': 'custom_nodes', 'download_model_base': 'models'}}
    (root / 'extra_model_paths.yaml').write_text(yaml.safe_dump(paths), encoding='utf-8')
    workflows = root / 'user/default/workflows/00_H3_Image_Manual_v2'
    workflows.mkdir(parents=True, exist_ok=True)
    for source in (PACKAGE / 'workflows').glob('*.json'):
        target = workflows / source.name
        if not target.exists():
            with target.open('xb') as dest, source.open('rb') as src:
                shutil.copyfileobj(src, dest)
    manager = root / 'user/__manager'
    manager.mkdir(parents=True, exist_ok=True)
    config = manager / 'config.ini'
    if not config.exists():
        config.write_text('[default]\nsecurity_level = normal\nnetwork_mode = public\nbypass_ssl = False\n', encoding='utf-8')
    # Added node folders persist. Their third-party dependencies are not silently reinstalled at startup.


def main():
    root = data_root()
    prepare(root)
    import torch
    # The persistent lock lives in the exact backend process and remains held across exec.
    guard = (root / 'run/backend.lock')
    guard.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock_fd = os.open(guard, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another H3 Image backend is already running. Use control.py restart, not a second launcher.')
    os.set_inheritable(lock_fd, True)
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU is available. Files/Jupyter stay running for diagnosis.')
    print('GPU:', torch.cuda.get_device_name(), 'CUDA:', torch.version.cuda, flush=True)
    download()
    command = [sys.executable, '/opt/ComfyUI/main.py', '--listen', '127.0.0.1', '--port', '8189',
               '--input-directory', str(root / 'input'), '--output-directory', str(root / 'output'),
               '--user-directory', str(root / 'user'), '--temp-directory', str(root / 'temp'),
               '--extra-model-paths-config', str(root / 'extra_model_paths.yaml'),
               '--enable-manager', '--preview-method', 'none', '--disable-auto-launch',
               '--cache-none', '--disable-pinned-memory']
    # No cached pipeline graphs or pinned host buffers by default after the user's OOM history.
    # This can cost repeat-run speed; it is not a guarantee that all resolutions fit.
    env = terminal_environment()
    os.chdir('/opt/ComfyUI')
    os.execve(sys.executable, command, env)


if __name__ == '__main__':
    main()
