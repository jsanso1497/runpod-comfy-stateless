"""Build-time checks that do not need a GPU or download model weights."""
from pathlib import Path
import json
import re
import subprocess
import sys
import torch

if torch.__version__.split('+')[0] != '2.9.0':
    raise SystemExit('Expected the existing PyTorch 2.9.0 base; do not replace it silently.')
help_text = Path('/opt/video-restore/seedvr2-help.txt').read_text()
for arg in ['--resolution', '--max_resolution', '--batch_size', '--temporal_overlap',
            '--chunk_size', '--model_dir', '--output_format', '--attention_mode',
            '--vae_encode_tiled', '--vae_decode_tiled', '--cache_dit', '--cache_vae']:
    if arg not in help_text:
        raise SystemExit(f'Pinned SeedVR2 CLI does not support {arg}')
manifest = json.loads(Path('/opt/video-restore/models.json').read_text())
registry = Path('/opt/seedvr2/src/utils/model_registry.py').read_text()
for entry in manifest['models'].values():
    if entry['filename'] not in registry or entry['sha256'] not in registry:
        raise SystemExit(f'Model registry mismatch: {entry["filename"]}')
filters = subprocess.check_output(['ffmpeg', '-hide_banner', '-filters'], text=True)
for f in ['hqdn3d', 'unsharp', 'scale', 'eq', 'hstack']:
    if not re.search(r'\b' + f + r'\b', filters):
        raise SystemExit(f'FFmpeg filter missing: {f}')
print('Build-time imports, CLI options, model hashes and FFmpeg filters: PASS')
print('GPU inference is NOT validated by this build-time check.')
