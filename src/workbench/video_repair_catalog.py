"""Add isolated SeedVR2 video jobs to the Qwen workspace; leave Qwen weights intact."""
from __future__ import annotations

VERSION = '2026-11-03.video-repair-1'
MODEL = 'seedvr2_ema_7b_sharp_fp16.safetensors'
STANDARD = 'seedvr2_ema_7b_fp16.safetensors'
VAE = 'ema_vae_fp16.safetensors'
VARIANTS = [
    ('Sharp FP16 - 1080p full frame', 'SeedVR2_Sharp_1080p_Streaming.json'),
    ('Sharp FP16 - 4K full frame', 'SeedVR2_Sharp_4K_Streaming.json'),
    ('Sharp FP16 - 2x3 left crop, source-size finish', 'SeedVR2_Sharp_2x3_Streaming.json'),
    ('Standard FP16 - 1080p full frame', 'SeedVR2_Standard_1080p_Streaming.json'),
]

def apply(name: str, value: dict) -> dict:
    if name == 'models.json':
        for row in value['models']:
            if row['id'] == 'qwen':
                if 'seedvr_stream' not in row['local_nodes']:
                    row['local_nodes'].append('seedvr_stream')
                row['video_repair_version'] = VERSION
    elif name == 'tasks.json':
        # Hide the old whole-clip video task in Qwen, not its image finishers
        # and not its existing use in the other workspace families.
        for row in value['tasks']:
            if row['id'] == 'R03':
                row['workspaces'] = [x for x in row['workspaces'] if x != 'qwen']
        if not any(x['id'] == 'RVID' for x in value['tasks']):
            value['tasks'].append({
                'id': 'RVID', 'title': 'SeedVR2 video: isolated streaming FP16',
                'description': '1080p, UHD 4K, and left-anchored 2:3. Full clip in disk-backed chunks. Sharp and standard FP16; wavelet. Cancel kills the worker, not ComfyUI.',
                'family': 'restoration', 'category': 'tasks', 'workspaces': ['qwen'],
                'requires': ['seedvr2', 'seedvr2_sharp'], 'sources': [],
                'status': 'CPU contracts tested; GPU inference requires validation',
                'required_nodes': ['WBSeedVR2StreamVideo', 'LoadVideo'],
                'variants': [{'label': label, 'file': 'workflows/restoration/RVID_Streaming/' + filename,
                              'source': 'workbench-generated'} for label, filename in VARIANTS],
            })
    elif name == 'assets.json':
        if not any(x['id'] == 'restoration-seedvr2_7b_sharp_fp16' for x in value['assets']):
            value['assets'].append({
                'id': 'restoration-seedvr2_7b_sharp_fp16', 'family': 'restoration',
                'group': 'seedvr2_sharp', 'repo_id': 'numz/SeedVR2_comfyUI',
                'filename': MODEL, 'revision': 'main', 'destination': 'SEEDVR2/' + MODEL,
                'sha256': '20a93e01ff24beaeebc5de4e4e5be924359606c356c9c51509fba245bd2d77dd',
                'estimated_bytes': 16500000000,
                'integrity_source': 'Hugging Face published SHA256; verified on 2026-11-03',
                'verification': 'verified by runtime downloader; no weights bundled',
            })
    return value
