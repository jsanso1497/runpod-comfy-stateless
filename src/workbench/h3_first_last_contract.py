"""Build-time source contract and read-only live schema checks for HFL."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

RELEASE = '2026-10-19.1'
COMFY_COMMIT = 'b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c'
NATIVE_SOURCE_BLOB = '1ad3c6a4095fdcb756d6222f08ba589452b72091'
REQUIRED_NODES = {
    'UNETLoader', 'CLIPLoader', 'VAELoader', 'LoadImage',
    'MiniMaxH3ReferenceToVideo', 'MiniMaxH3AddGuide', 'MiniMaxH3SigmaShift',
    'BasicGuider', 'BasicScheduler', 'KSamplerSelect', 'RandomNoise', 'SamplerCustomAdvanced',
    'VAEDecode', 'VAEDecodeAudio', 'CreateVideo', 'SaveVideo',
    'WBH3LosslessMaster', 'WBH3LoadNativeMaster', 'WBH3ReleaseMemory',
    'SeedVR2LoadDiTModel', 'SeedVR2LoadVAEModel', 'SeedVR2VideoUpscaler',
}
MODEL_FILES = {
    'UNETLoader': ('unet_name', ['minimax_h3_ref2va_bf16.safetensors']),
    'CLIPLoader': ('clip_name', ['qwen3vl_32b_minimax_h3_bf16.safetensors']),
    'VAELoader': ('vae_name', ['minimax_h3_video_vae_fp16.safetensors', 'minimax_h3_audio_vae_fp32.safetensors']),
    'SeedVR2LoadDiTModel': ('model', ['seedvr2_ema_7b_fp16.safetensors']),
    'SeedVR2LoadVAEModel': ('model', ['ema_vae_fp16.safetensors']),
}


def verify_core(root: Path, commit: str = COMFY_COMMIT):
    if commit != COMFY_COMMIT:
        raise ValueError('HFL core pin changed. Review the native contract before upgrading.')
    root = Path(root)
    content = (root / 'comfy_extras/nodes_minimax_h3.py').read_bytes()
    digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    if digest != NATIVE_SOURCE_BLOB:
        raise ValueError('Native H3 source differs from the inspected first/last-frame implementation.')
    cli = (root / 'comfy/cli_args.py').read_text(encoding='utf-8')
    for flag in ('--use-pytorch-cross-attention', '--disable-comfy-compiler'):
        if flag not in cli:
            raise ValueError('Missing H3 launch flag: ' + flag)
    return {'comfy_commit': commit, 'native_source_blob': digest, 'source_contract': 'passed'}


def h3_launch_options(root: Path, disable_compiler: str = '1'):
    if disable_compiler not in ('0', '1'):
        raise ValueError('WB_DISABLE_COMFY_COMPILER must be 0 or 1.')
    cli = (Path(root) / 'comfy/cli_args.py').read_text(encoding='utf-8')
    flags = ['--use-pytorch-cross-attention']
    if disable_compiler == '1':
        flags.append('--disable-comfy-compiler')
    for flag in flags:
        if flag not in cli:
            raise ValueError('The H3 core does not support ' + flag)
    return flags


def _inputs(node):
    result = {}
    for group in ('required', 'optional'):
        result.update(node.get('input', {}).get(group, {}))
    return result


def inspect_schema(schema: dict) -> dict:
    missing_nodes = sorted(REQUIRED_NODES - schema.keys())
    problems = []
    checks = {
        'CLIPLoader': ('type', 'minimax'),
        'MiniMaxH3ReferenceToVideo': ('ref_image_size', 'max'),
        'KSamplerSelect': ('sampler_name', 'res_multistep'),
        'BasicScheduler': ('scheduler', 'simple'),
    }
    for node, (name, option) in checks.items():
        if node not in schema:
            continue
        spec = _inputs(schema[node]).get(name)
        if not spec or not isinstance(spec[0], list) or option not in spec[0]:
            problems.append(node + '.' + name + ' is missing the expected option: ' + option)
    for node, (name, filenames) in MODEL_FILES.items():
        if node not in schema:
            continue
        spec = _inputs(schema[node]).get(name, [[]])
        choices = spec[0] if isinstance(spec[0], list) else []
        for filename in filenames:
            if not any(str(c).replace('\\', '/').rsplit('/', 1)[-1] == filename for c in choices):
                problems.append('Missing prepared model in ' + node + ': ' + filename)
    if 'MiniMaxH3AddGuide' in schema:
        fields = _inputs(schema['MiniMaxH3AddGuide'])
        for name in ('positive', 'latent', 'frame_idx', 'image', 'vae'):
            if name not in fields:
                problems.append('MiniMaxH3AddGuide is missing ' + name)
    return {'missing_nodes': missing_nodes, 'problems': problems,
            'ok': not missing_nodes and not problems,
            'scope': 'Node names, selected options and prepared-model visibility only. No inference was run.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--url', default='http://127.0.0.1:8189/object_info')
    args = parser.parse_args()
    if args.core:
        report = verify_core(args.core)
    elif args.live:
        # Run inside the Pod. This is a read-only GET, never a queue submission.
        with urllib.request.urlopen(args.url, timeout=60) as response:
            report = inspect_schema(json.load(response))
    else:
        parser.error('Choose --core /opt/ComfyUI or --live.')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report.get('ok', True) else 1)


if __name__ == '__main__':
    main()
