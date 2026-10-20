"""Pinned still-source contract and read-only runtime checks. Never queues work."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

COMFY_COMMIT = 'b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c'
FIZGIG_COMMIT = '10d5171f6e45b0757a03df977f53a660e060c33d'
SOURCE_BLOBS = {
    'comfy/ldm/minimax/vae.py': '9f17dad58b2dae6d41193c574a8bb87f2c6df0f1',
    'custom_nodes/ComfyUI-Fizgig-H3-Still/__init__.py': '543b3bb79d346bcf22a0dbcfac47d912337fb7f9',
}
REQUIRED = {'UNETLoader', 'CLIPLoader', 'VAELoader', 'BasicGuider', 'BasicScheduler',
            'KSamplerSelect', 'RandomNoise', 'SamplerCustomAdvanced', 'MiniMaxH3SigmaShift',
            'FizgigH3StillLatent', 'VAEDecode', 'WBH3StillSize', 'WBH3SaveStillMaster', 'SaveImage'}


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def verify_sources(comfy_root, commit=COMFY_COMMIT):
    if commit != COMFY_COMMIT:
        raise ValueError('H3 still core pin changed. Review and re-test before upgrading.')
    root = Path(comfy_root)
    results = {}
    for filename, expected in SOURCE_BLOBS.items():
        data = (root / filename).read_bytes()
        actual = git_blob(data)
        if actual != expected:
            raise ValueError('H3 still source differs from the inspected revision: ' + filename)
        results[filename] = actual
    return {'ok': True, 'comfy_commit': commit, 'fizgig_commit': FIZGIG_COMMIT,
            'source_blobs': results, 'scope': 'Source identity only; no GPU inference.'}


def inputs(node):
    result = {}
    for group in ('required', 'optional'):
        result.update(node.get('input', {}).get(group, {}))
    return result


def inspect_schema(schema, text_only=False):
    required = REQUIRED | ({'MiniMaxH3ImageToVideo'} if text_only else {'LoadImage', 'MiniMaxH3ReferenceToVideo'})
    missing = sorted(required - schema.keys())
    problems = []
    checks = [('CLIPLoader', 'type', 'minimax'), ('KSamplerSelect', 'sampler_name', 'er_sde'),
              ('BasicScheduler', 'scheduler', 'simple'),
              ('UNETLoader', 'unet_name', 'minimax_h3_fl2va_bf16.safetensors' if text_only else 'minimax_h3_ref2va_bf16.safetensors'),
              ('CLIPLoader', 'clip_name', 'qwen3vl_32b_minimax_h3_bf16.safetensors'),
              ('VAELoader', 'vae_name', 'minimax_h3_video_vae_fp16.safetensors')]
    if not text_only:
        checks.append(('MiniMaxH3ReferenceToVideo', 'ref_image_size', 'max'))
    for node, field, choice in checks:
        spec = inputs(schema.get(node, {})).get(field)
        if node not in schema:
            continue
        if not spec or not isinstance(spec[0], list) or choice not in spec[0]:
            problems.append(f'{node}.{field} does not offer the exact required value: {choice}')
    for node, fields in {'FizgigH3StillLatent': ['width', 'height', 'batch_size'],
                         'VAEDecode': ['samples', 'vae'],
                         'WBH3StillSize': ['target_megapixels', 'aspect_ratio', 'reference'],
                         'WBH3SaveStillMaster': ['images', 'filename_prefix']}.items():
        if node not in schema:
            continue
        actual = inputs(schema[node])
        for name in fields:
            if name not in actual:
                problems.append(f'{node} is missing {name}')
    return {'ok': not missing and not problems, 'missing_nodes': missing, 'problems': problems,
            'task': 'HSTXT' if text_only else 'HSTILL',
            'scope': 'Read-only node/schema/model-visibility checks. Not a workflow execution or image-quality test.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sources', type=Path)
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--text', action='store_true', help='Check optional FL2VA text-only still assets.')
    parser.add_argument('--url', default='http://127.0.0.1:8189/object_info')
    args = parser.parse_args()
    if args.sources:
        report = verify_sources(args.sources)
    else:
        with urllib.request.urlopen(args.url, timeout=60) as response:
            report = inspect_schema(json.load(response), text_only=args.text)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
