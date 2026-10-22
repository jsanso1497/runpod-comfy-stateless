"""Read-only build/source and live-schema checks. Does not certify GPU inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

COMFY_COMMIT = 'b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c'
SOURCE_BLOBS = {
    'comfy_extras/nodes_minimax_h3.py': '1ad3c6a4095fdcb756d6222f08ba589452b72091',
    'comfy_extras/nodes_model_patch.py': '30374ce1e9ab0a3c769a8137ebe08252dc96e1a1',
    'comfy_extras/nodes_sdpose.py': 'badb467868fdff8c292c9950ddbbc3073f837c8c',
    'comfy_extras/nodes_rtdetr.py': '653f3af2f248f5895ab82da1c7dbe3519dc94fd6',
}
REQUIRED_NODES = {
    'LoadImage', 'LoadVideo', 'CheckpointLoaderSimple', 'UNETLoader', 'CLIPLoader', 'VAELoader',
    'RTDETR_detect', 'SDPoseKeypointExtractor', 'SDPoseDrawKeypoints', 'DrawBBoxes',
    'ModelPatchLoader', 'MiniMaxH3FunControlNetApply', 'MiniMaxH3ReferenceToVideo', 'MiniMaxH3SigmaShift',
    'BasicGuider', 'BasicScheduler', 'RandomNoise', 'KSamplerSelect', 'SamplerCustomAdvanced',
    'VAEDecode', 'CreateVideo', 'SaveVideo', 'PreviewAny', 'WBH3LosslessMaster',
    'WBH3MotionVideo', 'WBH3MotionTrackPerson', 'WBH3MotionReady', 'WBH3MotionCheckPatch', 'WBH3MotionCompare',
}
MODEL_FILES = {
    'UNETLoader': ('unet_name', ['minimax_h3_ref2va_bf16.safetensors', 'rt_detr_v4-x-hgnet_fp16.safetensors']),
    'CLIPLoader': ('clip_name', ['qwen3vl_32b_minimax_h3_bf16.safetensors']),
    'VAELoader': ('vae_name', ['minimax_h3_video_vae_fp16.safetensors']),
    'CheckpointLoaderSimple': ('ckpt_name', ['sdpose_wholebody_fp16.safetensors']),
    'ModelPatchLoader': ('name', ['minimax_h3_fun_controlnet_union_2.0_pruned_bf16.safetensors']),
}


def verify_core(root, commit=COMFY_COMMIT):
    if commit != COMFY_COMMIT:
        raise ValueError('Review the H3 motion contract before changing the ComfyUI pin.')
    root = Path(root)
    for name, expected in SOURCE_BLOBS.items():
        data = (root / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected:
            raise ValueError('Motion-control source changed: ' + name)
    patch_source = (root / 'comfy_extras/nodes_model_patch.py').read_text()
    if '50 // num_blocks' not in patch_source or 'inpaint_post_norm' not in patch_source:
        raise ValueError('The core does not implement ControlNet Union 2.0 loading.')
    return {'ok': True, 'core_commit': commit, 'source_blobs': SOURCE_BLOBS,
            'scope': 'Source integrity only. No model download or inference.'}


def inputs(node):
    result = {}
    for group in ('required', 'optional'):
        result.update(node.get('input', {}).get(group, {}))
    return result


def inspect_schema(schema):
    problems = []
    missing = sorted(REQUIRED_NODES - set(schema))
    for node, (field, files) in MODEL_FILES.items():
        spec = inputs(schema.get(node, {})).get(field, [[]])
        options = spec[0] if spec and isinstance(spec[0], list) else []
        for filename in files:
            if filename not in options:
                problems.append('Missing prepared file: ' + filename + ' in ' + node)
    for node, names in {
        'MiniMaxH3FunControlNetApply': ['model', 'model_patch', 'vae', 'control_video', 'strength', 'start_percent', 'end_percent'],
        'SDPoseKeypointExtractor': ['model', 'vae', 'image', 'bboxes', 'batch_size'],
        'RTDETR_detect': ['model', 'image', 'threshold', 'class_name', 'max_detections'],
    }.items():
        absent = set(names) - set(inputs(schema.get(node, {})))
        if absent:
            problems.append(node + ' missing inputs: ' + ', '.join(sorted(absent)))
    return {'ok': not missing and not problems, 'missing_nodes': missing, 'problems': problems,
            'scope': 'Node/schema/model visibility only, not inference or image quality.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--core', type=Path)
    group.add_argument('--live', action='store_true')
    a = p.parse_args()
    if a.core:
        result = verify_core(a.core)
    else:
        with urllib.request.urlopen('http://127.0.0.1:8189/object_info', timeout=60) as r:
            result = inspect_schema(json.load(r))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ok'] else 1)

if __name__ == '__main__':
    main()
