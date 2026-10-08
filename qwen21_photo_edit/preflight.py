#!/usr/bin/env python3
"""Check local core interfaces without initializing CUDA or downloading models."""
from __future__ import annotations
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request


def require_method(path, class_name, method_name, arguments):
    if not path.is_file(): raise RuntimeError(f'Missing core file: {path}')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
    if cls is None: raise RuntimeError(f'Core lacks {class_name}. Update/rebuild ComfyUI first.')
    method = next((n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name), None)
    if method is None: raise RuntimeError(f'Core lacks {class_name}.{method_name}.')
    present = {a.arg for a in method.args.args + method.args.kwonlyargs}
    missing = set(arguments) - present
    if missing: raise RuntimeError(f'Core API changed: {class_name}.{method_name} missing {sorted(missing)}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--comfy-home',type=Path,default=Path('/workspace/ComfyUI'))
    p.add_argument('--sam',action='store_true')
    p.add_argument('--server',default='',help='Optional running ComfyUI URL after restart, e.g. http://127.0.0.1:8188')
    args=p.parse_args(); home=args.comfy_home.resolve()
    require_method(home/'comfy_extras/nodes_qwen.py','TextEncodeQwenImage21','execute', ['clip','prompt','negative_prompt','vae','resolution','images'])
    require_method(home/'comfy_extras/nodes_qwen.py','QwenImage21Cache','execute',['model','device','dtype'])
    if args.sam:
        require_method(home/'comfy_extras/nodes_sam3.py','SAM3_Detect','execute',
                       ['model','image','conditioning','threshold','refine_iterations','individual_masks'])
    for package in ['torch','numpy','PIL','scipy']:
        if importlib.util.find_spec(package) is None: raise RuntimeError('Missing Python dependency: '+package)
    if args.server:
        with urllib.request.urlopen(args.server.rstrip('/')+'/object_info',timeout=30) as response:
            info=json.load(response)
        required=['Q21PhotoMask','Q21PhotoCrop','Q21PhotoModels','Q21PhotoLoRAs','Q21PhotoReference','Q21PhotoEncode','Q21PhotoStitch','Q21PhotoReviewSave']
        missing=[name for name in required if name not in info]
        if missing: raise RuntimeError('Node(s) not registered. Restart ComfyUI and inspect startup errors: '+', '.join(missing))
        print('All eight add-on node classes registered in the live server.')
    print('Core interface and dependency checks passed. This is not a GPU inference test.')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('PREFLIGHT FAILED:',exc,file=sys.stderr)
        raise SystemExit(2)
