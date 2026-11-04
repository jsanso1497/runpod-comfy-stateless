#!/usr/bin/env python3
"""Validate installed node availability and actual model selectors, not generated-image quality."""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent

def validate(info: dict, check_models: bool = True) -> list[str]:
    errors = []
    for path in sorted((ROOT/'api').glob('*.json')):
        for node in json.loads(path.read_text()).values():
            kind = node['class_type']
            schema = info.get(kind)
            if not schema:
                errors.append(f'{path.name}: missing node {kind}')
                continue
            supplied = node['inputs']
            groups = schema.get('input', {})
            specs = {**groups.get('required', {}), **groups.get('optional', {})}
            # Dynamic Combo and Autogrow inputs expand at runtime. Node-presence and
            # fixed selector checks are reliable here; this is not a fake GPU test.
            for key, value in supplied.items():
                if isinstance(value, list) or key not in specs:
                    continue
                spec = specs[key]
                if not isinstance(spec, (list, tuple)) or not spec:
                    continue
                options = spec[0]
                if not check_models and key in ('unet_name','clip_name','vae_name','lora_name','image'):
                    continue
                # Source uploads are intentionally unset until the user chooses a photo.
                if key == 'image':
                    continue
                if isinstance(options, list) and options and all(isinstance(x, (str,int,float)) for x in options):
                    if value not in options:
                        errors.append(f'{path.name}: {kind}.{key} does not offer {value!r}')
            if kind == 'TextEncodeMOSSImageEdit' and 'edit_format' not in specs:
                errors.append('TextEncodeMOSSImageEdit does not expose the required native edit_format selector.')
    return sorted(set(errors))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8189')
    parser.add_argument('--object-info', type=Path)
    parser.add_argument('--skip-models', action='store_true')
    args = parser.parse_args()
    if args.object_info:
        info = json.loads(args.object_info.read_text())
    else:
        req = urllib.request.Request(args.url.rstrip('/')+'/object_info')
        password = os.environ.get('COMFY_PASSWORD')
        if password:
            token = base64.b64encode(('workbench:'+password).encode()).decode()
            req.add_header('Authorization','Basic '+token)
        with urllib.request.urlopen(req, timeout=60) as response:
            info = json.load(response)
    errors = validate(info, not args.skip_models)
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Installed node/selector checks passed. GPU inference and mobile rendering are NOT certified by this check.')

if __name__ == '__main__':
    main()
