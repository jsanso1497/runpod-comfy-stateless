"""Validate the actual installed ComfyUI schema against the portable graphs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import urllib.request
from workflow_tools import PACKAGE, PRESETS, LOADER_FIELDS, validate_graph

FRONTEND_ONLY = {'MarkdownNote', 'Note', 'Reroute'}


def validate(object_info: dict, folder: Path | None = None) -> list[str]:
    check_studio = folder is None
    folder = folder or PACKAGE / 'workflows'
    issues = []
    for _, filename, profile in PRESETS:
        graph = json.loads((folder / filename).read_text())
        try:
            validate_graph(graph, profile)
        except (ValueError, KeyError, StopIteration) as error:
            issues.append(f'{filename}: {error}')
            continue
        for node in graph['nodes']:
            kind = node['type']
            if kind in FRONTEND_ONLY:
                continue
            spec = object_info.get(kind)
            if not spec:
                issues.append(f'{filename}: missing node {kind}')
                continue
            fields = {}
            for group in ('required', 'optional'):
                fields.update(spec.get('input', {}).get(group, {}))
            named = node.get('widgets_values_named', {})
            connected = {p['name'] for p in node.get('inputs',[]) if p.get('link') is not None}
            for required_name in spec.get('input',{}).get('required',{}):
                if kind == 'ResolutionSelector' and required_name == 'preview':
                    continue  # UI-only live resolution display, no execution value.
                if required_name not in named and required_name not in connected:
                    issues.append(f'{filename}: required {kind}.{required_name} has no value or connection')
            # Native nested reference sockets are dynamic. Check each concrete
            # scalar widget in the schema without rejecting dynamic input names.
            for name, value in named.items():
                if name in ('upload', 'control_after_generate'):
                    continue
                field = fields.get(name)
                if not field:
                    # New Comfy dynamic widgets may not be flattened in object_info.
                    # All critical known scalar fields must still be present.
                    if name in ('sampler_name', 'scheduler', 'steps', 'denoise',
                                'clip_name', 'unet_name', 'vae_name', 'lora_name',
                                'megapixels', 'multiple', 'resolution_steps', 'divisible_by', 'strength_model'):
                        issues.append(f'{filename}: {kind}.{name} missing from installed schema')
                    continue
                options = field[0]
                if name == 'image' and kind == 'LoadImage':
                    # The user must upload their own image. No source photograph
                    # is bundled or downloaded from an unrelated account.
                    continue
                if isinstance(options, list) and value not in options:
                    issues.append(f'{filename}: unsupported {kind}.{name}={value!r}')
                elif options in ('INT', 'FLOAT') and isinstance(value, (int, float)):
                    limits = field[1] if len(field) > 1 else {}
                    if value < limits.get('min', float('-inf')) or value > limits.get('max', float('inf')):
                        issues.append(f'{filename}: {kind}.{name} outside installed limits')
            if kind in LOADER_FIELDS:
                name = LOADER_FIELDS[kind][0]
                if name not in fields:
                    issues.append(f'{filename}: loader schema has no {name}')
    if check_studio:
        from studio_validate import validate_studio
        try:
            validate_studio(object_info)
        except (ValueError, KeyError, AssertionError) as error:
            issues.append('H3 Studio: ' + str(error))
    if check_studio:
        try:
            from media_validate import validate_all
            validate_all(object_info)
        except (ValueError, KeyError, AssertionError) as error:
            issues.append('Media v2: ' + str(error))
    return sorted(set(issues))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8189/object_info')
    args = parser.parse_args()
    with urllib.request.urlopen(args.url, timeout=30) as response:
        issues = validate(json.load(response))
    if issues:
        raise SystemExit('\n'.join(issues))
    print('Actual installed node schemas and model selections passed. No inference was queued.')


if __name__ == '__main__':
    main()
