"""Offline source/package checks. Does not claim model inference has been tested."""
from __future__ import annotations
import ast
import json
from pathlib import Path, PurePosixPath
import re
from workflow_tools import PRESETS, validate_graph

PACKAGE = Path(__file__).resolve().parent


def validate_manifest(data: dict) -> None:
    seen_ids, seen_paths = set(), set()
    for row in data['models']:
        if row['id'] in seen_ids or row['destination'] in seen_paths:
            raise ValueError('Duplicate model ID or destination.')
        seen_ids.add(row['id']); seen_paths.add(row['destination'])
        for key in ('file', 'destination'):
            value = row[key]
            path = PurePosixPath(value)
            if path.is_absolute() or '..' in path.parts or '\\' in value or not path.parts:
                raise ValueError('Unsafe model path.')
        if not re.fullmatch(r'[a-f0-9]{64}', row['sha256']):
            raise ValueError('Invalid SHA-256.')
        if not re.fullmatch(r'[a-f0-9]{40}', row['revision']):
            raise ValueError('Model revision is not an immutable commit.')
        if row['approx_gb'] <= 0:
            raise ValueError('Invalid storage estimate.')
    expected = {'h3_fl2va_int8', 'h3_text_encoder', 'h3_video_vae', 'h3_turbo_v4_600_ema', 'h3_ref2va_int8'}
    if seen_ids != expected:
        raise ValueError('Workspace must download the four developer assets plus the locked INT8 reference checkpoint.')


def main() -> None:
    for path in PACKAGE.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    validate_manifest(json.loads((PACKAGE / 'models.lock.json').read_text()))
    sources = json.loads((PACKAGE / 'sources.lock.json').read_text())
    for row in sources['sources'].values():
        if not re.fullmatch(r'[a-f0-9]{40}', row['commit']):
            raise ValueError('Source must have an immutable commit.')
    folder = PACKAGE / 'workflows'
    if folder.is_dir():
        for _, filename, profile in PRESETS:
            validate_graph(json.loads((folder / filename).read_text()), profile)
        print('Verified generated workflow topology and preset settings.')
    else:
        print('Source-package checks passed. Official workflows are fetched and checked during Docker build.')
    from studio_validate import validate_studio
    validate_studio()
    print('Studio graph, blank input defaults and model routing contracts passed.')
    print('No Docker build, GPU memory or generated-image quality claim is implied.')


if __name__ == '__main__':
    main()
