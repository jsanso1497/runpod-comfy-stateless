#!/usr/bin/env python3
"""Download an explicit allowlist only. Resume, SHA-256 verify, atomically publish."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_destination(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError(f'Unsafe destination: {relative}')
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Destination escapes model directory: {relative}')
    return result


def load_assets(manifest: Path, profile: str, include_bfs: bool = False) -> list[dict]:
    if profile not in ('identity', 'upscale'):
        raise ValueError('Unknown native profile')
    data = json.loads(manifest.read_text())
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported manifest version')
    allowed_profiles = {'identity'} | ({'upscale'} if profile == 'upscale' else set())
    if include_bfs:
        allowed_profiles.add('bfs_optional')
    assets = [a for a in data['assets'] if a['profile'] in allowed_profiles]
    seen = set()
    for asset in assets:
        if not re.fullmatch(r'[a-f0-9]{64}', asset['sha256']):
            raise ValueError(f"Invalid checksum for {asset['id']}")
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', asset['repo']):
            raise ValueError('Invalid Hugging Face repository')
        safe_destination(Path('/models'), asset['destination'])
        if asset['destination'] in seen:
            raise ValueError('Duplicate model destination')
        seen.add(asset['destination'])
    return assets


def download_one(asset: dict, models_dir: Path, token: str | None = None,
                 retries: int = 4, session_factory=requests.Session) -> Path:
    dest = safe_destination(models_dir, asset['destination'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        print(f"Checking existing {asset['destination']}", flush=True)
        if sha256_file(dest) == asset['sha256']:
            print(f"Verified {asset['destination']}", flush=True)
            return dest
        raise RuntimeError(f'Existing model failed SHA-256: {dest}. Move that file aside and retry; it was not overwritten.')
    partial = dest.with_name(dest.name + '.part')
    url = 'https://huggingface.co/{}/resolve/{}/{}?download=true'.format(
        asset['repo'], quote(asset['revision'], safe=''), quote(asset['file'], safe='/'))
    with session_factory() as session:
        if token:
            session.headers.update({'Authorization': f'Bearer {token}'})
        session.headers.update({'User-Agent': 'Qwen21-Native-Suite/1.1', 'Accept-Encoding': 'identity'})
        for attempt in range(retries):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset and sha256_file(partial) == asset['sha256']:
                os.replace(partial, dest)
                return dest
            try:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                with session.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                    if response.status_code in (401, 403):
                        raise PermissionError(f"Access denied to {asset['repo']}/{asset['file']}. Set HF_TOKEN with access and accept any upstream access conditions.")
                    if response.status_code == 416:
                        partial.unlink(missing_ok=True)
                        raise requests.RequestException('Server rejected incomplete range; restarting this file')
                    response.raise_for_status()
                    if response.status_code == 206:
                        content_range = response.headers.get('Content-Range', '')
                        match = re.match(r'bytes (\d+)-(\d+)/(\d+|\*)', content_range)
                        if not match or int(match.group(1)) != offset:
                            raise RuntimeError('Invalid Content-Range; refusing to append inconsistent bytes')
                        mode = 'ab' if offset else 'wb'
                    else:
                        # Server ignored Range. Restart, never append a second complete file.
                        offset = 0
                        mode = 'wb'
                    print(f"Downloading {asset['destination']} (resume offset {offset:,} bytes)", flush=True)
                    with partial.open(mode) as stream:
                        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                actual = sha256_file(partial)
                if actual != asset['sha256']:
                    # A transport can terminate cleanly before all expected bytes arrive.
                    # Do not promote a mismatch or silently select another precision/model.
                    raise RuntimeError(f"SHA-256 mismatch for {asset['destination']}: expected {asset['sha256']}, received {actual}. Partial retained for inspection; remove it before a clean retry.")
                os.replace(partial, dest)
                print(f"Verified {asset['destination']}", flush=True)
                return dest
            except requests.RequestException as exc:
                if attempt + 1 == retries:
                    # Do not echo URLs or redirect query strings, which may carry signed credentials.
                    raise RuntimeError(f"Download failed for {asset['destination']} after {retries} attempts ({type(exc).__name__}).") from None
                print(f"Retrying {asset['destination']} ({type(exc).__name__})", flush=True)
                time.sleep(min(2 ** attempt, 15))
    raise RuntimeError('Unreachable download state')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'config/models.json')
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=['identity', 'upscale'], default='identity')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include-bfs', action='store_true', help='Opt in to two BFS adapters, only for comparison workflows')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    assets = load_assets(args.manifest, args.profile, args.include_bfs)
    approximate_total = sum(a['approx_bytes'] for a in assets)
    print(f'Profile {args.profile}: {len(assets)} explicit model files, approximately {approximate_total/1e9:.2f} GB decimal.')
    for a in assets:
        print('  ' + a['destination'])
    if args.dry_run:
        return
    args.models_dir.mkdir(parents=True, exist_ok=True)
    outstanding = 0
    for asset in assets:
        dest = safe_destination(args.models_dir, asset['destination'])
        partial = dest.with_name(dest.name + '.part')
        if not dest.exists():
            outstanding += max(0, asset['approx_bytes'] - (partial.stat().st_size if partial.exists() else 0))
    free = shutil.disk_usage(args.models_dir).free
    if free < outstanding * 1.1 + 5_000_000_000:
        raise RuntimeError(f'Insufficient disk space: {free/1e9:.1f} GB free; need about {(outstanding*1.1+5e9)/1e9:.1f} GB including headroom.')
    token = os.environ.get('HF_TOKEN') or None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda a: download_one(a, args.models_dir, token), assets))
    print('All selected model files are checksum verified.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, PermissionError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
