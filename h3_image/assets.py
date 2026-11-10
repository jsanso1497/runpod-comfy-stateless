#!/usr/bin/env python3
"""Resolve published model metadata before building; download only pinned assets on the Pod.

The source repository/revisions are not silently substituted if access fails.
No repository write or GPU operation is performed by 'resolve'.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

PACKAGE = Path(__file__).resolve().parent


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2)
            stream.write('\n')
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_destination(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError('Unsafe relative asset path.')
    candidate = Path(root) / path
    # Parent traversal is checked without following a legitimate final model symlink.
    if not str(candidate.parent.resolve()).startswith(str(Path(root).resolve()) + os.sep):
        raise ValueError('Asset parent escaped model directory.')
    return candidate


def hf_metadata(repo, token=None):
    url = 'https://huggingface.co/api/models/' + urllib.parse.quote(repo, safe='/') + '?blobs=true'
    headers = {'User-Agent': 'H3-Image-Workspace-Preflight/1.0'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError('Hugging Face returned HTTP %s for %s. The exact quantized source must be accessible. '
                           'If authorized access is required, provide a read-only HF_TOKEN as an Actions secret. '
                           'Do not rent a Pod or substitute video/P-H3 weights.' % (error.code, repo)) from None


def lock_row(row, metadata):
    revision = metadata.get('sha', '')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('A model repository did not provide an immutable revision.')
    matches = [s for s in metadata.get('siblings', []) if s.get('rfilename') == row['file']]
    if len(matches) != 1:
        raise ValueError('Required published file is missing: %s/%s. No alternative is downloaded.' % (row['repo'], row['file']))
    entry = matches[0]
    lfs = entry.get('lfs') or {}
    sha = lfs.get('sha256', '')
    size = lfs.get('size', entry.get('size', 0))
    if not re.fullmatch(r'[0-9a-f]{64}', sha) or not isinstance(size, int) or size <= 0:
        raise ValueError('No trustworthy LFS size/SHA-256 for ' + row['file'])
    return dict(row, revision=revision, sha256=sha, size_bytes=size)


def resolve(output, metadata_getter=None):
    source = read_json(PACKAGE / 'sources.json')
    getter = metadata_getter or hf_metadata
    metadata = {}
    locked = []
    for row in source['models']:
        if row['repo'] not in metadata:
            metadata[row['repo']] = getter(row['repo'], os.environ.get('HF_TOKEN'))
        locked.append(lock_row(row, metadata[row['repo']]))
    result = {'release': source['release'], 'comfy_commit': source['comfy_commit'], 'models': locked}
    atomic_json(output, result)
    print('Published assets resolved and locked. No model weights downloaded.')
    for row in locked:
        print('%s: %.2f GB at %s' % (row['id'], row['size_bytes'] / 1e9, row['revision']))
    print('Total model bytes: %.2f GB; an identical existing encoder can be reused.' % (sum(r['size_bytes'] for r in locked) / 1e9))
    return result


def verify(path, row, root):
    if not Path(path).is_file():
        return False
    stat = Path(path).stat()
    if stat.st_size != row['size_bytes']:
        return False
    marker = Path(root) / 'verified' / (row['id'] + '.json')
    signature = {'path': str(Path(path).resolve()), 'size': stat.st_size,
                 'mtime_ns': stat.st_mtime_ns, 'sha256': row['sha256']}
    try:
        if read_json(marker) == signature:
            return True
    except (OSError, ValueError):
        pass
    print('Checking SHA-256: ' + row['id'], flush=True)
    if digest(path) != row['sha256']:
        return False
    atomic_json(marker, signature)
    return True


def data_root():
    root = Path(os.environ.get('H3_IMAGE_ROOT', '/workspace/h3-image')).resolve()
    workspace = Path('/workspace').resolve()
    if root == workspace or workspace not in root.parents:
        raise ValueError('H3_IMAGE_ROOT must be a subdirectory of /workspace.')
    return root


def download():
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = PACKAGE / 'models.lock.json'
    if not lock_path.is_file():
        raise RuntimeError('Image has no resolved model manifest. Use the successfully built H3 Image image.')
    rows = read_json(lock_path)['models']
    from filelock import FileLock
    with FileLock(str(root / '.model-download.lock'), timeout=1):
        for row in rows:
            target = safe_destination(root / 'models', row['destination'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if verify(target, row, root):
                print('Using verified: ' + row['id'], flush=True)
                continue
            if target.exists() or target.is_symlink():
                raise RuntimeError('Existing file is unverified; it was not overwritten: ' + str(target))
            reuse = row.get('reuse')
            if reuse and verify(Path(reuse), row, root):
                target.symlink_to(Path(reuse).resolve())
                print('Reusing existing encoder without modifying it: ' + reuse, flush=True)
                continue
            cache = root / 'downloads' / row['id']
            expected_source = cache / row['file']
            if not verify(expected_source, row, root):
                if shutil.disk_usage(root).free < row['size_bytes'] + 4 * 1024 ** 3:
                    raise RuntimeError('Insufficient free /workspace storage for ' + row['id'] + '. Keep 4 GB extra headroom.')
                from huggingface_hub import hf_hub_download
                print('Downloading %s (%.2f GB)' % (row['id'], row['size_bytes'] / 1e9), flush=True)
                cache.mkdir(parents=True, exist_ok=True)
                downloaded = Path(hf_hub_download(repo_id=row['repo'], filename=row['file'], revision=row['revision'],
                    local_dir=str(cache), token=os.environ.get('HF_TOKEN') or None))
                if downloaded.resolve() != expected_source.resolve():
                    raise RuntimeError('Download returned an unexpected file path.')
                if not verify(downloaded, row, root):
                    raise RuntimeError('Checksum verification failed: ' + row['id'])
            target.symlink_to(expected_source.resolve())
        print('H3 Image model set ready.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('resolve')
    check.add_argument('--output', type=Path, default=PACKAGE / 'models.lock.json')
    commands.add_parser('download')
    args = parser.parse_args()
    if args.command == 'resolve':
        resolve(args.output)
    else:
        download()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('STOPPED: ' + str(error), file=sys.stderr)
        sys.exit(1)
