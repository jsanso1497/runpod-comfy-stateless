"""Prepare only the locked MOSS assets on persistent storage. Never deletes user data."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_child(root: Path, relative: str) -> Path:
    candidate = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Unsafe model path in manifest.')
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError('Model path escapes its storage directory.')
    return candidate


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2)
        temp = Path(stream.name)
    temp.replace(path)


def verified(path: Path, row: dict, marker: Path, force=False) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    expected = {'sha256': row['sha256'], 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'revision': row['revision']}
    if not force and marker.is_file():
        try:
            if json.loads(marker.read_text()) == expected:
                return True
        except (ValueError, OSError):
            pass
    print('Checking SHA-256:', row['id'], flush=True)
    if digest(path) != row['sha256']:
        return False
    atomic_json(marker, expected)
    return True


def prepare(root: Path, download: bool = True) -> None:
    root = root.resolve()
    for name in ('models', 'input', 'output', 'user', 'temp', '.verified'):
        (root / name).mkdir(parents=True, exist_ok=True)
    # Lock survives process crashes and prevents two pods writing the same cache concurrently.
    import fcntl
    with (root / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads((PACKAGE / 'models.lock.json').read_text())['models']
        for row in rows:
            dest = safe_child(root / 'models', row['destination'])
            marker = root / '.verified' / (row['id'] + '.json')
            force = os.environ.get('MOSS_VERIFY_ALL', '0') == '1'
            if verified(dest, row, marker, force):
                print('Cached and verified:', row['id'], flush=True)
                continue
            if dest.exists() or dest.is_symlink():
                raise RuntimeError('Model checksum mismatch: ' + str(dest) +
                                   '. Move this file aside and restart; it is not silently replaced.')
            if not download:
                raise FileNotFoundError('Missing model: ' + str(dest))
            required = int(row['approx_gb'] * 1_000_000_000) + 2_000_000_000
            if shutil.disk_usage(root).free < required:
                raise RuntimeError('Not enough free persistent disk for ' + row['id'])
            from huggingface_hub import hf_hub_download
            # local_dir holds the only copy. The model folder uses symlinks, not a duplicate cache.
            local_dir = root / 'models' / '.sources' / row['id']
            print('Preparing pinned model:', row['id'], flush=True)
            downloaded = Path(hf_hub_download(repo_id=row['repo'], filename=row['file'],
                              revision=row['revision'], local_dir=str(local_dir),
                              token=os.environ.get('HF_TOKEN') or None))
            if not verified(downloaded, row, marker, force=True):
                raise RuntimeError('Downloaded model failed SHA-256 verification: ' + row['id'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(os.path.relpath(downloaded, dest.parent))
        factory = root / 'user' / 'default' / 'workflows' / '00_MOSS_Mobile_HQ_v1'
        factory.mkdir(parents=True, exist_ok=True)
        for graph in sorted((PACKAGE / 'workflows').glob('*.json')):
            target = factory / graph.name
            if not target.exists():
                shutil.copy2(graph, target)
        # New factory versions get a new folder; saved personal graphs are never overwritten.
        import yaml
        config = {'moss_mobile': {'base_path': str(root / 'models'), 'is_default': True,
                                 **{name: name for name in ('diffusion_models', 'text_encoders', 'vae', 'loras')}}}
        (root / 'extra_model_paths.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        print('MOSS Mobile assets and native apps are ready.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('MOSS_DATA_ROOT', '/workspace/moss-mobile')))
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    prepare(args.root, not args.offline)


if __name__ == '__main__':
    main()
