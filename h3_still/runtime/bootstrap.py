"""Persistent, verified downloads for exactly the developer's four model files."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]
FACTORY = '00_H3_Still_Developer_2026_11_05_v1'


def safe_child(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative or not p.parts:
        raise ValueError('Unsafe relative path.')
    result = root / p
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes persistent storage.')
    return result


def atomic_json(path: Path, value) -> None:
    if path.is_symlink():
        raise ValueError('Refusing a symlinked metadata file.')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as file:
            tmp = Path(file.name)
            json.dump(value, file, indent=2)
            file.write('\n')
        tmp.replace(path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified(path: Path, row: dict, marker: Path, force: bool = False) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    expected = {'sha256': row['sha256'], 'size': stat.st_size,
                'mtime_ns': stat.st_mtime_ns, 'revision': row['revision']}
    if marker.is_symlink():
        raise ValueError('Verification marker must not be a symlink.')
    if marker.is_file() and not force:
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


def seed_workflows(root: Path) -> None:
    factory = safe_child(root, 'user/default/workflows/' + FACTORY)
    factory.mkdir(parents=True, exist_ok=True)
    for source in sorted((PACKAGE / 'workflows').glob('*.json')):
        target = safe_child(root, str(factory.relative_to(root) / source.name))
        # Exclusive create means even user edits to factory copies are preserved.
        try:
            with target.open('xb') as file:
                file.write(source.read_bytes())
        except FileExistsError:
            pass


def prepare(root: Path, download: bool = True) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ('models', 'input', 'output', 'user', 'temp', '.verified'):
        safe_child(root, name).mkdir(parents=True, exist_ok=True)
    import fcntl
    lock_path = safe_child(root, '.prepare.lock')
    if lock_path.is_symlink():
        raise ValueError('Prepare lock must not be a symlink.')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads((PACKAGE / 'models.lock.json').read_text())['models']
        for row in rows:
            dest = safe_child(root / 'models', row['destination'])
            marker = safe_child(root, '.verified/' + row['id'] + '.json')
            force = os.environ.get('H3_STILL_VERIFY_ALL', '0') == '1'
            if verified(dest, row, marker, force):
                print('Using verified cache:', row['id'], flush=True)
                continue
            if dest.exists() or dest.is_symlink():
                raise RuntimeError(f'Checksum mismatch: {dest}. Move that file aside and restart. It was not overwritten.')
            if not download:
                raise FileNotFoundError('Missing model: ' + str(dest))
            local_dir = safe_child(root / 'models', '.sources/' + row['id'])
            local_dir.mkdir(parents=True, exist_ok=True)
            expected_file = safe_child(local_dir, row['file'])
            # A completed source download can survive a crash before linking.
            # Do not require a second full model's free space in that case.
            if not verified(expected_file, row, marker, force):
                required = int(row['approx_gb'] * 1_000_000_000) + 2_000_000_000
                if shutil.disk_usage(root).free < required:
                    raise RuntimeError('Insufficient free persistent disk for ' + row['id'] + '. Allow 100 GB total.')
                from huggingface_hub import hf_hub_download
                print('Downloading pinned asset:', row['id'], flush=True)
                downloaded = Path(hf_hub_download(
                    repo_id=row['repo'], filename=row['file'], revision=row['revision'],
                    local_dir=str(local_dir), token=os.environ.get('HF_TOKEN') or None))
                if not downloaded.resolve().is_relative_to(local_dir.resolve()):
                    raise RuntimeError('Model download escaped the expected source directory.')
                if not verified(downloaded, row, marker, True):
                    raise RuntimeError('Downloaded SHA-256 does not match the lock: ' + row['id'])
                expected_file = downloaded
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(os.path.relpath(expected_file, dest.parent))
        seed_workflows(root)
        import yaml
        config = {'h3_still': {'base_path': str(root / 'models'), 'is_default': True,
                              **{name: name for name in ('diffusion_models', 'text_encoders', 'vae', 'loras')}}}
        path = safe_child(root, 'extra_model_paths.yaml')
        if path.is_symlink():
            raise ValueError('Model-path config must not be a symlink.')
        path.write_text(yaml.safe_dump(config), encoding='utf-8')
        print('Verified assets and four developer-based workflows are ready.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/h3-still'))
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    prepare(args.root, not args.offline)


if __name__ == '__main__':
    main()
