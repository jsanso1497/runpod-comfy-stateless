"""Resumable, official-host-only downloads at immutable revisions; verified SHA256."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil

HERE = Path(__file__).resolve().parent
ALLOWED_REPOS = {'black-forest-labs/FLUX.2-klein-9B', 'Comfy-Org/vae-text-encorder-for-flux-klein-9b'}


def validate_manifest(rows):
    destinations = set()
    for row in rows:
        if row['repo_id'] not in ALLOWED_REPOS or not re.fullmatch('[0-9a-f]{40}', row['revision']):
            raise ValueError('Models must use approved official repositories at immutable commits.')
        for field in ('filename', 'destination'):
            p = PurePosixPath(row[field])
            if p.is_absolute() or '..' in p.parts or '\\' in row[field] or ':' in row[field]:
                raise ValueError('Unsafe model path.')
        if not row['destination'].startswith('models/') or not row['destination'].endswith('.safetensors'):
            raise ValueError('Model destination must be a safetensors file under models/.')
        if row['destination'] in destinations:
            raise ValueError('Duplicate model destination.')
        destinations.add(row['destination'])
        if row.get('sha256') and not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid SHA256.')
    return rows


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_models(home, manifest=HERE/'config/models.json'):
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    rows = validate_manifest(json.loads(Path(manifest).read_text()))
    home = Path(home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    token = os.environ.get('HF_TOKEN')
    for row in rows:
        dst = (home / row['destination']).resolve()
        if not dst.is_relative_to(home):
            raise ValueError('Destination escaped ComfyUI home through a symlink.')
        try:
            meta = get_hf_file_metadata(hf_hub_url(row['repo_id'], row['filename'], revision=row['revision']), token=token)
        except Exception as exc:
            raise RuntimeError(f"Cannot access {row['name']}. Accept BFL's model agreement and supply "
                               'HF_TOKEN with read access. Check connectivity and the startup log. '
                               'No alternate model or mirror is substituted.') from exc
        if meta.commit_hash != row['revision']:
            raise ValueError('Hugging Face returned an unexpected model revision.')
        expected = (meta.etag or '').strip('"')
        if not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Expected a SHA256 LFS ETag for the large model file.')
        if row.get('sha256') and row['sha256'] != expected:
            raise ValueError('Published model digest does not match the recorded digest.')
        if dst.is_file() and dst.stat().st_size == meta.size and digest(dst) == expected:
            print('VERIFIED existing:', row['name'], flush=True)
            continue
        # Fail before a very large download, retaining 10 GiB for uploads/outputs.
        needed = int(meta.size or row['estimated_bytes']) + 10 * 1024**3
        if shutil.disk_usage(home).free < needed:
            raise RuntimeError(f"Not enough free disk for {row['name']}; need model size plus 10 GiB headroom.")
        print('DOWNLOADING:', row['name'], 'revision=' + row['revision'], flush=True)
        staging = home / '.flux-photo-downloads' / Path(row['destination']).stem
        file = Path(hf_hub_download(repo_id=row['repo_id'], filename=row['filename'],
                                   revision=row['revision'], token=token, local_dir=staging))
        if file.stat().st_size != meta.size or digest(file) != expected:
            file.unlink(missing_ok=True)
            raise ValueError('Model integrity failure. Bad download removed; restart to retry.')
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(file, dst)
        record = {'repo_id': row['repo_id'], 'revision': row['revision'],
                  'filename': row['filename'], 'sha256': expected, 'bytes': meta.size}
        dst.with_suffix('.verified.json').write_text(json.dumps(record, indent=2) + '\n')
        print('VERIFIED:', row['name'], flush=True)
