"""Verified small companion files from a fixed public model revision only.

Separate from the private LoRA/checkpoint downloader; its URL rules are unchanged.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from urllib.parse import quote
from . import provider_downloads as pd

ALLOWED = {'system_prompt.txt', 'model.safetensors.index.json'}
MAX_BYTES = 2 * 1024 * 1024


def prepare_companion(asset, target, transport):
    if (asset.get('group') != 'qwen_pe_i2i' or asset.get('repo_id') != 'Qwen/Qwen-Image-2.1-PE-I2I'
            or asset.get('filename') not in ALLOWED or not re.fullmatch('[0-9a-f]{40}', asset.get('revision', ''))):
        raise pd.LinkError('Companion source is not in the fixed public allowlist.')
    if target.is_symlink():
        raise pd.LinkError('Refusing a symlinked companion file.')
    target.parent.mkdir(parents=True, exist_ok=True)
    revision = asset['revision']
    metadata = transport.json(f'https://huggingface.co/api/models/{asset["repo_id"]}/revision/{revision}?blobs=true')
    if metadata.get('sha') != revision:
        raise pd.LinkError('The companion repository revision did not match the pinned source.')
    entries = [f for f in metadata.get('siblings', []) if f.get('rfilename') == asset['filename']]
    if len(entries) != 1:
        raise pd.LinkError('Required companion is absent from the pinned model revision.')
    source = entries[0]
    size = source.get('size', 0)
    blob = source.get('blobId', '')
    if type(size) is not int or not 0 < size <= MAX_BYTES or not re.fullmatch('[0-9a-f]{40}', blob):
        raise pd.LinkError('Companion metadata lacks a valid size or Git-blob checksum.')
    row = {'git_blob_sha1': blob}
    if not pd.verified(target, row):
        url = f'https://huggingface.co/{asset["repo_id"]}/resolve/{revision}/{quote(asset["filename"])}'
        with transport.request('GET', url, stream=True, redirects=True) as response:
            pd.status_error(response.status_code)
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > size or len(data) > MAX_BYTES:
                    raise pd.LinkError('Companion transfer exceeded its declared size.')
        if len(data) != size:
            raise pd.LinkError('Companion transfer was incomplete.')
        import hashlib
        if hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() != blob:
            raise pd.LinkError('Companion checksum failed.')
        try:
            text = data.decode('utf-8')
            if asset['filename'].endswith('.json'):
                value = json.loads(text)
                if not isinstance(value.get('weight_map'), dict) or not value['weight_map']:
                    raise ValueError()
            elif not text.strip() or '<html' in text[:100].lower():
                raise ValueError()
        except (UnicodeDecodeError, ValueError, AttributeError):
            raise pd.LinkError('Companion file contents were invalid.') from None
        import tempfile
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            handle.write(data)
            temp = Path(handle.name)
        temp.chmod(0o644)
        temp.replace(target)
    return {'bytes': size, 'format': 'utf-8', 'git_blob_sha1': blob}
