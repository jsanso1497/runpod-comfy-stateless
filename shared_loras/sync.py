#!/usr/bin/env python3
"""Download a shared LoRA library from one Civitai/Hugging Face URL per line.

No user hashes, model IDs, profiles or JSON entries. Provider metadata supplies
file selection and checksums. Downloads are never applied to a model by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import struct
import sys
import time
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import requests

VERSION = '1.0.0'
PROVIDERS = {'civitai.com', 'www.civitai.com', 'huggingface.co', 'www.huggingface.co'}
SECRET_PARAMS = {'token', 'access_token', 'api_key', 'apikey', 'authorization', 'auth'}
SHA256 = re.compile(r'^[0-9a-f]{64}$', re.I)
SHA1 = re.compile(r'^[0-9a-f]{40}$', re.I)
RESERVE = 8 * 1024**3


class LinkError(ValueError):
    """An actionable, credential-free error safe to put in a startup log."""


def _positive_int(value, label):
    if isinstance(value, bool) or not str(value).isdigit() or int(value) < 1:
        raise LinkError(f'Invalid {label} in the link. Copy the original provider link again.')
    return int(value)


def clean_url(value):
    value = str(value).strip().removeprefix('<').removesuffix('>')
    try:
        p = urlsplit(value)
        port = p.port
    except ValueError:
        raise LinkError('Malformed URL. Paste one complete HTTPS provider link per line.') from None
    if p.scheme != 'https' or p.hostname not in PROVIDERS or port not in (None, 443) or p.username or p.password:
        raise LinkError('Use an HTTPS civitai.com or huggingface.co link, without embedded credentials.')
    if any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise LinkError('Spaces must be URL-encoded. Copy the link from the browser or download button.')
    query = parse_qs(p.query, keep_blank_values=True)
    if any(k.casefold() in SECRET_PARAMS for k in query):
        raise LinkError('This link contains an API key/token. Remove it; use CIVITAI_TOKEN or HF_TOKEN in RunPod Secrets.')
    host = p.hostname.removeprefix('www.')
    return urlunsplit(('https', host, p.path.rstrip('/'), p.query, ''))


def read_links(path, errors=None):
    path = Path(path)
    if not path.is_file():
        raise LinkError('Missing shared LoRA list: config/lora_links.txt. Upload the included file.')
    links, seen = [], set()
    for number, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith('#'):
            continue
        try:
            url = clean_url(text)
        except LinkError as e:
            if errors is not None:
                errors.append({'line':number,'error':str(e)})
                continue
            raise LinkError(f'lora_links.txt line {number}: {e}') from None
        if url not in seen:
            seen.add(url)
            links.append((number, url))
    return links


def _headers(url, auth_host):
    headers = {'User-Agent': 'RunPod-Shared-LoRAs/' + VERSION, 'Accept-Encoding': 'identity'}
    host = urlsplit(url).hostname
    # Each redirect request is built afresh. Never forward a provider token to its CDN.
    if host == auth_host:
        env = 'HF_TOKEN' if host == 'huggingface.co' else 'CIVITAI_TOKEN' if host == 'civitai.com' else ''
        token = os.environ.get(env, '').strip() if env else ''
        if token:
            if token.startswith('{{'):
                raise LinkError(f'{env} did not resolve. Check its RunPod Secret reference.')
            headers['Authorization'] = 'Bearer ' + token
    return headers


def _check_public_https(url):
    p = urlsplit(url)
    try:
        port = p.port
    except ValueError:
        raise LinkError('Provider returned an invalid download address.') from None
    if p.scheme != 'https' or not p.hostname or p.username or p.password or port not in (None, 443):
        raise LinkError('Provider returned a non-HTTPS or credential-bearing download address.')
    try:
        addresses = socket.getaddrinfo(p.hostname, 443, type=socket.SOCK_STREAM)
    except OSError:
        raise LinkError('Could not resolve the provider/download host. Retry on a working network.') from None
    if not addresses or any(not ipaddress.ip_address(x[4][0]).is_global for x in addresses):
        raise LinkError('Refusing a download redirect to a private or local network address.')


class Transport:
    def __init__(self):
        self.session = requests.Session()
        self.session.trust_env = False

    def close(self):
        self.session.close()

    def request(self, method, url, *, stream=False, headers=None, redirects=False):
        auth_host = urlsplit(url).hostname
        current = url
        for _ in range(8):
            _check_public_https(current)
            request_headers = _headers(current, auth_host)
            request_headers.update(headers or {})
            self.session.cookies.clear()
            response = None
            for attempt in range(3):
                try:
                    response = self.session.request(method, current, headers=request_headers,
                        timeout=(20, 120), stream=stream, allow_redirects=False)
                except requests.RequestException:
                    if attempt == 2:
                        raise LinkError('Provider network request failed. Check connectivity and retry.') from None
                    time.sleep(2**attempt)
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 2:
                        response.close()
                        time.sleep(2**attempt)
                        continue
                break
            if response.status_code in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                response.close()
                if not redirects or not location:
                    raise LinkError('Unexpected redirect while reading provider metadata.')
                current = urljoin(current, location)
                continue
            return response
        raise LinkError('Provider sent too many download redirects.')

    def json(self, url):
        with self.request('GET', url, stream=True) as response:
            status_error(response.status_code)
            data = bytearray()
            for block in response.iter_content(65536):
                data.extend(block)
                if len(data) > 16 * 1024 * 1024:
                    raise LinkError('Provider metadata is too large. Use a specific file link.')
            try:
                return json.loads(data)
            except (ValueError, UnicodeDecodeError):
                raise LinkError('Provider returned a web page instead of metadata. Check access and the link.') from None


def status_error(code):
    if code in (200, 206):
        return
    if code in (401, 403):
        raise LinkError(f'Provider HTTP {code}: check your RunPod token and accept/request access on the model page.')
    if code == 404:
        raise LinkError('Provider HTTP 404: model/version/file not found or not accessible with your token.')
    raise LinkError(f'Provider HTTP {code}. Retry later or check the selected link.')


def _safe_name(name):
    stem = PurePosixPath(str(name).replace('\\', '/')).name
    stem = re.sub(r'\.safetensors$', '', stem, flags=re.I)
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', stem).strip('._')[:100]
    return stem or 'lora'


def _civitai_file(version, query):
    files = [f for f in version.get('files', []) if isinstance(f, dict)
             and str(f.get('name', '')).lower().endswith('.safetensors')]
    requested = query.get('fileid') or query.get('file_id')
    if requested:
        fid = _positive_int(requested[-1], 'file ID')
        files = [f for f in files if f.get('id') == fid]
    # Honor the exact variant encoded by Civitai's Download button.
    for selector in ('fp', 'size', 'format', 'type'):
        if selector not in query:
            continue
        target = query[selector][-1].casefold()
        if selector == 'format' and target in ('safetensor', 'safetensors'):
            continue
        files = [f for f in files if str(f.get('type', '') if selector == 'type'
                  else f.get('metadata', {}).get(selector, '')).casefold() == target]
    primary = [f for f in files if f.get('primary') is True]
    if len(files) == 1:
        return files[0]
    if len(primary) == 1:
        return primary[0]
    if not files:
        raise LinkError('No matching safetensors file. Copy the desired LoRA file\'s Download link.')
    raise LinkError('Multiple safetensors files and no single primary file. Paste the chosen file\'s Download link instead.')


def resolve_civitai(url, transport):
    p = urlsplit(url)
    query = {k.casefold(): v for k, v in parse_qs(p.query).items()}
    main_match = re.fullmatch(r'/models/(\d+)(?:/[^/]+)?', p.path)
    download_match = re.fullmatch(r'/api/download/models/(\d+)', p.path)
    version_match = re.fullmatch(r'/api/v1/model-versions/(\d+)', p.path)
    page_model = int(main_match.group(1)) if main_match else None
    if download_match or version_match:
        version_id = int((download_match or version_match).group(1))
    elif main_match:
        if query.get('modelversionid'):
            version_id = _positive_int(query['modelversionid'][-1], 'version ID')
        else:
            model = transport.json(f'https://civitai.com/api/v1/models/{page_model}')
            versions = [v for v in model.get('modelVersions', []) if isinstance(v, dict)
                        and v.get('status', 'Published') == 'Published' and v.get('files')]
            if not versions:
                raise LinkError('No downloadable published version on that Civitai page.')
            # Civitai normally orders versions newest first; timestamps remove ambiguity.
            versions = sorted(versions, key=lambda v: str(v.get('publishedAt') or v.get('createdAt') or ''), reverse=True)
            version_id = _positive_int(versions[0].get('id'), 'version ID')
    else:
        raise LinkError('Use a Civitai model page, version-specific page, or file Download link.')
    version = transport.json(f'https://civitai.com/api/v1/model-versions/{version_id}')
    if version.get('id') != version_id or (page_model is not None and version.get('modelId') != page_model):
        raise LinkError('Civitai model/version did not match the supplied page. Copy the link again.')
    kind = str(version.get('model', {}).get('type', '')).casefold()
    if kind and kind not in ('lora', 'locon', 'dora', 'lycoris'):
        raise LinkError('This Civitai link is not a LoRA. Base models stay in the model catalog.')
    chosen = _civitai_file(version, query)
    sha = str(chosen.get('hashes', {}).get('SHA256', '')).lower()
    if not SHA256.fullmatch(sha):
        raise LinkError('Civitai did not publish a full file checksum. Cannot verify this download.')
    dl = clean_url(chosen.get('downloadUrl', ''))
    if urlsplit(dl).hostname != 'civitai.com' or not urlsplit(dl).path.startswith('/api/download/models/'):
        raise LinkError('Civitai metadata did not supply an official file download link.')
    fid = _positive_int(chosen.get('id'), 'file ID')
    kb = float(chosen.get('sizeKB', 0))
    if not math.isfinite(kb) or kb <= 0:
        raise LinkError('Civitai did not publish a valid file size.')
    size = math.ceil(kb * 1024)
    if size <= 0:
        raise LinkError('Civitai did not publish a valid file size.')
    triggers = ', '.join(x for x in version.get('trainedWords', []) if isinstance(x, str))
    return {'provider': 'civitai', 'source_url': url, 'download_url': dl,
            'name': str(version.get('model', {}).get('name') or chosen['name']),
            'lora_name': f'Shared/{_safe_name(chosen["name"])}__civitai_{version_id}_{fid}.safetensors',
            'sha256': sha, 'git_blob_sha1': '', 'estimated_bytes': size,
            'trigger_words': triggers, 'reported_base_model': version.get('baseModel', ''),
            'version_id': version_id, 'file_id': fid}


def parse_hf_url(url):
    p = urlsplit(url)
    parts = p.path.strip('/').split('/')
    if len(parts) < 2 or parts[0] in ('datasets', 'spaces'):
        raise LinkError('Use a Hugging Face model repository or its safetensors file link.')
    owner, name = map(unquote, parts[:2])
    if not all(re.fullmatch(r'[A-Za-z0-9_.-]+', x) and x not in ('.', '..') for x in (owner, name)):
        raise LinkError('Invalid Hugging Face repository link.')
    repo = owner + '/' + name
    if len(parts) == 2:
        return repo, 'main', None
    if parts[2] not in ('blob', 'resolve', 'tree') or len(parts) < 4:
        raise LinkError('Use the Hugging Face repository page or the exact safetensors file page.')
    revision = unquote(parts[3])
    filename = unquote('/'.join(parts[4:]))
    if not revision or any(x in ('.', '..') for x in revision.split('/')):
        raise LinkError('Invalid Hugging Face revision.')
    if parts[2] == 'tree' and not filename:
        return repo, revision, None
    if parts[2] == 'tree':
        raise LinkError('This is a folder link. Open the desired safetensors file and paste its page link.')
    if not filename.lower().endswith('.safetensors') or '\\' in filename or any(x in ('', '.', '..') for x in filename.split('/')):
        raise LinkError('Open the LoRA .safetensors file on Hugging Face and copy that page link.')
    return repo, revision, filename


def resolve_hf(url, transport):
    repo, revision, filename = parse_hf_url(url)
    metadata = transport.json(f'https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe="")}?blobs=true')
    commit = str(metadata.get('sha', ''))
    if not SHA1.fullmatch(commit):
        raise LinkError('Hugging Face did not provide a resolved repository revision.')
    files = [x for x in metadata.get('siblings', []) if isinstance(x, dict)
             and str(x.get('rfilename', '')).lower().endswith('.safetensors')]
    if filename is None:
        if len(files) != 1:
            raise LinkError('This Hugging Face repo has multiple or no safetensors files. Open the desired LoRA file and paste its page link.')
        selected = files[0]
        filename = selected['rfilename']
    else:
        files = [x for x in files if x.get('rfilename') == filename]
        if len(files) != 1:
            raise LinkError('That safetensors file is not present in the Hugging Face revision.')
        selected = files[0]
    # LFS/Xet file SHA-256 and Git blob ID are different hashes. Never confuse them.
    sha = str((selected.get('lfs') or {}).get('sha256', '')).lower()
    blob = str(selected.get('blobId', '')).lower()
    size = (selected.get('lfs') or {}).get('size', selected.get('size', 0))
    if sha and not SHA256.fullmatch(sha):
        raise LinkError('Invalid provider file checksum.')
    if not sha and not SHA1.fullmatch(blob):
        raise LinkError('Hugging Face supplied no verifiable file checksum.')
    if not isinstance(size, int) or size <= 0:
        raise LinkError('Hugging Face supplied no valid file size.')
    filekey = hashlib.sha256((repo + '/' + filename).encode()).hexdigest()[:10]
    card = metadata.get('cardData') or {}
    base = card.get('base_model', '') if isinstance(card, dict) else ''
    if isinstance(base, list):
        base = ', '.join(str(x) for x in base)
    return {'provider': 'huggingface', 'source_url': url,
            'download_url': f'https://huggingface.co/{repo}/resolve/{commit}/{quote(filename, safe="/")}',
            'name': filename, 'lora_name': f'Shared/{_safe_name(filename)}__hf_{filekey}_{(sha or blob)[:12]}.safetensors',
            'sha256': sha, 'git_blob_sha1': '' if sha else blob, 'estimated_bytes': size,
            'trigger_words': '', 'reported_base_model': str(base), 'revision': commit, 'repo_id': repo}


def resolve_link(url, transport):
    url = clean_url(url)
    return resolve_hf(url, transport) if urlsplit(url).hostname == 'huggingface.co' else resolve_civitai(url, transport)


def file_hash(path, kind='sha256'):
    if kind == 'git':
        digest = hashlib.sha1()
        digest.update(f'blob {Path(path).stat().st_size}\0'.encode())
    else:
        digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def verified(path, row):
    if not Path(path).is_file():
        return False
    if row.get('sha256'):
        return file_hash(path) == row['sha256']
    return file_hash(path, 'git') == row['git_blob_sha1']


def check_adapter_file(path):
    """Check safetensors structure/adapter-like keys without deserializing tensors."""
    with Path(path).open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise LinkError('Downloaded file is not safetensors.')
        length = struct.unpack('<Q', raw)[0]
        if not 2 <= length <= 32 * 1024 * 1024 or length > Path(path).stat().st_size - 8:
            raise LinkError('Downloaded file is not a valid safetensors file.')
        try:
            metadata = json.loads(f.read(length))
        except (ValueError, UnicodeDecodeError):
            raise LinkError('Invalid safetensors header.') from None
    if not isinstance(metadata, dict):
        raise LinkError('Invalid safetensors header.')
    # Reject obvious base checkpoints accidentally pasted into the LoRA list.
    keys = [k.casefold() for k in metadata if k != '__metadata__']
    patterns = ('lora_', '.lora.', 'hada_', 'lokr_', 'oft_blocks', 'boft', 'dora_scale', '.diff', '.set')
    if not keys or not any(any(x in k for x in patterns) for k in keys):
        raise LinkError('Safetensors file has no recognized adapter keys. Check that the link is a LoRA, not a base checkpoint.')
    from safetensors import safe_open
    try:
        with safe_open(str(path), framework='numpy') as f:
            if not list(f.keys()):
                raise LinkError('Empty safetensors adapter.')
    except Exception:
        raise LinkError('Safetensors file failed structural validation.') from None


def _destination(root, name):
    base = (Path(root) / 'models' / 'loras').resolve()
    target = base / name
    if target.is_symlink() or not target.resolve().is_relative_to(base) or target.resolve() == base:
        raise LinkError('Unsafe LoRA destination.')
    return target


def download(row, root, transport):
    target = _destination(root, row['lora_name'])
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.part')
    if partial.is_symlink():
        raise LinkError('Refusing a symlinked partial LoRA file.')
    if verified(target, row):
        check_adapter_file(target)
        return target
    last = 'download incomplete'
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset and verified(partial, row):
            check_adapter_file(partial)
            partial.replace(target)
            return target
        if shutil.disk_usage(target.parent).free < max(0, row['estimated_bytes'] - offset) + RESERVE:
            raise LinkError('Not enough temporary disk for this LoRA plus 8 GiB headroom. Increase container disk or remove a link.')
        try:
            with transport.request('GET', row['download_url'], stream=True,
                    headers={'Range': f'bytes={offset}-'} if offset else {}, redirects=True) as response:
                if response.status_code == 416:
                    partial.unlink(missing_ok=True)
                    raise LinkError('Server range changed; restarting this file.')
                status_error(response.status_code)
                if response.status_code == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise LinkError('Unexpected download range.')
                    mode = 'ab'
                else:
                    offset, mode = 0, 'wb'
                    if shutil.disk_usage(target.parent).free < row['estimated_bytes'] + RESERVE:
                        raise LinkError('Server restarted download; insufficient temporary disk headroom.')
                if any(x in response.headers.get('Content-Type', '').lower() for x in ('text/html', 'application/json')):
                    raise LinkError('Provider returned a login/error page instead of model bytes.')
                count, last_log = offset, time.monotonic()
                with partial.open(mode) as handle:
                    for block in response.iter_content(4 * 1024 * 1024):
                        if not block:
                            continue
                        handle.write(block)
                        count += len(block)
                        if count > row['estimated_bytes'] * 1.02 + 1024 * 1024:
                            raise LinkError('Download exceeds the provider file size; refusing unexpected content.')
                        if time.monotonic() - last_log >= 15:
                            print(f'LoRA {target.name}: {count/2**30:.2f} GiB', flush=True)
                            last_log = time.monotonic()
                        if shutil.disk_usage(target.parent).free < RESERVE:
                            raise LinkError('Stopped LoRA download to preserve temporary disk headroom.')
            if not verified(partial, row):
                partial.unlink(missing_ok=True)
                raise LinkError('Provider checksum mismatch; bad bytes discarded.')
            check_adapter_file(partial)
            partial.replace(target)
            return target
        except (LinkError, OSError, requests.RequestException) as exc:
            last = str(exc) if isinstance(exc, LinkError) else 'Network or local file error during download.'
            if attempt < 2:
                time.sleep(attempt + 1)
    raise LinkError(last)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise LinkError('Refusing a symlinked library index.')
    temp = path.with_suffix(path.suffix + '.tmp')
    if temp.is_symlink():
        raise LinkError('Refusing a symlinked library index temporary file.')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sync_library(links_path, comfy_home, transport=None):
    own_transport = transport is None
    transport = transport or Transport()
    rows, failures, seen = [], [], set()
    try:
        try:
            links = read_links(links_path, errors=failures)
        except LinkError as exc:
            links = []
            failures.append({'line': None, 'error': str(exc)})
        for failure in failures:
            print(f'SHARED LORA WARNING: lora_links.txt line {failure["line"]}: {failure["error"]}',file=sys.stderr,flush=True)
        for number, url in links:
            try:
                row = resolve_link(url, transport)
                key = row.get('sha256') or row.get('git_blob_sha1')
                if key in seen:
                    continue
                path = download(row, comfy_home, transport)
                row['sha256'] = file_hash(path)
                # No expiring signed CDN URLs, passwords, or tokens in the persistent report.
                row.pop('download_url', None)
                rows.append(row)
                seen.add(key)
                print(f'SHARED LORA READY: {row["lora_name"]}', flush=True)
            except (LinkError, OSError, ValueError, TypeError, KeyError):
                exc = sys.exc_info()[1]
                message = str(exc) if isinstance(exc, LinkError) else 'Provider metadata or local file error; check the link.'
                failures.append({'line': number, 'error': message})
                print(f'SHARED LORA WARNING: lora_links.txt line {number}: {message}', file=sys.stderr, flush=True)
    finally:
        if own_transport:
            transport.close()
    root = Path(comfy_home)
    atomic_json(root / 'models/loras/link-library.json', {'version': VERSION, 'loras': rows, 'failures': failures})
    print(f'SHARED LORA LIBRARY: {len(rows)} ready, {len(failures)} failed. Downloads are not auto-applied.', flush=True)
    return rows, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--links', type=Path, required=True)
    parser.add_argument('--comfy-home', type=Path, default=Path('/workspace/ComfyUI'))
    parser.add_argument('--check', action='store_true', help='Validate link syntax only; no downloads or tokens required.')
    args = parser.parse_args()
    if args.check:
        print(f'SHARED LORA LINK SYNTAX PASS: {len(read_links(args.links))} unique links')
    else:
        sync_library(args.links, args.comfy_home)


if __name__ == '__main__':
    try:
        main()
    except LinkError as exc:
        print('SHARED LORA ERROR: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
