"""Observable, serialized asset preparation with credential-free progress.

No URLs, request headers, private filenames, or provider error bodies enter the
progress report. Existing provider selection and precision rules remain in assets.
"""
from __future__ import annotations
import asyncio
import contextvars
import hashlib
import json
from pathlib import Path
import threading
import time

_CURRENT = contextvars.ContextVar('workbench_preparation_progress', default=None)


class Progress:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = {'stage': 'starting', 'current_asset': None,
                      'bytes_done': 0, 'bytes_total': 0,
                      'assets_done': 0, 'assets_total': 0,
                      'last_progress_at': time.time()}
        self.hashes = {}

    def update(self, **changes):
        with self.lock:
            changed = any(self.value.get(k) != v for k, v in changes.items())
            self.value.update(changes)
            if changed:
                self.value['last_progress_at'] = time.time()

    def snapshot(self):
        with self.lock:
            out = dict(self.value)
        out['seconds_since_progress'] = round(max(0, time.time()-out['last_progress_at']), 1)
        total = out.get('bytes_total', 0)
        out['asset_percent'] = round(100*out['bytes_done']/total, 1) if total else None
        return out


def update(**fields):
    progress = _CURRENT.get()
    if progress is not None:
        progress.update(**fields)


def tracked_hash(path, kind='sha256'):
    """Hash in bounded chunks, report reads, and cache only within one job."""
    p = Path(path)
    st = p.stat()
    progress = _CURRENT.get()
    key = (str(p.resolve()), st.st_dev, st.st_ino, st.st_size,
           st.st_mtime_ns, st.st_ctime_ns, kind)
    if progress is not None and key in progress.hashes:
        return progress.hashes[key]
    digest = hashlib.sha1() if kind == 'git' else hashlib.sha256()
    if kind == 'git':
        digest.update(f'blob {st.st_size}\0'.encode())
    update(stage='verifying checksum', bytes_done=0, bytes_total=st.st_size)
    done = 0
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
            done += len(block)
            update(bytes_done=done)
    after = p.stat()
    if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
            st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns):
        raise ValueError('Model file changed during checksum verification; retry preparation.')
    result = digest.hexdigest()
    if progress is not None:
        progress.hashes[key] = result
    return result


def _read_status(path):
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeError):
        # Nothing in a malformed status file establishes readiness.
        return {}
    if not isinstance(data, dict):
        return {}
    for key in ('installed', 'failures', 'models'):
        if key in data:
            rows = data[key] if isinstance(data[key], list) else []
            required = ('name',) if key == 'models' else (('id', 'destination') if key == 'installed' else ('id', 'error'))
            data[key] = [row for row in rows if isinstance(row, dict) and
                         all(isinstance(row.get(field), str) for field in required)]
    return data


def prepare_standard_catalog(ns, groups):
    """Use the original downloader, but publish readiness after EACH asset."""
    read = ns['read_catalog']
    error = ns['AssetError']
    pd = ns['pd']
    assets = read('assets.json')['assets']
    known = {a['group'] for a in assets} | {'ollama'}
    if set(groups)-known:
        raise error('Unknown model group requested.')
    selected = [a for a in assets if a['group'] in groups]
    path = ns['state_root']()/'asset-status.json'
    previous = _read_status(path)
    lookup = {r['id']: r for r in previous.get('installed', []) if isinstance(r, dict) and 'id' in r}
    failed = {r['id']: r for r in previous.get('failures', []) if isinstance(r, dict) and 'id' in r}
    current_failures = []
    transport = pd.Transport()

    def persist():
        ns['atomic_json'](path, {'installed': list(lookup.values()),
            'failures': list(failed.values()), 'last_groups': groups,
            'updated': time.time()})
    update(assets_total=len(selected), assets_done=0)
    try:
        for index, asset in enumerate(selected):
            ident = asset['id']
            update(current_asset=ident, stage='resolving metadata', bytes_done=0, bytes_total=0)
            print('Preparing standard asset: '+ident, flush=True)
            # Old readiness must not survive a failed revalidation.
            lookup.pop(ident, None)
            failed.pop(ident, None)
            persist()
            try:
                target = ns['safe_path'](ns['model_root'](), asset['destination'])
                precision = ns['_standard_asset_high_precision'](asset)
                if asset.get('format') == 'utf-8':
                    from .public_companions import prepare_companion
                    info = prepare_companion(asset, target, transport)
                    resolved = asset['revision']
                elif asset.get('sha256') and target.is_file() and pd.file_hash(target) == asset['sha256']:
                    info = ns['inspect_safetensors'](target, high_precision=precision)
                    resolved = asset.get('revision')
                else:
                    row = ns['resolve_standard'](asset, transport)
                    info = ns['transfer'](row, target, transport, high_precision=precision)
                    resolved = row.get('revision')
                lookup[ident] = {'id': ident, 'group': asset['group'],
                    'destination': asset['destination'], 'sha256': pd.file_hash(target),
                    'resolved_revision': resolved, 'inspection': info,
                    'verified_at': time.time()}
                print('Asset ready: '+ident, flush=True)
            except (error, pd.LinkError, OSError, ValueError, KeyError, TypeError) as exc:
                message = str(exc) if isinstance(exc, (error, pd.LinkError)) else 'Asset metadata or local storage error.'
                row = {'id': ident, 'error': message}
                current_failures.append(row)
                failed[ident] = row
                print('Asset unavailable: '+ident+'. '+message, flush=True)
            update(assets_done=index+1)
            persist()
    finally:
        transport.close()
    return {'installed': list(lookup.values()), 'failures': current_failures, 'last_groups': groups}


async def wait_worker(func, *args):
    """Do not release the download lock while a cancelled thread still writes."""
    worker = asyncio.create_task(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # Repeated cancellation must not leave a still-writing worker behind.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()  # Retrieve a worker failure before propagating cancellation.
        raise


async def pull_ollama(session, name, timeout):
    from aiohttp import ClientTimeout
    success = False
    update(stage='downloading Ollama', current_asset='local prompt model', bytes_done=0, bytes_total=0)
    async with session.post('http://127.0.0.1:11434/api/pull',
            json={'model': name, 'stream': True},
            timeout=ClientTimeout(total=timeout, sock_read=180)) as response:
        if response.status != 200:
            raise ValueError('Local prompt model preparation was rejected.')
        async for line in response.content:
            if len(line) > 256*1024:
                raise ValueError('Oversized local model status response.')
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or event.get('error'):
                raise ValueError('Local prompt model preparation failed; check model access and disk space.')
            if event.get('status') == 'success':
                success = True
            completed, total = event.get('completed', 0), event.get('total', 0)
            if type(completed) is int and type(total) is int and 0 <= completed <= total:
                stage = 'verifying Ollama' if 'verif' in str(event.get('status', '')).lower() else 'downloading Ollama'
                update(stage=stage, bytes_done=completed, bytes_total=total)
    if not success:
        raise ValueError('Local prompt download ended without a success confirmation.')


def preparation_view(runtime):
    result = dict(runtime.job_state)
    progress = getattr(runtime, '_audit_progress', None)
    if progress is not None:
        result['progress'] = progress.snapshot()
    return result


async def run_preparation(runtime, groups, private, ns):
    from filelock import FileLock
    from aiohttp import ClientTimeout
    tracker = Progress()
    runtime._audit_progress = tracker
    runtime.job_state = {'state': 'preparing', 'groups': groups,
                        'started': time.time(), 'private_library': private}
    token = _CURRENT.set(tracker)
    lock = FileLock(str(ns['state_root']()/'downloads.lock'), timeout=0)
    heartbeat = None

    async def pulse():
        while True:
            ns['atomic_json'](ns['state_root']()/'preparation.json', preparation_view(runtime))
            p = tracker.snapshot()
            percent = '' if p['asset_percent'] is None else f" {p['asset_percent']:.1f}%"
            print(f"[PREP] {p['stage']}: {p['current_asset'] or 'workspace'}{percent}; "
                  f"{p['assets_done']}/{p['assets_total']} standard assets; "
                  f"{p['seconds_since_progress']:.0f}s since progress", flush=True)
            await asyncio.sleep(20)
    try:
        # Acquire synchronously without waiting, then keep it through worker shutdown.
        with lock:
            heartbeat = asyncio.create_task(pulse())
            report = await wait_worker(ns['prepare_standard'], [g for g in groups if g != 'ollama'])
            update(stage='preparing private library', current_asset='private library (redacted)', bytes_done=0, bytes_total=0)
            private_report = await wait_worker(ns['prepare_private']) if private else {}
            if 'ollama' in groups:
                update(stage='starting Ollama', current_asset='local prompt model', bytes_done=0, bytes_total=0)
                await runtime.start_ollama()
                expected = ns['read_catalog']('assets.json')['ollama']['models']
                details_by_name = {}
                for name in expected:
                    await pull_ollama(runtime.session, name, timeout=14400)
                    if ns['workspace']() == 'qwen':
                        from .qwen_prompt_local import verify_model
                        async with runtime.session.post('http://127.0.0.1:11434/api/show',
                                json={'model': name}, timeout=ClientTimeout(total=120)) as response:
                            if response.status != 200:
                                raise ValueError('Prompt-model metadata is unavailable.')
                            details = await response.json()
                        verify_model(details, name)
                        details_by_name[name] = details.get('details', {})
                async with runtime.session.get('http://127.0.0.1:11434/api/tags', timeout=ClientTimeout(total=120)) as response:
                    if response.status != 200:
                        raise ValueError('Local model list unavailable.')
                    tags = await response.json()
                by_name = {m['name'].casefold(): m for m in tags.get('models', [])}
                records = []
                for name in expected:
                    entry = by_name.get(name.casefold())
                    if entry is None:
                        raise ValueError('Prepared prompt model missing from local model list.')
                    records.append({'name': name, 'digest': entry.get('digest'), 'size': entry.get('size'),
                                    'details': details_by_name.get(name, entry.get('details', {}))})
                ns['atomic_json'](ns['state_root']()/'ollama-status.json', {'models': records})
            failures = len(report.get('failures', []))+len(private_report.get('failures', []))
            runtime.job_state.update(state='finished', failures=failures, finished=time.time())
            update(stage='finished', current_asset=None, bytes_done=0, bytes_total=0)
    except asyncio.CancelledError:
        runtime.job_state.update(state='cancelled', finished=time.time())
        raise
    except Exception as exc:
        runtime.job_state.update(state='failed', error_type=type(exc).__name__, finished=time.time(),
            message='Preparation failed. Review the asset status, selected groups, provider access and disk space. Secrets are not logged.')
        print('[PREP] Failed ('+type(exc).__name__+'). Inspect Workbench status.', flush=True)
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
            try:
                await heartbeat
            except (asyncio.CancelledError, OSError, ValueError):
                pass
        try:
            ns['atomic_json'](ns['state_root']()/'preparation.json', preparation_view(runtime))
        except (OSError, ValueError):
            print('[PREP] Unable to persist final status; inspect storage permissions and free space.', flush=True)
        finally:
            _CURRENT.reset(token)
        print('[PREP] '+runtime.job_state['state']+'; failures='+str(runtime.job_state.get('failures', 'see status')), flush=True)


async def request_preparation(runtime, request, ns):
    from aiohttp import web
    if runtime.job and not runtime.job.done():
        return web.json_response({'error': 'A preparation job is already running.'}, status=409)
    data = await request.json()
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object.')
    requested, extra, private = data.get('tasks', []), data.get('features', []), data.get('private', False)
    if (not isinstance(requested, list) or not isinstance(extra, list) or type(private) is not bool or
            any(not isinstance(x, str) for x in requested+extra)):
        raise ValueError('Invalid preparation selection.')
    if not requested and not extra and not private:
        raise ValueError('No asset selection was provided.')
    tasks = {t['id']: t for t in ns['read_catalog']('tasks.json')['tasks']}
    groups = set(extra)
    if groups-set(ns['model_config']()['toolbox']):
        raise ValueError('Unsupported feature.')
    for ident in requested:
        if ident not in tasks or ns['workspace']() not in tasks[ident]['workspaces']:
            raise ValueError('Unsupported task.')
        groups.update(tasks[ident]['requires'])
    # Parsing request.json() yields to the event loop. A second check prevents
    # two concurrent clicks from starting separate download jobs.
    if runtime.job and not runtime.job.done():
        return web.json_response({'error': 'A preparation job is already running.'}, status=409)
    runtime.job_state = {'state': 'preparing', 'groups': sorted(groups), 'private_library': private, 'started': time.time()}
    runtime.job = asyncio.create_task(runtime.prepare(sorted(groups), private))
    return web.json_response({'state': 'preparing', 'groups': sorted(groups)}, status=202)
