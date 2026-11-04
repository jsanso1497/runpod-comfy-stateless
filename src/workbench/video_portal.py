"""Authenticated saved files and diagnostics on the existing Workbench gateway."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from aiohttp import web, ClientTimeout
from .video_memory import snapshot
from .video_repair_catalog import VERSION

MEDIA = {'.mp4', '.mov', '.mkv', '.png', '.jpg', '.jpeg', '.webp', '.gif', '.wav', '.mp3', '.flac'}
LOGS = {'request.json', 'plan.json', 'status.json', 'worker_status.json', 'preflight.json',
        'worker_config_1.json', 'worker_config_2.json', 'worker_result_1.json', 'worker_result_2.json',
        'prepare.log', 'inference_1.log', 'inference_2.log', 'encode.log', 'supervisor.log'}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def list_media(root: Path) -> list[dict]:
    result = []
    if not root.exists():
        return result
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [x for x in dirs if not x.startswith('.') and not (Path(parent)/x).is_symlink()]
        for name in files:
            p = Path(parent) / name
            if p.suffix.lower() not in MEDIA or '.partial.' in p.name or p.is_symlink():
                continue
            try:
                st = p.stat()
                rel = p.relative_to(root).as_posix()
                result.append({'path': rel, 'bytes': st.st_size, 'mtime': st.st_mtime,
                               'download': '/_workbench/files?' + urlencode({'kind': 'output', 'path': rel})})
            except OSError:
                continue
    return sorted(result, key=lambda x: x['mtime'], reverse=True)


def list_jobs(root: Path) -> list[dict]:
    jobs = []
    if not root.exists():
        return jobs
    for p in root.iterdir():
        if not p.is_dir() or p.is_symlink() or not re.fullmatch(r'seedvr_[a-z0-9_]+_[a-f0-9]{16}', p.name):
            continue
        state = read_json(p/'status.json')
        worker = read_json(p/'worker_status.json')
        jobs.append({'id': p.name, 'mtime': p.stat().st_mtime, 'status': state,
                     'worker': worker, 'logs': sorted(x.name for x in p.iterdir() if x.name in LOGS and x.is_file())})
    return sorted(jobs, key=lambda x: x['mtime'], reverse=True)


def routes(app: web.Application, runtime) -> None:
    root = Path(os.environ.get('WB_DATA_ROOT', '/workspace'))
    async def page(request):
        return web.Response(text=PAGE, content_type='text/html', headers={'Cache-Control': 'no-store'})
    async def data(request):
        section = request.query.get('section', 'files')
        if section == 'files':
            rows = await asyncio.to_thread(list_media, root/'output')
            query = request.query.get('q', '').lower()
            if query:
                rows = [x for x in rows if query in x['path'].lower()]
            offset = max(0, int(request.query.get('offset', '0')))
            return web.json_response({'files': rows[offset:offset+200], 'total': len(rows), 'offset': offset, 'version': VERSION})
        if section == 'jobs':
            return web.json_response({'jobs': await asyncio.to_thread(list_jobs, root/'workbench/video_jobs')})
        if section == 'memory':
            result = await asyncio.to_thread(snapshot)
            result['last_job'] = read_json(root/'workbench/last-job-memory.json')
            try:
                async with runtime.session.get('http://127.0.0.1:8189/queue', timeout=ClientTimeout(total=5)) as response:
                    queue = await response.json()
                result['queue'] = {key: len(queue.get(key, [])) for key in ('queue_running', 'queue_pending')}
                async with runtime.session.get('http://127.0.0.1:8189/history?max_items=5', timeout=ClientTimeout(total=5)) as response:
                    history = await response.json()
                result['recent_errors'] = []
                for job_id, job in history.items():
                    for event, detail in job.get('status', {}).get('messages', []):
                        if event in ('execution_error', 'execution_interrupted'):
                            result['recent_errors'].append({'job_id': job_id, 'event': event, **{
                                k: detail.get(k) for k in ('timestamp', 'node_id', 'node_type', 'exception_type', 'exception_message', 'traceback')}})
            except Exception as exc:
                result['comfy_status'] = type(exc).__name__
            return web.json_response(result)
        raise ValueError('Unknown section')
    async def log(request):
        job = request.query.get('job', '')
        name = request.query.get('file', '')
        if job == 'comfy' and name in ('comfy.log', 'comfy.log.1', 'comfy.log.2', 'comfy.log.3'):
            target = root/'workbench/logs'/name
        else:
            if not re.fullmatch(r'seedvr_[a-z0-9_]+_[a-f0-9]{16}', job) or name not in LOGS:
                raise ValueError('Invalid log')
            target = root/'workbench/video_jobs'/job/name
        if (not target.resolve().is_relative_to((root/'workbench').resolve())
                or not target.is_file() or target.is_symlink()):
            raise FileNotFoundError()
        return web.FileResponse(target, headers={'Content-Disposition': 'attachment', 'Cache-Control': 'no-store'})
    async def free(request):
        async with runtime.session.get('http://127.0.0.1:8189/queue', timeout=ClientTimeout(total=5)) as response:
            queue = await response.json()
        if queue.get('queue_running') or queue.get('queue_pending'):
            return web.json_response({'error': 'Cancel or finish queued jobs first. Nothing was interrupted.'}, status=409)
        if not {'queue_running', 'queue_pending'}.issubset(queue):
            raise ValueError('Cannot verify idle queue')
        async with runtime.session.post('http://127.0.0.1:8189/free', json={'unload_models': True, 'free_memory': True},
                                        timeout=ClientTimeout(total=10)) as response:
            response.raise_for_status()
        return web.json_response({'state': 'cleanup_requested', 'note': 'Refresh memory to verify; external processes are not killed.'})
    app.router.add_get('/_workbench/repair', page)
    app.router.add_get('/_workbench/repair-data', data)
    app.router.add_get('/_workbench/repair-log', log)
    app.router.add_post('/_workbench/repair-free', free)

PAGE = r'''<!doctype html><meta charset="utf-8"><title>Workbench files and video jobs</title>
<style>body{font:15px system-ui;background:#12171e;color:#e6edf3;margin:24px}a{color:#89caff}button,input{font:inherit;padding:9px;margin:4px;background:#202c39;color:inherit;border:1px solid #526579;border-radius:5px}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #34404e;padding:9px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1c2530;padding:16px}small{color:#b5c1cf}</style>
<h2>Saved files, video jobs, and memory</h2><small>Reads files on disk. Clearing memory does not delete media. No separate file-browser port.</small>
<nav><button data-tab="files">Saved files</button><button data-tab="jobs">Video jobs / logs</button><button data-tab="memory">Memory / errors</button><button id="refresh">Refresh</button><a href="/" target="_top">ComfyUI</a></nav>
<div id="search"><input id="query" placeholder="Filter output paths"><button id="find">Search</button><button id="prev">Previous</button><button id="next">Next 200</button></div>
<div id="actions" hidden><button id="free">Unload idle models and caches</button><a href="/_workbench/repair-log?job=comfy&file=comfy.log">Download Comfy log</a></div><div id="body"></div>
<script>
let tab='files', offset=0, total=0; const body=document.getElementById('body');
const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n};
async function load(){document.getElementById('search').hidden=tab!=='files';document.getElementById('actions').hidden=tab!=='memory';body.replaceChildren(el('p','Loading...'));try{const p=new URLSearchParams({section:tab,offset,q:document.getElementById('query').value});const r=await fetch('/_workbench/repair-data?'+p);if(!r.ok)throw Error('HTTP '+r.status);const d=await r.json();body.replaceChildren();if(tab==='files'){total=d.total;body.append(el('p',`${total} saved media files. Showing ${offset+1} to ${Math.min(offset+200,total)}.`));const table=el('table');for(const f of d.files){const row=el('tr'),cell=el('td'),a=el('a',f.path);a.href=f.download;a.setAttribute('download','');cell.append(a);row.append(cell,el('td',(f.bytes/1048576).toFixed(1)+' MiB'),el('td',new Date(f.mtime*1000).toLocaleString()));table.append(row)}body.append(table)}else if(tab==='jobs'){for(const j of d.jobs){body.append(el('h3',j.id));body.append(el('pre',JSON.stringify({status:j.status,worker:j.worker},null,2)));const links=el('p');for(const file of j.logs){const a=el('a',file);a.href='/_workbench/repair-log?'+new URLSearchParams({job:j.id,file});a.style.marginRight='16px';links.append(a)}body.append(links)}}else body.append(el('pre',JSON.stringify(d,null,2)))}catch(e){body.replaceChildren(el('pre',String(e)))}}
for(const b of document.querySelectorAll('[data-tab]'))b.onclick=()=>{tab=b.dataset.tab;offset=0;load()};document.getElementById('refresh').onclick=load;document.getElementById('find').onclick=()=>{offset=0;load()};document.getElementById('prev').onclick=()=>{offset=Math.max(0,offset-200);load()};document.getElementById('next').onclick=()=>{if(offset+200<total){offset+=200;load()}};document.getElementById('free').onclick=async()=>{const r=await fetch('/_workbench/repair-free',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const d=await r.json();alert(d.note||d.error||d.state);load()};if(new URLSearchParams(location.search).get('job'))tab='jobs';load();
</script>'''
