"""Small authenticated browser for recovery JSONs on the existing gateway."""
from __future__ import annotations
import html
import urllib.parse
from pathlib import Path
from aiohttp import web
from .config import data_root, safe_path


def download_url(path, output_root):
    rel = Path(path).resolve().relative_to(Path(output_root).resolve()).as_posix()
    return '/_swap/recovery/file?' + urllib.parse.urlencode({'path': rel})


def allowed_file(output_root, relative):
    if not isinstance(relative, str):
        raise ValueError('Missing recovery path')
    path = safe_path(Path(output_root), relative)
    rel = path.relative_to(Path(output_root).resolve())
    allowed_roots = {'workflow_history', 'disk_chain_v2', 'native_preview'}
    if (len(rel.parts) < 3 or rel.parts[0] != 'h3_character_swap' or rel.parts[1] not in allowed_roots
            or not path.name.endswith(('.workflow.json', '.api.json'))
            or not path.is_file() or path.stat().st_size > 16*1024**2):
        raise ValueError('Only saved workflow/API JSONs can be downloaded here')
    return path


def install_routes(app):
    async def download(request):
        try:
            path = allowed_file(data_root()/'output', request.query.get('path'))
        except (ValueError, OSError):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={'Content-Type': 'application/json',
            'Content-Disposition': "attachment; filename*=UTF-8''"+urllib.parse.quote(path.name, safe='')})

    async def page(request):
        output = data_root()/'output'
        journal = output/'h3_character_swap/workflow_history'
        items = []
        if journal.is_dir():
            dirs = sorted([p for p in journal.iterdir() if p.is_dir() and not p.is_symlink()],
                          key=lambda p: p.name, reverse=True)[:100]
            for directory in dirs:
                for name in ('submitted.workflow.json', 'submitted.api.json'):
                    path = directory/name
                    if path.is_file() and not path.is_symlink():
                        url = html.escape(download_url(path, output), quote=True)
                        label = html.escape(directory.name+' / '+name)
                        items.append('<li><a href="'+url+'">'+label+'</a></li>')
        body = ('<!doctype html><html><meta charset="utf-8"><title>H3 Saved Workflows</title>'
                '<style>body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 20px}'
                'li{margin:10px 0;overflow-wrap:anywhere}</style><h1>Saved H3 workflows</h1>'
                '<p><a href="/">ComfyUI</a> | <a href="/_swap/chain">Chain runs and recovery</a> | '
                '<a href="/files/">FileBrowser</a></p>'
                '<p>Download an editable <strong>.workflow.json</strong> and drag it into ComfyUI. '
                'These are submitted settings, including canceled or failed attempts, not proof of a completed render.</p>'
                '<p>For a chain, use its review page: <strong>Export workflow files</strong> works for older saved runs. '
                '<strong>repeat</strong> starts a new run; <strong>resume</strong> continues its saved state. '
                'Inputs and model weights must still be available. Old videos without saved settings cannot be reconstructed.</p>'
                '<p>JSONs and embedded metadata include prompts and input filenames. Treat them as private when sharing media.</p>'
                '<h2>Latest 100 submissions</h2><ul>'+(''.join(items) or '<li>No saved submissions yet.</li>')+'</ul></html>')
        return web.Response(text=body, content_type='text/html')
    app.router.add_get('/_swap/recovery', page)
    app.router.add_get('/_swap/recovery/file', download)
