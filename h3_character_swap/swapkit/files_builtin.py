"""Authenticated file access inside the gateway. No daemon, database or model imports.

Only the gateway exposes these routes. No shell execution, arbitrary URLs, deletes,
symlinks or dotfiles. Uploads stream to a temporary file and never overwrite a file.
"""
from __future__ import annotations
import asyncio
import os
from pathlib import Path, PurePosixPath
import tempfile
from aiohttp import web
from .config import data_root, boolean


def backend():
    value = os.environ.get('H3_SWAP_FILES_BACKEND', 'classic')
    if value not in ('builtin', 'classic'):
        raise ValueError('H3_SWAP_FILES_BACKEND must be builtin or classic')
    return value


def local(root, name='', *, must_exist=True):
    root = Path(root).resolve()
    if not isinstance(name, str) or '\\' in name or '\x00' in name or name.startswith('/'):
        raise web.HTTPBadRequest(text='Use a workspace-relative path.')
    rel = PurePosixPath(name)
    if any(x in ('.', '..') or x.startswith('.') for x in rel.parts):
        raise web.HTTPForbidden(text='Hidden and parent paths are not exposed.')
    p = root
    for part in rel.parts:
        p = p / part
        if p.is_symlink():
            raise web.HTTPForbidden(text='Symlinks are not exposed.')
    if not p.resolve().is_relative_to(root):
        raise web.HTTPForbidden()
    if must_exist and not p.exists():
        raise web.HTTPNotFound(text='File or folder no longer exists.')
    return p


def list_dir(root, name):
    p = local(root, name)
    if not p.is_dir(): raise web.HTTPBadRequest(text='Not a folder.')
    entries=[]
    for child in p.iterdir():
        if child.name.startswith('.') or child.is_symlink(): continue
        try:
            st=child.stat()
            if not child.is_dir() and not child.is_file(): continue
            entries.append({'name':child.name,'path':child.relative_to(root).as_posix(),
                'directory':child.is_dir(),'bytes':st.st_size,'modified':st.st_mtime})
        except OSError: continue
        if len(entries)>4000:
            raise web.HTTPBadRequest(text='Too many files in this folder. Organize into subfolders.')
    entries.sort(key=lambda x:(not x['directory'], x['name'].lower()))
    return {'path':p.relative_to(root).as_posix() if p!=root else '', 'entries':entries}


def install(app, status):
    root=data_root().resolve();root.mkdir(parents=True,exist_ok=True)
    active=boolean('H3_SWAP_FILE_BROWSER',True)
    status['filebrowser']={'enabled':active,'ready':active,'stage':'ready' if active else 'disabled',
        'backend':'builtin','path':'/files/','note':'Runs inside gateway; independent of model preparation.'}
    def check():
        if not active: raise web.HTTPNotFound(text='File access is disabled.')
    async def home(request):
        check()
        return web.Response(text=HTML,content_type='text/html',headers={
            'Content-Security-Policy':"default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"})
    async def redirect(request):
        check();raise web.HTTPTemporaryRedirect(location='/files/')
    async def listing(request):
        check();result=await asyncio.to_thread(list_dir,root,request.query.get('path',''))
        return web.json_response(result)
    async def download(request):
        check();p=local(root,request.query.get('path',''))
        if not p.is_file(): raise web.HTTPBadRequest(text='Choose a file.')
        from urllib.parse import quote
        return web.FileResponse(p,headers={'Content-Disposition':"attachment; filename*=UTF-8''"+quote(p.name,safe=''),
                                          'Content-Type':'application/octet-stream'})
    async def upload(request):
        check()
        if request.headers.get('X-H3-Files')!='1': raise web.HTTPForbidden(text='Missing upload origin guard.')
        directory=local(root,request.query.get('path',''))
        name=request.query.get('name','')
        if not name or PurePosixPath(name).name!=name or name in ('.','..'):
            raise web.HTTPBadRequest(text='Choose a simple filename.')
        dest=local(root,(directory.relative_to(root)/name).as_posix(),must_exist=False)
        if not directory.is_dir(): raise web.HTTPBadRequest(text='Not a folder.')
        if dest.exists(): raise web.HTTPConflict(text='File already exists. Rename the upload first.')
        limit=request.app['upload_limit'];size=0
        if request.content_length and request.content_length>limit:
            raise web.HTTPRequestEntityTooLarge(max_size=limit,actual_size=request.content_length)
        fd,tmp=tempfile.mkstemp(dir=directory,prefix='.upload-')
        try:
            with os.fdopen(fd,'wb') as out:
                async for chunk in request.content.iter_chunked(1024*1024):
                    size+=len(chunk)
                    if size>limit: raise web.HTTPRequestEntityTooLarge(max_size=limit,actual_size=size)
                    await asyncio.to_thread(out.write,chunk)
                out.flush()
            # A concurrent upload may win the same name. Hard link is atomic and
            # refuses overwrite; then remove the private staging filename.
            try: os.link(tmp,dest)
            except FileExistsError: raise web.HTTPConflict(text='File appeared during upload. Nothing was overwritten.')
            return web.json_response({'saved':dest.relative_to(root).as_posix(),'bytes':size})
        finally:
            Path(tmp).unlink(missing_ok=True)
    async def mkdir(request):
        check()
        if request.headers.get('X-H3-Files')!='1': raise web.HTTPForbidden()
        p=local(root,request.query.get('path',''),must_exist=False)
        try:p.mkdir()
        except FileExistsError:raise web.HTTPConflict(text='Folder already exists.')
        except FileNotFoundError:raise web.HTTPBadRequest(text='Parent folder does not exist.')
        return web.json_response({'created':p.relative_to(root).as_posix()})
    app.router.add_get('/files',redirect)
    app.router.add_get('/files/',home)
    app.router.add_get('/files/api/list',listing)
    app.router.add_get('/files/api/download',download)
    app.router.add_post('/files/api/upload',upload)
    app.router.add_post('/files/api/mkdir',mkdir)
    print('[H3 FILES READY] Built-in browser: /files/ on HTTP 8188. No separate FileBrowser port.',flush=True)


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3 Files</title><style>body{font:16px system-ui;background:#12171e;color:#e7ebef;max-width:1100px;margin:32px auto;padding:0 20px}a{color:#9acbff}button,input{font:inherit;padding:9px;margin:4px;border-radius:5px}table{width:100%;border-collapse:collapse}th,td{padding:12px;text-align:left;border-bottom:1px solid #39424f}#state{white-space:pre-wrap;padding:12px;background:#1d2632}small{color:#bbc4cf}nav{display:flex;gap:20px;flex-wrap:wrap}</style>
<h1>H3 Files</h1><nav><a href="/">ComfyUI</a><a href="/_swap/status">Startup status</a><a href="/_swap/economy">Automatic batches</a><a href="/_swap/recovery">Workflow records</a></nav>
<p>File access is available while models load. Root: <strong>/workspace</strong>.</p>
<p><button id="up">Up one folder</button><button id="refresh">Refresh</button><button id="newdir">New folder</button><label>Upload <input type="file" id="upload" multiple></label></p>
<h2 id="path"></h2><div id="state" role="status">Opening workspace...</div><table><thead><tr><th>Name</th><th>Size</th><th>Modified</th><th></th></tr></thead><tbody id="entries"></tbody></table>
<script>
let current='';const q=x=>document.querySelector(x),note=x=>q('#state').textContent=x;
const enc=encodeURIComponent, url=p=>'/files/api/download?path='+enc(p);
async function openDir(p){try{const r=await fetch('/files/api/list?path='+enc(p),{cache:'no-store'});if(!r.ok)throw Error(await r.text());const d=await r.json();current=d.path;q('#path').textContent='/workspace'+(current?'/'+current:'');q('#entries').replaceChildren();for(const f of d.entries){const tr=document.createElement('tr');const td=document.createElement('td');const a=document.createElement('a');a.textContent=(f.directory?'[Folder] ':'')+f.name;a.href=f.directory?'#':url(f.path);if(f.directory)a.onclick=e=>{e.preventDefault();openDir(f.path)};else a.download=f.name;td.append(a);tr.append(td);for(const v of [f.directory?'':(f.bytes/1048576).toFixed(2)+' MB',new Date(f.modified*1000).toLocaleString()]){const t=document.createElement('td');t.textContent=v;tr.append(t)}const t=document.createElement('td');if(!f.directory){const b=document.createElement('button');b.textContent='Copy path';b.onclick=()=>navigator.clipboard.writeText(f.path);t.append(b)}tr.append(t);q('#entries').append(tr)}note('Ready. Downloads and uploads do not wait for H3.')}catch(e){note(String(e))}}
q('#up').onclick=()=>openDir(current.split('/').slice(0,-1).join('/'));q('#refresh').onclick=()=>openDir(current);
q('#newdir').onclick=async()=>{const n=prompt('New folder name');if(!n)return;const p=(current?current+'/':'')+n;const r=await fetch('/files/api/mkdir?path='+enc(p),{method:'POST',headers:{'X-H3-Files':'1'}});if(!r.ok)return note(await r.text());openDir(current)};
function upload(f,p){return new Promise((resolve,reject)=>{const x=new XMLHttpRequest();x.open('POST','/files/api/upload?path='+enc(p)+'&name='+enc(f.name));x.setRequestHeader('X-H3-Files','1');x.upload.onprogress=e=>note('Uploading '+f.name+': '+(e.lengthComputable?Math.round(e.loaded/e.total*100)+'%':e.loaded+' bytes'));x.onerror=()=>reject(Error('Upload connection failed.'));x.onload=()=>x.status<300?resolve():reject(Error(x.responseText));x.send(f)})}
q('#upload').onchange=async e=>{const p=current;try{for(const f of e.target.files)await upload(f,p);await openDir(p)}catch(x){note(String(x))}e.target.value=''};openDir('');
</script></html>'''
