"""Gateway review page for the independent batch controller."""
import asyncio
import json
from aiohttp import web
from .config import data_root
from . import economy_batch as batch


def install_routes(app):
    async def home(request):return web.Response(text=HTML,content_type='text/html')
    async def state(request):
        name=request.query.get('run','')
        if not name:return web.json_response({'runs':await asyncio.to_thread(runs)})
        try:return web.json_response(await asyncio.to_thread(batch.summary,batch.folder(name)))
        except (ValueError,OSError) as exc:raise web.HTTPBadRequest(text=str(exc))
    async def act(request):
        if request.headers.get('X-H3-Economy')!='1':raise web.HTTPForbidden()
        data=await request.json()
        try:return web.json_response(await asyncio.to_thread(batch.action,data.get('run',''),data.get('action','')))
        except (ValueError,OSError) as exc:raise web.HTTPBadRequest(text=str(exc))
    app.router.add_get('/_swap/economy',home)
    app.router.add_get('/_swap/economy/state',state)
    app.router.add_post('/_swap/economy/action',act)


def runs():
    root=data_root()/'output/h3_character_swap/economy'
    return sorted([p.name for p in root.iterdir() if p.is_dir() and (p/'config.json').is_file()],reverse=True) if root.exists() else []


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>H3 Economy Batches</title>
<style>body{background:#131922;color:#e6edf6;font:16px system-ui;max-width:1100px;margin:28px auto;padding:0 20px}a{color:#a0ceff}button,select{padding:10px;font:inherit;margin:6px}video{width:100%;max-height:65vh;background:#000}pre{white-space:pre-wrap;background:#1e2938;padding:16px}#error{color:#ffcd99}</style>
<h1>H3 Economy Batches</h1><p><a href="/">ComfyUI</a> | <a href="/files/">Files</a> | <a href="/_swap/status">Startup</a></p><p>Generate the first clip, approve it, then let the remaining independent clips render and join. No inherited generated appearance. Joins are not guaranteed invisible.</p>
<label>Run <select id="runs"></select></label><button id="refresh">Refresh</button><p id="error" role="alert"></p>
<video id="preview" controls playsinline></video><p id="links"></p><div id="buttons"></div><pre id="state">Select a batch started with workflow 15.</pre>
<script>const q=x=>document.querySelector(x);let run=new URLSearchParams(location.search).get('run')||'',last='';
async function refresh(){try{if(!run)return;const r=await fetch('/_swap/economy/state?run='+encodeURIComponent(run));if(!r.ok)throw Error(await r.text());const d=await r.json();q('#state').textContent=JSON.stringify(d,null,2);const u=d.final_url||d.preview||'';if(u!==last){q('#preview').src=u;last=u}q('#links').replaceChildren();if(d.final_url){let a=document.createElement('a');a.href=d.final_url;a.textContent='Open final video';q('#links').append(a)}q('#buttons').replaceChildren();for(const action of ['Approve and continue','Retry candidate','Resume','Stop']){const b=document.createElement('button');b.textContent=action;b.disabled=(action!=='Stop'&&d.running)||(action.startsWith('Approve')||action==='Retry candidate')&&d.status!=='needs_review';b.onclick=()=>act(action);q('#buttons').append(b)}}catch(e){q('#error').textContent=String(e)}}
async function act(action){q('#error').textContent='';try{const r=await fetch('/_swap/economy/action',{method:'POST',headers:{'Content-Type':'application/json','X-H3-Economy':'1'},body:JSON.stringify({run,action})});if(!r.ok)throw Error(await r.text());refresh()}catch(e){q('#error').textContent=String(e)}}
async function init(){const r=await fetch('/_swap/economy/state');const d=await r.json();for(const n of d.runs){const o=document.createElement('option');o.value=n;o.textContent=n;q('#runs').append(o)}if(run)q('#runs').value=run;else run=q('#runs').value;q('#runs').onchange=e=>{run=e.target.value;refresh()};q('#refresh').onclick=refresh;refresh();setInterval(refresh,4000)}init().catch(e=>q('#error').textContent=String(e));</script></html>'''
