"""Authenticated review UI mounted inside the EXISTING 8188 gateway.
No new public ports, servers or authentication bypasses.
"""
from __future__ import annotations
import asyncio
from aiohttp import web
from .config import data_root
from .disk_common import ChainError
from .disk_control import summaries, act

HTML='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3 Low-Memory Chain</title><style>
body{font:16px system-ui;background:#151922;color:#e9edf2;max-width:1100px;margin:28px auto;padding:0 18px}
a{color:#98cdfb}h1{font-size:26px}h2{font-size:20px}button,select{font:inherit;margin:5px;padding:10px;border-radius:5px}
section{padding:18px;margin:18px 0;border:1px solid #465268;border-radius:10px}video{width:min(100%,650px);max-height:450px;display:block;margin:15px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px ui-monospace}.warn{color:#ffce85}img{max-height:280px;max-width:48%;object-fit:contain}
</style></head><body><h1>H3 Low-Memory Chain</h1>
<p><a href="/">ComfyUI</a> | <a href="/_swap/status">Model preparation</a></p>
<p>Start a new shot in workflow <strong>06_H3_LowMemory_Chain_Control</strong> or <strong>09_H3_LowMemory_Face_Body_Chain</strong>. Keep other generation jobs out of the queue while a chain runs.</p>
<p class="warn">A green check from the heuristic is not a seamless-motion or identity guarantee. Watch every join at normal speed. Approving a candidate explicitly accepts its visible result.</p>
<p id="message"></p><div id="runs">Loading...</div>
<script>
const node=document.getElementById('runs'),msg=document.getElementById('message');
let last='';
async function action(run,action){
 if(action==='Approve candidate and continue'&&!confirm('Have you watched this candidate and its join? Approve this exact saved result?'))return;
 try{const r=await fetch('/_swap/chain/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run,action})});
 const d=await r.json();if(!r.ok)throw Error(d.error||r.status);msg.textContent='Action requested: '+action;last='';await refresh();}
 catch(e){msg.textContent=e.message;}
}
function el(tag,text,parent){const n=document.createElement(tag);if(text)n.textContent=text;if(parent)parent.append(n);return n;}
async function refresh(){try{
 const r=await fetch('/_swap/chain/runs',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const rows=await r.json();
 const selected=new URLSearchParams(location.search).get('run');
 const filtered=selected?rows.filter(x=>x.run===selected):rows;
 const snapshot=JSON.stringify(filtered);if(snapshot===last)return;last=snapshot;node.replaceChildren();
 if(!filtered.length){el('p','No matching runs. Start one in ComfyUI, then refresh this page.',node);return;}
 for(const row of filtered){const s=el('section',null,node);el('h2',row.run+' | '+row.status,s);
 el('p',`${row.accepted_chunks}/${row.total_chunks||'?'} accepted chunks; ${row.accepted_seconds.toFixed(3)} seconds. Controller: ${row.controller_running?'running':'stopped'}`,s);
 if(row.reference_roles)el('pre',JSON.stringify({references:row.reference_files,roles:row.reference_roles},null,2),s);
 if(row.error)el('pre',row.error,s);
 if(row.candidate_quality)el('pre',JSON.stringify(row.candidate_quality,null,2),s);
 for(const [name,url] of Object.entries(row.links||{})){const a=el('a',name,s);a.href=url;a.target='_blank';el('br',null,s);
  if(name==='join_review.mp4'||(name==='preview.mp4'&&!row.links['join_review.mp4'])){const v=el('video',null,s);v.controls=true;v.preload='metadata';v.src=url;}
  if(name==='source_mask.png'){const i=el('img',null,s);i.src=url;i.alt='Raw mask at the first new source frame';}}
 const options=row.controller_running?['Stop']:(row.status==='needs_review'?['Approve candidate and continue','Retry candidate','Resume']:(row.status==='complete'?[]:['Resume']));
 for(const a of options){const b=el('button',a,s);b.onclick=()=>action(row.run,a);}
 el('pre',row.output_folder,s);}
 }catch(e){msg.textContent='Status error: '+e.message;}}
refresh();setInterval(refresh,4000);
</script></body></html>'''


def install_routes(app):
    async def page(request): return web.Response(text=HTML,content_type='text/html')
    async def runs(request):
        return web.json_response(await asyncio.to_thread(summaries,data_root()/'output'))
    async def control(request):
        # Existing gateway middleware requires Basic auth and rejects cross-site
        # writes before this handler. An explicit JSON content type also blocks
        # simple cross-origin form submissions.
        if request.content_type!='application/json':
            return web.json_response({'error':'JSON required'},status=415)
        if request.content_length and request.content_length>4096:
            return web.json_response({'error':'Request too large'},status=413)
        try:
            data=await request.json()
            if not isinstance(data,dict) or set(data)!={'run','action'}:
                raise ValueError('Expected run and action only')
            if data['action'] not in ('Status','Resume','Approve candidate and continue','Retry candidate','Stop'):
                raise ValueError('Unsupported review action')
            return web.json_response(await asyncio.to_thread(act,data_root()/'output',data['run'],data['action']))
        except (ChainError,OSError,ValueError,KeyError) as exc:
            return web.json_response({'error':str(exc)},status=400)
    app.router.add_get('/_swap/chain',page)
    app.router.add_get('/_swap/chain/runs',runs)
    app.router.add_post('/_swap/chain/action',control)
