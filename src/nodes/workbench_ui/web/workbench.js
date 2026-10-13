import { app } from '../../scripts/app.js';
const css=`.wb-panel{font:13px/1.5 system-ui;padding:16px;color:var(--input-text,#eee);height:100%;overflow:auto}.wb-panel h2{font-size:18px;margin:0 0 8px}.wb-panel input,.wb-panel select{box-sizing:border-box;width:100%;margin:7px 0;padding:9px;background:var(--comfy-input-bg,#20252b);color:inherit;border:1px solid #455149;border-radius:6px}.wb-row{border:1px solid #414a45;border-radius:8px;padding:12px;margin:10px 0}.wb-row h3{font-size:13px;margin:0 0 6px}.wb-row p{font-size:11px;opacity:.8;margin:5px 0}.wb-panel button,.wb-panel a.wb-button{font:12px system-ui;background:#2d4437;color:#e0f5e8;border:1px solid #5a7e66;border-radius:5px;padding:8px;margin:5px 6px 2px 0;cursor:pointer;text-decoration:none;display:inline-block}.wb-status{font-size:11px;color:#c9c6a1;margin:7px 0;white-space:pre-wrap}.wb-muted{opacity:.72;font-size:11px}.wb-file{overflow-wrap:anywhere;margin:9px 0}.wb-floating{position:fixed;right:10px;top:80px;z-index:1000}.wb-fallback{position:fixed;right:15px;top:125px;bottom:20px;width:min(400px,90vw);z-index:1000;background:#181e1a;border:1px solid #4a6654;border-radius:12px;overflow:auto}`;
async function get(path,body){const r=await fetch('/_workbench/'+path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});if(!r.ok)throw Error('Workbench request failed ('+r.status+').');return r.json();}
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
async function render(container){
 container.replaceChildren();const p=el('div',undefined,'wb-panel');container.append(p);p.append(el('h2','Workbench HQ'));const note=el('div','Source candidate. Node presence is not proof of GPU or image-quality validation.','wb-muted');p.append(note);
 const msg=el('div','Loading catalog...','wb-status');p.append(msg);
 try{
  const c=await get('catalog');p.append(el('p','Workspace: '+c.workspace));const nav=el('div');p.append(nav);
  const search=el('input');search.placeholder='Search tasks, tools or IDs';p.append(search);const list=el('div');p.append(list);let mode='Tasks',status={tasks:[]};
  const footer=el('div');p.append(footer);const config=el('a','RunPod configurator','wb-button');config.href='/_workbench/';config.target='_blank';config.rel='noopener';footer.append(config);
  const refresh=el('button','Refresh model choices');refresh.onclick=async()=>{if(app.refreshComboInNodes)await app.refreshComboInNodes();else msg.textContent='Save your graph, then refresh ComfyUI to update loader choices.';};footer.append(refresh);
  async function prep(tasks=[],features=[],priv=false){try{msg.textContent='Preparing selected assets. You can keep using already-ready tasks.';await get('prepare',{tasks,features,private:priv});await poll();}catch(e){msg.textContent=e.message;}}
  async function show(){list.replaceChildren();
   if(mode==='Files'){
    for(const kind of ['input','output']){list.append(el('h3',kind));try{const files=await get('files?kind='+kind);for(const f of files.files){const a=el('a',f.path+' ('+(f.bytes/1e6).toFixed(1)+' MB)','wb-file');a.href='/_workbench/files?kind='+kind+'&path='+encodeURIComponent(f.path);a.download='';a.style.display='block';list.append(a);}}catch(e){list.append(el('p',e.message));}}return;
   }
   if(mode==='Toolbox'){
    const model=c.models.find(m=>m.id===c.workspace);const extra=el('div',undefined,'wb-row');extra.append(el('h3','Optional assets, not automatic graph steps'));
    for(const f of model.toolbox){const b=el('button','Prepare '+f);b.onclick=()=>prep([], [f]);extra.append(b);}
    const b=el('button','Refresh private library');b.onclick=()=>prep([],[],true);extra.append(b);extra.append(el('p','Private URLs are taken only from runtime variables or your private runtime file. Refreshing this library does not apply LoRAs.'));list.append(extra);
   }
   const q=search.value.toLowerCase();for(const t of c.tasks){const tool=t.id.startsWith('U')||t.id.startsWith('D')||t.id.startsWith('R');if(mode==='Toolbox'&&!tool)continue;if(mode==='Tasks'&&tool)continue;if(!(t.id+' '+t.title+' '+t.description).toLowerCase().includes(q))continue;
    const row=el('div',undefined,'wb-row');row.append(el('h3',t.id+' | '+t.title));row.append(el('p',t.description));const current=status.tasks.find(x=>x.id===t.id);const select=el('select');for(const v of t.variants){const o=el('option',v.label);o.value=v.file;select.append(o);}row.append(select);
    const state=el('div',undefined,'wb-status');row.append(state);function update(){const v=current?.variants.find(x=>x.file===select.value);state.textContent=!v?'Checking runtime...':v.state==='ready'?'Models and node types are present. GPU validation is separate.':(current.missing_assets.length?'Missing assets: '+current.missing_assets.join(', ')+'. ':'')+(v.missing_nodes.length?'Missing nodes: '+v.missing_nodes.join(', '):'');}select.onchange=update;update();
    const open=el('button','Open workflow');open.onclick=async()=>{try{const g=await get('workflow?file='+encodeURIComponent(select.value));await app.loadGraphData(g);msg.textContent='Opened '+t.id+'. Review inputs and references before running.';}catch(e){msg.textContent=e.message;}};
    const prepare=el('button','Prepare assets');prepare.onclick=()=>prep([t.id]);row.append(open,prepare);list.append(row);
   }
  }
  async function poll(){try{status=await get('status');const j=status.preparation;msg.textContent=(status.comfy_running?'ComfyUI running. ':'ComfyUI is starting. ')+(j.state==='preparing'?'Preparing: '+j.groups.join(', '):'Preparation: '+j.state+(j.failures?' ('+j.failures+' asset failures)':''));await show();}catch(e){msg.textContent=e.message;}}
  for(const name of ['Tasks','Toolbox','Files']){const b=el('button',name);b.onclick=()=>{mode=name;show();};nav.append(b);}search.oninput=show;await poll();
  const timer=setInterval(()=>{if(!p.isConnected){clearInterval(timer);return;}if(mode!=='Files')poll();},15000);
 }catch(e){msg.textContent=e.message+' Open ComfyUI through the authenticated Workbench port, not its internal port.';}
}
app.registerExtension({name:'Workbench.TaskLibrary',async setup(){
 const style=document.createElement('style');style.textContent=css;document.head.append(style);
 if(app.extensionManager?.registerSidebarTab){app.extensionManager.registerSidebarTab({id:'workbench-hq',icon:'pi pi-th-large',title:'Workbench',tooltip:'Task library, optional tools and readiness',type:'custom',render});}
 else{const b=el('button','Workbench','wb-floating');document.body.append(b);let panel;b.onclick=()=>{if(panel){panel.remove();panel=null;}else{panel=el('div',undefined,'wb-fallback');document.body.append(panel);render(panel);}};}
}});
