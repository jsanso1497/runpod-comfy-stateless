(function(){
'use strict';
const C=window.WORKBENCH_CATALOG,L=window.WorkbenchLogic,$=s=>document.querySelector(s);
const defaults=()=>Object.fromEntries(C.models.map(m=>[m.id,new Set(m.default_tasks)]));
const state={model:'qwen',selected:defaults(),features:Object.fromEntries(C.models.map(m=>[m.id,new Set()])),filter:'All',query:''};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function saveText(name,text,type='text/plain'){const a=document.createElement('a');const url=URL.createObjectURL(new Blob([text],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function copy(text){try{await navigator.clipboard.writeText(text);$('#copy-status').textContent='Copied.';}catch{const t=document.createElement('textarea');t.value=text;document.body.append(t);t.select();document.execCommand('copy');t.remove();$('#copy-status').textContent='Copied. Use the text panel if your browser blocked copying.';}}
function options(){return {repository:$('#repository').value,tag:$('#image-tag').value,storage:$('#storage').value,loras:$('#private-loras').checked,checkpoints:$('#private-checkpoints').checked};}
function plan(){return L.plan(C,state.model,[...state.selected[state.model]],[...state.features[state.model]],options());}
function cards(){
 $('#models').innerHTML=C.models.map(m=>`<button class="model-card ${state.model===m.id?'active':''}" data-model="${m.id}" aria-pressed="${state.model===m.id}"><span class="model-initial ${m.id}">${esc(m.short.slice(0,1))}</span><span class="model-card-copy"><strong>${esc(m.name)}</strong><span>${esc(m.kind)}</span><small>${esc(m.precision)}</small></span><span class="radio-dot"></span></button>`).join('');
 $('#models').querySelectorAll('button').forEach(b=>b.onclick=()=>{state.model=b.dataset.model;state.filter='All';render();});
}
function taskList(){
 const model=C.models.find(m=>m.id===state.model);$('#task-heading').textContent=model.short+' tasks';
 const all=C.tasks.filter(t=>t.workspaces.includes(state.model)).sort((a,b)=>(b.category==='Standard HQ')-(a.category==='Standard HQ')||a.id.localeCompare(b.id,undefined,{numeric:true}));
 const tabs=['All','Selected','Standard HQ','Tasks','Toolbox','Diagnostics'];
 $('#filters').innerHTML=tabs.map(t=>`<button data-filter="${t}" class="filter ${state.filter===t?'active':''}">${t}</button>`).join('');
 $('#filters').querySelectorAll('button').forEach(b=>b.onclick=()=>{state.filter=b.dataset.filter;taskList();});
 const rows=all.filter(t=>{const selected=state.selected[state.model].has(t.id);const utility=t.family==='toolbox'||t.category==='Toolbox'||t.id.startsWith('U');const diagnostic=t.id.startsWith('D');const standard=t.category==='Standard HQ';return (state.filter==='All'||state.filter==='Selected'&&selected||state.filter==='Standard HQ'&&standard||state.filter==='Tasks'&&!utility&&!diagnostic&&!standard||state.filter==='Toolbox'&&utility||state.filter==='Diagnostics'&&diagnostic)&&(t.id+' '+t.title+' '+t.description).toLowerCase().includes(state.query);});
 $('#tasks').innerHTML=rows.map(t=>`<label class="task-row"><input type="checkbox" data-task="${t.id}" ${state.selected[state.model].has(t.id)?'checked':''}><span class="task-body"><span class="task-title"><code>${t.id}</code><strong>${esc(t.title)}</strong>${/experimental/i.test(t.status)?'<span class="pill experimental">Experimental</span>':''}</span><span class="description">${esc(t.description)}</span><span class="task-foot">${t.variants.length} workflow${t.variants.length===1?'':'s'}${t.requires.includes('ollama')?' · Requires local F16 prompt helpers':''}${t.requires.includes('sam')?' · Requires SAM':''}${t.requires.includes('seedvr2')&&state.model!=='restoration'?' · Requires SeedVR2':''}</span></span></label>`).join('')||'<p class="empty">No matching tasks.</p>';
 $('#tasks').querySelectorAll('[data-task]').forEach(i=>i.onchange=()=>{if(i.checked)state.selected[state.model].add(i.dataset.task);else state.selected[state.model].delete(i.dataset.task);summary();});
 $('#selection-count').textContent=state.selected[state.model].size+' selected / '+all.length+' available';
 const labels={sam:['SAM 3.1','Text-based selection. Available separately; never automatically inserted.'],seedvr2:['SeedVR2 7B FP16','Restoration weights available when needed. No automatic finishing.'],ollama:['Local F16 prompt helpers','Two 32B helpers, loaded sequentially. About 134 GB additional download.']};
 $('#features').innerHTML=model.toolbox.map(f=>`<label class="feature"><input type="checkbox" data-feature="${f}" ${state.features[state.model].has(f)?'checked':''}><span><strong>${labels[f][0]}</strong><small>${labels[f][1]}</small></span></label>`).join('')||'<p class="subtle">Manual masks, compositing, resizing and compatible model loaders are included. This workspace already contains SeedVR2.</p>';
 $('#features').querySelectorAll('[data-feature]').forEach(i=>i.onchange=()=>{if(i.checked)state.features[state.model].add(i.dataset.feature);else state.features[state.model].delete(i.dataset.feature);summary();});
}
function summary(){
 try{const p=plan();$('#config-error').textContent='';$('#result').value=L.text(p);$('#image').textContent=p.image;$('#summary-name').textContent=p.name;
 $('#stats').innerHTML=`<div><strong>${p.selected_tasks.length}</strong><span>Selected tasks</span></div><div><strong>~${p.estimated_model_download_gb}${p.unestimated_assets.length?'+':''} GB</strong><span>Model downloads</span></div><div><strong>${p.container_disk_gb+p.volume_disk_gb} GB</strong><span>Planned storage</span></div>`;
 $('#planning').textContent=p.planning_gpu_vram_gb+' GB VRAM / '+p.planning_system_ram_gb+' GB system RAM: conservative planning targets, not tested minimums. '+p.hardware_note;
 $('#secret-rows').innerHTML=[['hf_token','Existing'],['civit_token','Existing'],['comfy_password','Create once, 16+ characters'],...(options().loras?[['comfy_loras','Create once, private link list']]:[]),...(options().checkpoints?[['comfy_checkpoints','Create once, private JSON manifest']]:[])].map(([name,detail])=>`<div><code>${name}</code><span>${detail}</span></div>`).join('');
 const required=p.asset_groups.filter(x=>!state.features[state.model].has(x)&&!C.models.find(m=>m.id===state.model).default_groups.includes(x));$('#requirements').textContent=required.length?'Selected tasks automatically require: '+required.join(', ')+'.':'No additional model group is required by your selection.';
 $('#selection-count').textContent=state.selected[state.model].size+' selected / '+C.tasks.filter(t=>t.workspaces.includes(state.model)).length+' available';
 }catch(err){$('#config-error').textContent=err.message;$('#result').value='';}
}
function render(){cards();taskList();summary();}
$('#repository').value=C.release.repository_default;$('#search').oninput=e=>{state.query=e.target.value.toLowerCase();taskList();};
['repository','image-tag','storage','private-loras','private-checkpoints'].forEach(id=>$('#'+id).addEventListener('input',summary));
$('#select-all').onclick=()=>{C.tasks.filter(t=>t.workspaces.includes(state.model)).forEach(t=>state.selected[state.model].add(t.id));render();};
$('#select-default').onclick=()=>{state.selected[state.model]=new Set(C.models.find(m=>m.id===state.model).default_tasks);render();};
$('#copy-settings').onclick=()=>{try{copy(L.text(plan()));}catch(e){$('#config-error').textContent=e.message;}};
$('#copy-env').onclick=()=>{try{copy(L.envText(plan()));}catch(e){$('#config-error').textContent=e.message;}};
$('#download-json').onclick=()=>{try{saveText(state.model+'-runpod-settings.json',JSON.stringify(plan(),null,2),'application/json');}catch(e){$('#config-error').textContent=e.message;}};
$('#download-all').onclick=()=>{try{saveText('workbench-all-workspace-settings.json',JSON.stringify(C.models.map(m=>L.plan(C,m.id,[...state.selected[m.id]],[...state.features[m.id]],options())),null,2),'application/json');}catch(e){$('#config-error').textContent=e.message;}};
$('#save-selection').onclick=()=>saveText('workbench-selection.json',JSON.stringify({schema:1,repository:options().repository,tag:options().tag,storage:options().storage,selected:Object.fromEntries(Object.entries(state.selected).map(([k,v])=>[k,[...v]])),features:Object.fromEntries(Object.entries(state.features).map(([k,v])=>[k,[...v]]))},null,2),'application/json');
$('#load-selection').onchange=async e=>{try{const a=JSON.parse(await e.target.files[0].text());if(a.schema!==1)throw Error('Unsupported selection schema.');for(const m of C.models){L.plan(C,m.id,a.selected[m.id],a.features[m.id],a);state.selected[m.id]=new Set(a.selected[m.id]);state.features[m.id]=new Set(a.features[m.id]);}$('#repository').value=a.repository;$('#image-tag').value=a.tag;$('#storage').value=a.storage;render();}catch(err){$('#config-error').textContent=err.message;}};
window.WorkbenchApp={state,plan,render};render();
})();
