/* Shared pure configuration logic. No secret values, analytics, or remote requests. */
(function(root){
'use strict';
function validRepo(repo){return /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(repo);}
function plan(catalog,modelId,selected,features,options={}){
 const m=catalog.models.find(x=>x.id===modelId);if(!m)throw Error('Unknown workspace.');
 const tasks=selected.map(id=>{const t=catalog.tasks.find(x=>x.id===id);if(!t||!t.workspaces.includes(modelId))throw Error('Task is not compatible with this workspace: '+id);return t;});
 if(features.some(f=>!m.toolbox.includes(f)))throw Error('Unsupported toolbox feature.');
 const groups=[...new Set([...m.default_groups,...features,...tasks.flatMap(t=>t.requires)])].sort();
 const repo=(options.repository||catalog.release.repository_default).trim();if(!validRepo(repo))throw Error('Use owner/repository, with no URL or spaces.');
 const tag=(options.tag||'hq').trim();if(!/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,90}$/.test(tag))throw Error('Invalid image tag.');
 const fixed=catalog.assets.filter(a=>groups.includes(a.group));
 const known=fixed.reduce((s,a)=>s+(a.estimated_bytes||0),0)+(groups.includes('ollama')?catalog.ollama.estimated_bytes:0);
 const modelGB=Math.ceil(known/1e9);const capacity=Math.max(m.storage_gb,Math.ceil((modelGB*1.15+90)/25)*25);
 const persistent=options.storage==='persistent';
 const env={WB_WORKSPACE:modelId,WB_TASKS:tasks.map(t=>t.id).join(','),WB_TOOLBOX:features.join(','),
    HF_TOKEN:'{{ RUNPOD_SECRET_hf_token }}',CIVITAI_TOKEN:'{{ RUNPOD_SECRET_civit_token }}',WB_PASSWORD:'{{ RUNPOD_SECRET_comfy_password }}'};
 if(options.loras)env.WB_LORA_URLS='{{ RUNPOD_SECRET_comfy_loras }}';
 if(options.checkpoints)env.WB_CHECKPOINTS='{{ RUNPOD_SECRET_comfy_checkpoints }}';
 const result={name:'Comfy Workbench | '+m.short+' HQ',workspace:modelId,image:'ghcr.io/'+repo.toLowerCase()+':'+modelId+'-'+tag,
   image_status:'Build required. This configurator does not prove the image exists.',container_disk_gb:persistent?80:capacity,
   volume_disk_gb:persistent?capacity:0,volume_mount_path:'/workspace',http_ports:'8188',tcp_ports:'',container_start_command:'',environment:env,
   selected_tasks:tasks.map(t=>({id:t.id,title:t.title})),asset_groups:groups,estimated_model_download_gb:modelGB,unestimated_assets:fixed.filter(a=>!a.estimated_bytes).map(a=>a.id),
   planning_gpu_vram_gb:m.vram_gb,planning_system_ram_gb:m.ram_gb,
   hardware_note:m.hardware_note+' Planning targets are not certified minima. Optional F16 helpers add substantial RAM/storage requirements.',
   storage_note:persistent?'Volume disk survives stop/start, not Pod deletion. Use a separately selected network volume for deletion durability.':'Stateless: container data can be lost on stop/restart or deletion. Export results before ending the Pod.',
   schema_note:'Human-readable RunPod setup details. Not a RunPod API request or guaranteed import schema.',
   username:'workbench',password_secret:'comfy_password',registry_note:'For a private GHCR image, configure a RunPod container-registry credential. Never put the registry token in this page.'};
 return result;
}
function envText(p){return Object.entries(p.environment).filter(([,v])=>v!=='').map(([k,v])=>k+'='+v).join('\n');}
function text(p){return ['TEMPLATE: '+p.name,'IMAGE: '+p.image,'STATUS: '+p.image_status,'CONTAINER DISK: '+p.container_disk_gb+' GB','VOLUME DISK: '+p.volume_disk_gb+' GB','VOLUME MOUNT: '+p.volume_mount_path,'HTTP PORTS: '+p.http_ports,'TCP PORTS: none','CONTAINER START COMMAND: leave blank','LOGIN USERNAME: workbench','LOGIN PASSWORD: value of RunPod Secret comfy_password','ASSET GROUPS: '+p.asset_groups.join(', '),'','ENVIRONMENT VARIABLES (one key/value row per line):',envText(p),'','NOTES:',p.storage_note,p.hardware_note,p.registry_note,'Download estimate excludes your private libraries. Allow extra room for images, videos and caches.'].join('\n');}
const api={plan,envText,text,validRepo};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.WorkbenchLogic=api;
})(typeof window==='undefined'?globalThis:window);
