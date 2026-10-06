import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

app.registerExtension({
  name: 'H3Portrait.ReferenceUpload',
  async beforeRegisterNodeDef(NodeType, data) {
    if (data.name !== 'H3PortraitReferences') return;
    const created = NodeType.prototype.onNodeCreated;
    NodeType.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const node = this;
      const state = node.widgets.find(w => w.name === 'reference_files');
      state.hidden = true;
      state.computeSize = () => [0, -4];
      const root = document.createElement('div');
      root.style.cssText = 'padding:8px;box-sizing:border-box;overflow:auto;height:100%;background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);font:13px sans-serif;';
      const picker = document.createElement('input');
      picker.type = 'file'; picker.multiple = true;
      picker.accept = 'image/png,image/jpeg,image/webp'; picker.style.display = 'none';
      const status = document.createElement('div');
      const list = document.createElement('div');
      const button = document.createElement('button');
      button.textContent = 'Upload reference images (1-9)';
      button.style.cssText='width:100%;padding:8px;cursor:pointer;margin-bottom:6px;';
      root.append(button,picker,status,list);
      const names = () => { try { const v=JSON.parse(state.value || '[]'); return Array.isArray(v)?v.map(x=>typeof x==='string'?{filename:x,role:'auto'}:x):[]; } catch { return []; } };
      const write = a => { state.value=JSON.stringify(a); state.callback?.(state.value); redraw(); node.setDirtyCanvas(true,true); };
      const redraw = () => {
        list.replaceChildren(); const files=names();status.textContent=`${files.length}/9 images. Use these numbers in your normal-language instructions.`;
        files.forEach((entry,index) => {
          const file=entry.filename;
          if(typeof file!=='string')return;
          const row=document.createElement('div');row.style.cssText='display:flex;align-items:center;gap:6px;padding:5px 0;border-bottom:1px solid #555;';
          const thumb=document.createElement('img');thumb.style.cssText='width:48px;height:58px;object-fit:contain;';
          const slash=file.lastIndexOf('/');
          thumb.src=api.apiURL('/view?'+new URLSearchParams({filename:file.slice(slash+1),subfolder:slash>=0?file.slice(0,slash):'',type:'input'}));
          const label=document.createElement('span');label.textContent=`${index+1}. ${file.slice(slash+1)}`;label.style.cssText='flex:1;overflow-wrap:anywhere;';
          row.append(thumb,label);
          const select=document.createElement('select');select.title='This role limits what this reference contributes.';
          for(const [value,text] of [['auto','Auto from brief'],['identity','Subject identity'],['face','Face only'],['body','Body/proportions only'],['hair','Hair only'],['wardrobe','Wardrobe only'],['pose_camera','Pose/camera only (text guide)'],['expression','Expression only (text guide)'],['scene','Scene only'],['ignore','Ignore']]){
            const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);
          }
          select.value=entry.role||'auto';select.style.maxWidth='205px';
          select.onchange=()=>{const a=names();a[index]={...a[index],role:select.value};write(a);};row.append(select);
          for (const [text,delta] of [['Up',-1],['Down',1],['Remove',0]]) {
            const b=document.createElement('button');b.textContent=text;
            b.disabled=delta<0?index===0:delta>0?index===files.length-1:false;
            b.onclick=()=>{const a=names();if(!delta)a.splice(index,1);else [a[index],a[index+delta]]=[a[index+delta],a[index]];write(a);};row.append(b);
          }
          list.append(row);
        });
      };
      button.onclick=()=>picker.click();
      picker.onchange=async()=>{
        const selected=Array.from(picker.files || []);picker.value='';
        if(names().length+selected.length>9){alert('H3 accepts at most 9 reference images. Remove some first; no images were uploaded.');return;}
        button.disabled=true;
        try {
          for(const file of selected){
            status.textContent=`Uploading ${file.name}...`;
            const body=new FormData();body.append('image',file);body.append('type','input');body.append('subfolder','h3_portrait');body.append('overwrite','false');
            const r=await api.fetchApi('/upload/image',{method:'POST',body});
            if(!r.ok)throw new Error(`Upload failed: HTTP ${r.status}`);
            const saved=await r.json();if(!saved.name)throw new Error('Upload did not return a filename.');
            write([...names(),{filename:[saved.subfolder,saved.name].filter(Boolean).join('/'),role:'auto'}]);
          }
        }catch(e){alert(e.message);}finally{button.disabled=false;redraw();}
      };
      node.addDOMWidget('reference_gallery','h3-reference-upload',root,{serialize:false,hideOnZoom:false,getMinHeight:()=>340,getMaxHeight:()=>560});
      node._h3PortraitRefresh=redraw;
      node.setSize([760,480]);setTimeout(redraw,0);
    };
    const configured=NodeType.prototype.onConfigure;
    NodeType.prototype.onConfigure=function(){configured?.apply(this,arguments);setTimeout(()=>this._h3PortraitRefresh?.(),0);};
  }
});
