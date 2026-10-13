import { app } from '../../scripts/app.js';

export const DIRECTOR_WIDGETS=['instruction','aspect','quality','seconds','seed','control_after_generate','use_loras','mode','prompt_variation','lora_1','strength_1','lora_2','strength_2','extra_trigger_words','reference_mode'];
export function migrateDirectorValues(values){
  if(!Array.isArray(values))return null;
  // 1.3 omitted the frontend-generated seed control, shifting every later value.
  if(values.length===13 && typeof values[5]==='boolean'){
    const corrected=[...values];corrected.splice(5,0,'fixed');
    corrected[7]='Draft only';corrected[12]=0;
    corrected.push('Role-aware (recommended)');return corrected;
  }
  // A user saved a previously shifted 1.3 node. Its original combo/float data
  // were already lost to coercion, so only preserve the unaffected first five.
  if(values.length>=14 && typeof values[5]==='boolean' && typeof values[7]!=='string'){
    return [...values.slice(0,5),'fixed',true,'Draft only',0,'(none)',1,'(none)',0,'','Role-aware (recommended)'];
  }
  return null;
}
app.registerExtension({
 name:'H3Portrait.DirectorControls',
 async beforeRegisterNodeDef(NodeType,data){
  if(data.name!=='H3PortraitDirector')return;
  const previous=NodeType.prototype.onConfigure;
  NodeType.prototype.onConfigure=function(info){
   previous?.apply(this,arguments);
   const corrected=migrateDirectorValues(info?.widgets_values);
   if(!corrected)return;
   for(let i=0;i<DIRECTOR_WIDGETS.length;i++){
    const name=DIRECTOR_WIDGETS[i];
    const widget=this.widgets?.find(w=>w.name===name || (name==='control_after_generate' && w.name==='control after generate'));
    if(widget)widget.value=corrected[i];
   }
   this.setDirtyCanvas(true,true);
   console.warn('H3 Portrait: repaired legacy shifted widgets. Review LoRA selection and use Draft only first.');
  };
 }
});
