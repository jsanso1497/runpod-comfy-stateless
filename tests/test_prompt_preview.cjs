/* Frontend extension contract test with mocked ComfyUI hooks, not a live UI test. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
let extension;
const app={registerExtension(value){extension=value;}};
const ComfyWidgets={STRING(node,name){return {widget:{name,inputEl:{},options:{},value:''}};}};
const code=fs.readFileSync(path.join(__dirname,'../src/nodes/qwen_prompt/web/prompt_preview.js'),'utf8').replace(/^import[^\n]+\n/gm,'');
vm.runInNewContext(code,{app,ComfyWidgets,Set});
let checks=0;
(async()=>{
 for(const name of ['Q21NEncodeReferences','Q21PhotoEncode','WBQwenPromptPreview']){
  function Node(){this.size=[300,150];this.calls=0;}
  Node.prototype.onExecuted=function(){this.calls++;return 'preserved';};
  Node.prototype.computeSize=function(){return [320,240];};
  Node.prototype.setSize=function(value){this.size=value;};
  Node.prototype.setDirtyCanvas=function(){};
  await extension.beforeRegisterNodeDef(Node,{name});
  const n=new Node();
  assert.equal(n.onExecuted({qwen_prompt:[{status:'ON',cache_hit:false,effective_prompt:'Changed text',original_prompt:'Original'}]}),'preserved');
  assert.equal(n.calls,1);assert.equal(n._wbPromptPreview.inputEl.readOnly,true);
  assert.equal(n._wbPromptPreview.options.serialize,false);assert.equal(n._wbPromptPreview.serializeValue(),undefined);
  assert.match(n._wbPromptPreview.value,/Changed text/);assert.match(n._wbPromptPreview.value,/Original/);
  const widget=n._wbPromptPreview;
  n.onExecuted({qwen_prompt:[{status:'OFF',cache_hit:false,effective_prompt:'Original',original_prompt:'Original'}]});
  assert.equal(n._wbPromptPreview,widget);assert.match(widget.value,/OFF/);
  checks+=9;
 }
 function Unrelated(){};await extension.beforeRegisterNodeDef(Unrelated,{name:'KSampler'});
 assert.equal(Unrelated.prototype.onExecuted,undefined);checks++;
 console.log(`${checks} prompt-preview frontend assertions passed.`);
})().catch(err=>{console.error(err);process.exit(1);});
