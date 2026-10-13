const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const path = require('path');
let extension;
const source = fs.readFileSync(path.join(__dirname, '../src/nodes/qwen_prompt/web/prompt_studio.js'), 'utf8')
  .replace(/^import .*;\s*$/gm, '');
const copied = [], downloads = [];
const context = {
  app: {registerExtension(value) { extension = value; }},
  ComfyWidgets: {STRING(node, name) {
    const widget = {name, value: '', options: {}, inputEl: {readOnly: false}};
    node.widgets.push(widget); return {widget};
  }},
  navigator: {clipboard: {async writeText(text) { copied.push(text); }}},
  URL: {createObjectURL() { return 'blob:test'; }, revokeObjectURL() {}},
  Blob: class { constructor(parts) { this.parts = parts; } },
  document: {createElement() { const a={click() { downloads.push(a.download); }}; return a; }},
  setTimeout(fn) { fn(); }
};
vm.runInNewContext(source, context);
class Node {
  constructor() { this.widgets=[]; this.size=[650,1150]; }
  addWidget(type,name,value,callback) { const w={type,name,value,callback}; this.widgets.push(w); return w; }
  computeSize() { return [650,1150]; }
  setSize(v) { this.size=v; }
  setDirtyCanvas() {}
}
(async () => {
  await extension.beforeRegisterNodeDef(Node, {name:'WBQwenPromptStudio'});
  const node = new Node(); node.onNodeCreated();
  assert.equal(node.widgets.length,4);
  const message={text:['Replace the dog.'],studio_status:['done'],studio_roles:['Image 1: canvas'],studio_filename:['test.txt'],studio_saved:['prompt_studio/test.txt']};
  node.onExecuted(message);
  assert.equal(node._studioPreview.value,'Replace the dog.');
  assert(node._studioStatus.value.includes('Image 1: canvas'));
  await node.widgets.find(w=>w.name==='Copy final prompt').callback();
  node.widgets.find(w=>w.name==='Download prompt .txt').callback();
  assert.deepEqual(copied,['Replace the dog.']);
  assert.deepEqual(downloads,['test.txt']);
  node.onExecuted(message);
  assert.equal(node.widgets.length,4,'repeated execution must not add duplicate widgets');
  for (const w of node.widgets) { assert.equal(w.options.serialize,false); assert.equal(w.serializeValue(),undefined); }
  node.onExecutionStart();
  assert.equal(node._studioPrompt,'');
  node.widgets.find(w=>w.name==='Download prompt .txt').callback();
  assert.equal(downloads.length,1,'do not download a stale prompt during a new execution');
  console.log('Prompt Studio UI: copy, text export, transient widgets and stale-output guard passed.');
})().catch(e=>{console.error(e);process.exitCode=1;});
