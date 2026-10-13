// DOM contract harness, not a real ComfyUI/frontend rendering test.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
class Element {
  constructor(tag) { this.tag=tag;this.style={};this.children=[];this.options=[];this.events={};this.value='';this.checked=false;this.textContent=''; }
  append(...items) { this.children.push(...items); }
  add(option) { this.options.push(option);if(this.options.length===1)this.value=option.value; }
  setAttribute(k,v) { this[k]=v; }
  addEventListener(k,v) { this.events[k]=v; }
  replaceChildren() { this.children=[];this.options=[];this.value=''; }
  get selectedIndex() { return this.options.findIndex(x=>x.value===this.value); }
  remove() { this.removed=true; }
}
let extension;
const context={
  document:{createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text})},
  Option:function(text,value){this.text=text;this.value=value;},
  URLSearchParams, console,
  app:{registerExtension:obj=>extension=obj},api:{apiURL:value=>value}
};
vm.createContext(context);
const source=fs.readFileSync(path.join(__dirname,'../src/nodes/everyday/web/green_suit_compare.js'),'utf8')
  .replace(/^import .*;\n/gm,'');
vm.runInContext(source,context);
class Node { constructor(){this.size=[500,500];} addDOMWidget(){return {};} setSize(size){this.size=size;} }
extension.beforeRegisterNodeDef(Node,{name:'WBGSReviewCompare'});
const node=new Node();node.onNodeCreated();
assert.ok(node._wbgsViewer);
const labels=['Original','SAM mask','Qwen green suit','4K geometry reference','Raw GPT native','Matte-locked native','Matte-locked 4K'];
function message(count){ const items=labels.slice(0,count).map((label,i)=>({filename:`test_${i}.png`,subfolder:'test folder',type:'temp',label}));
  return {wbgs_images:items,wbgs_saved:items.map(x=>({...x,type:'output'})),wbgs_report:['{"test":true}']}; }
node.onExecuted(message(2));
const root=node._wbgsViewer.root;
const [a,b]=root.children[0].children;
const mode=root.children[1].children[0],full=root.children[1].children[1].children[0];
const [imgA,imgB]=root.children[2].children;
const slider=root.children[3];
assert.equal(a.value,'0');assert.equal(b.value,'1');
node.onExecuted(message(4));assert.equal(a.value,'2');assert.equal(b.value,'3');
node.onExecuted(message(7));assert.equal(a.value,'3');assert.equal(b.value,'6');
assert.match(imgA.src,/filename=test_3.png/);assert.match(imgB.src,/filename=test_6.png/);
mode.value='overlay';mode.events.change();assert.equal(imgB.style.opacity,'0.5');
mode.value='wipe';slider.value='25';slider.events.input();assert.equal(imgB.style.clipPath,'inset(0 0 0 25%)');
full.checked=true;full.events.change();assert.match(imgA.src,/type=output/);
node.onRemoved();assert.equal(root.removed,true);
console.log('Viewer DOM contracts passed: stage defaults, A/B image URLs, overlay, wipe, full size, cleanup.');
