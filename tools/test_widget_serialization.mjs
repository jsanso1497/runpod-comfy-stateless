// Execute the actual migration function with a minimal app registration stub.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
let source=fs.readFileSync(path.join(root,'h3_portrait/node/web/director_controls.js'),'utf8');
source=source.replace(/^import .*$/mg,'').replaceAll('export ','');
const context={app:{registerExtension(){}},console};vm.createContext(context);
vm.runInContext(source+'\nglobalThis.migrate=migrateDirectorValues;globalThis.names=DIRECTOR_WIDGETS;',context);
const old=['brief','9:16','Standard',5,42,true,'Generate video',0,'(none)',1,'(none)',1,''];
const fixed=Array.from(context.migrate(old));
assert.equal(fixed.length,15);assert.equal(fixed[5],'fixed');assert.equal(fixed[6],true);
assert.equal(fixed[7],'Draft only');assert.equal(fixed[8],0);assert.equal(fixed[9],'(none)');assert.equal(fixed[10],1);assert.equal(fixed[11],'(none)');assert.equal(fixed[12],0);
assert.equal(fixed[14],'Role-aware (recommended)');
const corrupted=['brief','9:16','Standard',5,42,true,true,0,null,1,null,1,0,''];
assert.equal(context.migrate(corrupted)[7],'Draft only');assert.equal(context.migrate(corrupted)[9],'(none)');
assert.equal(context.migrate(fixed),null);
const graph=JSON.parse(fs.readFileSync(path.join(root,'h3_portrait/workflows/H3_Portrait_Auto.json'),'utf8'));
const values=graph.nodes.find(n=>n.type==='H3PortraitDirector').widgets_values;
assert.deepEqual(values.slice(1),fixed.slice(1));assert.equal(context.names.length,values.length);
console.log('H3 FRONTEND SERIALIZATION PASS: 15 widgets including seed control; legacy and corrupted migrations; new defaults unchanged.');
const stillGraph=JSON.parse(fs.readFileSync(path.join(root,'h3_portrait/workflows/H3_Portrait_Image_Lite.json'),'utf8'));
const stillValues=stillGraph.nodes.find(n=>n.type==='H3PortraitStillDirector').widgets_values;
assert.equal(stillValues.length,15);
assert.deepEqual(stillValues,[
  "Create one high-detail photorealistic still image of my subject. Preserve identity from the subject references. If I supply a pose/camera or scene reference, use it only for that assigned role and do not copy the guide person's identity.",
  'Match guide/primary','High-res (~2 MP, experimental)','High fidelity',42,'fixed',true,'Draft only',0,
  '(none)',1.0,'(none)',0.0,'','Still safe swap (pose/scene text-only)'
]);
assert.ok(stillValues.every(v=>!(typeof v==='number' && Number.isNaN(v))));
console.log('H3 REF2VA STILL SERIALIZATION PASS: 15 widgets including seed control; safe-swap + Draft-only defaults; no shifted/NaN values.');
