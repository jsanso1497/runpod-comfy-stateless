#!/usr/bin/env python3
"""Offline contract validation. This does not import ComfyUI or claim GPU compatibility."""
from pathlib import Path
import argparse,ast,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
def validate_graph(graph,path='graph'):
 errors=[];ids=[n['id'] for n in graph['nodes']];nodes={n['id']:n for n in graph['nodes']};links={l[0]:l for l in graph.get('links',[])}
 if len(ids)!=len(nodes):errors.append(f'{path}: duplicate node IDs')
 if len(links)!=len(graph.get('links',[])):errors.append(f'{path}: duplicate link IDs')
 for lid,l in links.items():
  if len(l)!=6 or l[1] not in nodes or l[3] not in nodes:errors.append(f'{path}: dangling link {lid}');continue
  try:
   out=nodes[l[1]]['outputs'][l[2]];inp=nodes[l[3]]['inputs'][l[4]]
   if lid not in (out.get('links') or []):errors.append(f'{path}: output does not contain link {lid}')
   if inp.get('link')!=lid:errors.append(f'{path}: input mismatch on link {lid}')
  except (IndexError,KeyError):errors.append(f'{path}: invalid socket index on link {lid}')
 for n in graph['nodes']:
  for port in n.get('inputs',[]):
   if port.get('link') is not None and port['link'] not in links:errors.append(f'{path}: input has nonexistent link')
  for port in n.get('outputs',[]):
   for lid in port.get('links') or []:
    if lid not in links:errors.append(f'{path}: output has nonexistent link')
 return errors

def run():
 errors=[];tasks=json.loads((ROOT/'catalog/tasks.json').read_text())['tasks'];models=json.loads((ROOT/'catalog/models.json').read_text())['models']
 required=['.github/workflows/build.yml','.github/workflows/ci.yml','.github/workflows/pages.yml','.gitignore','.dockerignore','.env.example']
 for name in required:
  if not (ROOT/name).is_file():errors.append('Missing upload file: '+name+'; restore it from the Mac/browser guide')
 ids=[t['id'] for t in tasks];files=[]
 if len(ids)!=len(set(ids)):errors.append('Duplicate task IDs')
 if set(ids)&{'Q29','Q30','Q31','Q32','Q33','D02','D03'}:errors.append('Removed task is active')
 for m in models:
  if m['id'] not in ('qwen','flux','h3','restoration'):errors.append('Unexpected model')
  for id in m['default_tasks']:
   if id not in ids:errors.append('Missing default task')
  if not re.fullmatch('[a-f0-9]{40}',m['comfy_commit']):errors.append('Unpinned Comfy commit')
  for pkg in m['local_nodes']+['workbench_tools','workbench_ui']:
   if not (ROOT/'src/nodes'/pkg/'__init__.py').is_file():errors.append('Missing local package: '+pkg)
 for t in tasks:
  if not t['variants']:errors.append(t['id']+': no workflow')
  for v in t['variants']:
   file=v['file'];files.append(file);p=ROOT/file
   if not p.is_file():errors.append('Missing workflow: '+file);continue
   graph=json.loads(p.read_text());errors.extend(validate_graph(graph,file))
   for n in graph['nodes']:
    if 'Krea' in n['type'] or 'Torso' in n['type']:errors.append('Removed node in '+file)
    for value in n.get('widgets_values',[]):
     if isinstance(value,str) and (re.search(r'\.safetensors$',value) and re.search(r'(int8|fp8|nvfp4|gguf)',value,re.I)):
      errors.append('Quantized weight selection in '+file)
     if isinstance(value,str) and re.search(r'https?://[^\s]+[?&](token|api_key|apikey)=',value,re.I):errors.append('Token-bearing URL in workflow')
 if len(files)!=len(set(files)):errors.append('Workflow file mapped more than once')
 for p in (ROOT/'src').rglob('*.py'):
  try:ast.parse(p.read_text())
  except SyntaxError as exc:errors.append(str(p.relative_to(ROOT))+': '+str(exc))
 migration=json.loads((ROOT/'catalog/migration.json').read_text())
 rows=migration if isinstance(migration,list) else migration.get('workflows',migration.get('source_workflows',migration.get('entries',[])))
 if len(rows)!=82:errors.append('Expected 82 source workflow dispositions, found '+str(len(rows)))
 for p in ROOT.rglob('*'):
  if not p.is_file() or '.git' in p.parts:continue
  if p.suffix in ('.safetensors','.ckpt','.gguf','.pt','.pth'):errors.append('Model file in source package')
  if 'lora' in p.name.lower() and p.suffix=='.txt':errors.append('Private LoRA text file in source package')
  if p.name=='.env':errors.append('Runtime environment file in source package')
 if 'COPY . ' in (ROOT/'Dockerfile').read_text():errors.append('Docker context copied without explicit allowlist')
 return {'tasks':len(tasks),'workflow_files':len(files),'source_dispositions':len(rows),'errors':errors,'test_scope':'static only; no GPU, Docker build or provider download'}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--no-imports',action='store_true');parser.add_argument('--report');a=parser.parse_args();report=run()
 if a.report:Path(a.report).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2));raise SystemExit(bool(report['errors']))
