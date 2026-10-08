#!/usr/bin/env python3
"""Validate node schemas against a RUNNING image. Never queues a GPU generation."""
from pathlib import Path
import argparse,json,os,sys,urllib.request,base64
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',required=True,choices=['qwen','flux','h3','restoration']);p.add_argument('--url',default='http://127.0.0.1:8189');p.add_argument('--report',default='live-schema-report.json');a=p.parse_args()
 headers={};password=os.environ.get('WB_PASSWORD')
 if password:headers['Authorization']='Basic '+base64.b64encode(('workbench:'+password).encode()).decode()
 req=urllib.request.Request(a.url.rstrip('/')+'/object_info',headers=headers)
 with urllib.request.urlopen(req,timeout=60) as r:schema=json.load(r)
 errors=[];count=0
 for task in json.loads((ROOT/'catalog/tasks.json').read_text())['tasks']:
  if a.workspace not in task['workspaces']:continue
  for variant in task['variants']:
   g=json.loads((ROOT/variant['file']).read_text());count+=1
   for n in g['nodes']:
    if n['type'] in ('Note','MarkdownNote','Reroute','PrimitiveNode'):continue
    entry=schema.get(n['type'])
    if entry is None:errors.append({'task':task['id'],'workflow':variant['file'],'node':n['id'],'error':'Missing node type: '+n['type']});continue
    allowed={**entry.get('input',{}).get('required',{}),**entry.get('input',{}).get('optional',{})}
    # Autogrow sockets use prefixes in native v3 schemas and are checked by live prompt validation later.
    for socket in n.get('inputs',[]):
     name=socket['name']
     if socket.get('link') is not None and name not in allowed and not any(name.startswith(k+'.') or name.startswith(k+'_') for k in allowed):
      errors.append({'task':task['id'],'workflow':variant['file'],'node':n['id'],'error':'Unknown linked input: '+name})
    if len(n.get('outputs',[]))>len(entry.get('output',[])):
     errors.append({'task':task['id'],'workflow':variant['file'],'node':n['id'],'error':'Output count differs from live schema'})
 report={'workspace':a.workspace,'workflows':count,'errors':errors,'gpu_execution_tested':False}
 Path(a.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return bool(errors)
if __name__=='__main__':raise SystemExit(main())
