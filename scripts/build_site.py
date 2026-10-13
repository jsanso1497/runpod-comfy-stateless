#!/usr/bin/env python3
"""Generate the offline page and ComfyUI catalog from one source of truth."""
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from workbench.config import read_catalog

def main():
 read=read_catalog
 assets=read('assets.json')
 catalog={'models':read('models.json')['models'],'tasks':read('tasks.json')['tasks'],'assets':assets['assets'],'ollama':assets['ollama'],'ollama_by_workspace':assets.get('ollama_by_workspace',{}),'release':read('release.json')}
 html=(ROOT/'site/index.template.html').read_text()
 parts={'STYLE':(ROOT/'site/style.css').read_text(),'CATALOG':'window.WORKBENCH_CATALOG='+json.dumps(catalog,separators=(',',':')).replace('</','<\\/')+';','LOGIC':(ROOT/'site/logic.js').read_text(),'APP':(ROOT/'site/app.js').read_text()}
 for name,value in parts.items():html=html.replace('/*'+name+'*/',value)
 (ROOT/'site/index.html').write_text(html)
 (ROOT/'site/catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
 print('Generated offline configurator: '+str(ROOT/'site/index.html'))
if __name__=='__main__':main()
