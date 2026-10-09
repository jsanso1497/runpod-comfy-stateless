#!/usr/bin/env python3
"""Generate the offline page and ComfyUI catalog from one source of truth."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
 read=lambda f:json.loads((ROOT/'catalog'/f).read_text())
 assets=read('assets.json')
 catalog={'models':read('models.json')['models'],'tasks':read('tasks.json')['tasks'],'assets':assets['assets'],'ollama':assets['ollama'],'release':read('release.json')}
 html=(ROOT/'site/index.template.html').read_text()
 parts={'STYLE':(ROOT/'site/style.css').read_text(),'CATALOG':'window.WORKBENCH_CATALOG='+json.dumps(catalog,separators=(',',':')).replace('</','<\\/')+';','LOGIC':(ROOT/'site/logic.js').read_text(),'APP':(ROOT/'site/app.js').read_text()}
 for name,value in parts.items():html=html.replace('/*'+name+'*/',value)
 (ROOT/'site/index.html').write_text(html)
 (ROOT/'site/catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
 print('Generated offline configurator: '+str(ROOT/'site/index.html'))
if __name__=='__main__':main()
