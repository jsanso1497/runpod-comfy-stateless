#!/usr/bin/env python3
"""Install versioned copies so old user-saved workflows cannot mask a release."""
import copy
import json
from pathlib import Path
import argparse
HERE=Path(__file__).resolve().parent


def configured_graph(graph,settings):
    graph=copy.deepcopy(graph);files=settings['model_files'];lite=settings.get('profile')=='lite'
    for node in graph['nodes']:
        typ=node['type'];values=node.get('widgets_values',[])
        if typ=='UNETLoader':values[0]=files['diffusion'];node['title']='H3 Ref2VA: full INT8' if lite else 'H3 Ref2VA: full BF16'
        elif typ=='CLIPLoader':values[0]=files['encoder'];node['title']='H3 encoder: INT8' if lite else 'H3 encoder: BF16'
        elif typ=='MiniMaxH3ReferenceToVideo' and lite:values[1:5]=[576,1024,124,'match']
        elif typ=='BasicScheduler' and lite:values[1]=16
    return graph


def install(home,settings,source=HERE):
    profile='Lite' if settings.get('profile')=='lite' else 'Full'
    folder=Path(home)/'user/default/workflows';folder.mkdir(parents=True,exist_ok=True)
    for src,stem in [('H3_Portrait_Auto.json','H3_Portrait'),('H3_Ref2VA_Standard.json','H3_Ref2VA_Standard')]:
        target=folder/f'{stem}_{profile}_v1_4.json'
        # Never overwrite the user's edits to this versioned copy on restart.
        if not target.exists():
            graph=configured_graph(json.loads((Path(source)/'workflows'/src).read_text()),settings)
            target.write_text(json.dumps(graph,indent=2)+'\n')
        print('H3 WORKFLOW INSTALLED: '+target.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',default='/workspace/ComfyUI');p.add_argument('--settings',default='/workspace/h3-portrait/config/settings.json');a=p.parse_args()
    install(a.home,json.loads(Path(a.settings).read_text()))
