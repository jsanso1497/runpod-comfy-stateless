#!/usr/bin/env python3
"""Install versioned copies so old user-saved workflows cannot mask a release."""
import copy
import json
from pathlib import Path
import argparse
# Repository checkout and Docker install share the same loader-policy code.
import sys
_safety = Path(__file__).resolve().parents[1]/'model_safety'
if not _safety.is_dir(): _safety = Path('/opt/model-safety')
sys.path.insert(0, str(_safety))
from bridge import protect_graph

HERE=Path(__file__).resolve().parent


def configured_graph(graph,settings):
    graph=copy.deepcopy(graph);files=settings['model_files'];lite=settings.get('profile')=='lite'
    for node in graph['nodes']:
        typ=node['type'];values=node.get('widgets_values',[])
        if typ=='UNETLoader':values[0]=files['diffusion'];node['title']='H3 Ref2VA: full INT8' if lite else 'H3 Ref2VA: full BF16'
        elif typ=='CLIPLoader':values[0]=files['encoder'];values[1]='minimax';node['title']='H3 encoder: INT8' if lite else 'H3 encoder: BF16'
        elif typ=='MiniMaxH3ReferenceToVideo' and lite:values[1:5]=[576,1024,124,'match']
        elif typ=='MiniMaxH3ReferencePack':
            values[3]='local';values[4]='';values[5]='(hosted models disabled)';values[6]='none'
            values[7]='http://127.0.0.1:11434/v1';values[8]=settings['analysis_model'];values[9]='replacement';values[13]=2048
            node['title']='2. Subject images + reference video | Local Ollama only'
        elif typ=='H3ReferenceVideoSettings' and lite:values[1]='Standard'
        elif typ=='BasicScheduler' and lite:values[1]=16
        if node.get('widgets_values_named'):
            fields={'UNETLoader':('unet_name','weight_dtype'),'CLIPLoader':('clip_name','type','device'),'VAELoader':('vae_name',)}.get(typ)
            if fields: node['widgets_values_named']=dict(zip(fields,values))
    return protect_graph(graph)


def install(home,settings,source=HERE):
    lite=settings.get('profile')=='lite';profile='Lite' if lite else 'Full'
    folder=Path(home)/'user/default/workflows';folder.mkdir(parents=True,exist_ok=True)
    recipes=[('H3_Portrait_Auto.json','H3_Portrait'),('H3_Ref2VA_Standard.json','H3_Ref2VA_Standard'),('H3_Reference_Video_Swap_Local.json','H3_Reference_Video_Swap_Local')]
    if lite:recipes.append(('H3_Portrait_Image_Lite.json','H3_Portrait_Image'))
    for src,stem in recipes:
        target=folder/f'{stem}_{profile}_v1_5_3.json'
        # Never overwrite the user's edits to this versioned copy on restart.
        if not target.exists():
            graph=configured_graph(json.loads((Path(source)/'workflows'/src).read_text()),settings)
            target.write_text(json.dumps(graph,indent=2)+'\n')
        print('H3 WORKFLOW INSTALLED: '+target.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',default='/workspace/ComfyUI');p.add_argument('--settings',default='/workspace/h3-portrait/config/settings.json');a=p.parse_args()
    cfg=json.loads(Path(a.settings).read_text())
    import os
    cfg['analysis_model']=os.environ.get('OLLAMA_ANALYSIS_MODEL','').strip() or cfg['analysis_model']
    install(a.home,cfg)
