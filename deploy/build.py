"""Build only source-pinned software. No private inputs or model weights are accepted."""
from pathlib import Path
import importlib.metadata as md,json,os,shutil,subprocess,sys
ROOT=Path('/opt/workbench');COMFY=Path('/opt/ComfyUI')
def run(*args,**kwargs):subprocess.run(args,check=True,**kwargs)
def clone(url,commit,dest):
 dest.mkdir(parents=True,exist_ok=True)
 run('git','init','-q',str(dest));run('git','-C',str(dest),'remote','add','origin',url)
 run('git','-C',str(dest),'fetch','--depth','1','origin',commit)
 run('git','-C',str(dest),'checkout','--detach','FETCH_HEAD')
 actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
 if actual!=commit:raise ValueError('Pinned checkout mismatch.')
 shutil.rmtree(dest/'.git')
def main():
 name=sys.argv[1];cat=json.loads((ROOT/'catalog/models.json').read_text());model=next(m for m in cat['models'] if m['id']==name)
 constraints=Path('/tmp/torch-constraints.txt')
 pins=[]
 for package in ('torch','torchvision','torchaudio'):
  try:pins.append(package+'=='+md.version(package))
  except md.PackageNotFoundError:pass
 constraints.write_text('\n'.join(pins)+'\n')
 def pip(*args):run(sys.executable,'-m','pip','install','--constraint',str(constraints),*args)
 clone('https://github.com/Comfy-Org/ComfyUI.git',model['comfy_commit'],COMFY)
 pip('-r',str(COMFY/'requirements.txt'));pip('-r',str(ROOT/'requirements.txt'))
 for id in model['external_nodes']:
  row=cat['external_nodes'][id];dest=COMFY/'custom_nodes'/row['directory'];clone(row['url'],row['commit'],dest)
  if (dest/'requirements.txt').exists():pip('-r',str(dest/'requirements.txt'))
  if id=='h3-refpack':run(sys.executable,str(ROOT/'deploy/patch_refpack_local.py'),str(dest))
 # No old base-image inheritance and no catch-all COPY of the uploaded source tree.
 for id in model['local_nodes']+['workbench_tools','workbench_ui']:
  dest=COMFY/'custom_nodes'/('Workbench_'+id)
  dest.symlink_to(ROOT/'src/nodes'/id,target_is_directory=True)
 run(sys.executable,'-m','pip','check')
 (ROOT/'BUILD_WORKSPACE').write_text(name+'\n')
 (ROOT/'build-provenance.json').write_text(json.dumps({'workspace':name,'comfy_commit':model['comfy_commit'],'external_nodes':{id:cat['external_nodes'][id] for id in model['external_nodes']},'python':sys.version,'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()},indent=2))
if __name__=='__main__':main()
