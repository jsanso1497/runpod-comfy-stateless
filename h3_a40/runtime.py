"""Focused startup: four existing H3/Eros assets, no inherited bulk preloads."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

P=Path(__file__).resolve().parent
sys.path.insert(0,'/opt/h3-still/runtime')


def checked_root(value=None):
    root=Path(value or os.environ.get('H3_STILL_DATA_ROOT','/workspace/h3-still')).resolve()
    if root==Path('/workspace') or not root.is_relative_to(Path('/workspace')):
        raise ValueError('H3_STILL_DATA_ROOT must be a directory within /workspace.')
    return root


def prepare_files(root):
    from bootstrap import safe_child
    for name in ('input','output','user','temp','logs','.verified','models/loras','models/diffusion_models','models/text_encoders','models/vae'):
        safe_child(root,name).mkdir(parents=True,exist_ok=True)
    factory=safe_child(root,'user/default/workflows/00_H3_A40_Identity')
    factory.mkdir(parents=True,exist_ok=True)
    for source in sorted((P/'workflows').glob('*.json')):
        target=safe_child(factory,source.name)
        if target.is_symlink():raise RuntimeError('Refusing a symlinked workflow: '+str(target))
        try:
            with target.open('xb') as f:f.write(source.read_bytes())
        except FileExistsError:pass
    config={'h3_a40':dict(base_path=str(root/'models'),is_default=True,
                       diffusion_models='diffusion_models',text_encoders='text_encoders',vae='vae',loras='loras')}
    dest=safe_child(root,'extra_model_paths_a40.yaml')
    if dest.is_symlink():raise RuntimeError('Refusing a symlinked model-path configuration.')
    # JSON is valid YAML and quotes paths without interpolation.
    dest.write_text(json.dumps(config,indent=2)+'\n')
    return dest


def download(root):
    if os.environ.get('H3V_LICENSE_CONFIRMED')!='1':
        raise RuntimeError('Confirm applicable model authorization before setting H3V_LICENSE_CONFIRMED=1. See LICENSE_NOTES.md. Files/Jupyter remain accessible.')
    from bootstrap import safe_child,verified
    from huggingface_hub import hf_hub_download
    import fcntl
    rows=json.loads((P/'manifest.json').read_text())['models']
    lock=safe_child(root,'.h3-a40-download.lock')
    if lock.is_symlink():raise ValueError('Refusing a symlinked download lock.')
    print('A40 profile: four model files, approximately %.1f GB total. Existing verified files are reused.'%sum(r['approx_gb'] for r in rows),flush=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        for index,row in enumerate(rows,1):
            dest=safe_child(root/'models',row['destination'])
            marker=safe_child(root,'.verified/a40-'+row['id']+'.json')
            if verified(dest,row,marker,os.environ.get('H3_STILL_VERIFY_ALL')=='1'):
                print('[%d/%d] Reusing verified %s'%(index,len(rows),row['id']),flush=True);continue
            if dest.exists() or dest.is_symlink():
                raise RuntimeError('Existing model failed verification; not overwritten: '+str(dest))
            local=safe_child(root/'models','.sources/a40-'+row['id']);local.mkdir(parents=True,exist_ok=True)
            completed=safe_child(local,row['file'])
            if not verified(completed,row,marker,False):
                need=int(row['approx_gb']*1e9)+2_000_000_000
                if shutil.disk_usage(root).free<need:
                    raise RuntimeError('Insufficient free storage for '+row['id']+'. Use a 160 GB or larger persistent volume for this fresh profile.')
                print('[%d/%d] Downloading %s (approximately %.1f GB)'%(index,len(rows),row['id'],row['approx_gb']),flush=True)
                completed=Path(hf_hub_download(repo_id=row['repo'],filename=row['file'],revision=row['revision'],
                    local_dir=str(local),token=os.environ.get('HF_TOKEN') or None))
                if not completed.resolve().is_relative_to(local.resolve()) or not verified(completed,row,marker,True):
                    raise RuntimeError('Downloaded checksum verification failed: '+row['id'])
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.symlink_to(os.path.relpath(completed,dest.parent))
    print('H3 A40 Identity Studio models ready. Starting ComfyUI.',flush=True)


def verify_metadata():
    from huggingface_hub import HfApi
    rows=json.loads((P/'manifest.json').read_text())['models'];cache={};checks=[]
    api=HfApi(token=os.environ.get('HF_TOKEN') or None)
    for row in rows:
        key=row['repo'],row['revision']
        if key not in cache:
            value=api.model_info(row['repo'],revision=row['revision'],files_metadata=True)
            if value.sha!=row['revision']:raise RuntimeError('Publisher revision mismatch.')
            cache[key]={s.rfilename:s for s in value.siblings}
        sibling=cache[key].get(row['file']);lfs=getattr(sibling,'lfs',None)
        sha=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
        if sha!=row['sha256']:raise RuntimeError('Publisher checksum mismatch or missing asset: '+row['id'])
        checks.append(dict(id=row['id'],sha256=sha,size_bytes=getattr(sibling,'size',None)))
    report=dict(checks=checks,model_weights_downloaded=False,gpu_tested=False)
    (P/'metadata-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def backend():
    import torch
    from recovery_launcher import terminal_environment
    if not torch.cuda.is_available():raise RuntimeError('CUDA is not available. Use a CUDA 13-compatible NVIDIA host. Files/Jupyter remain available.')
    root=checked_root();root.mkdir(parents=True,exist_ok=True)
    prop=torch.cuda.get_device_properties(0)
    print('GPU:',prop.name,'VRAM GiB:',round(prop.total_memory/1024**3,2),flush=True)
    if prop.total_memory<44*1024**3:print('WARNING: This profile targets a 48 GB A40; smaller GPUs have not been qualified.',flush=True)
    if not os.path.ismount('/workspace'):
        print('STORAGE WARNING: /workspace is not reported as a separate mount. Verify persistence before relying on it.',flush=True)
    config=prepare_files(root)
    # Deliberately never call the old bootstrap.prepare or video_studio.download.
    # H3V_PRELOAD=all from an old template cannot trigger the 375 GB download set.
    download(root)
    command=[sys.executable,'/opt/ComfyUI/main.py','--listen','127.0.0.1','--port','8189',
        '--input-directory',str(root/'input'),'--output-directory',str(root/'output'),
        '--user-directory',str(root/'user'),'--temp-directory',str(root/'temp'),
        '--extra-model-paths-config',str(config),'--preview-method','none','--disable-auto-launch',
        '--use-pytorch-cross-attention','--reserve-vram','2.0','--cache-classic','--max-upload-size','4096']
    # Keep normal model offloading. --highvram would prevent effective stage separation.
    env=terminal_environment();os.chdir('/opt/ComfyUI');os.execve(sys.executable,command,env)


def launch():
    root=checked_root();root.mkdir(parents=True,exist_ok=True);prepare_files(root)
    os.execv(sys.executable,[sys.executable,'/opt/h3-still/runtime/service_launcher.py'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','backend','verify-metadata','download'])
    args=parser.parse_args()
    if args.action=='launch':launch()
    elif args.action=='backend':backend()
    elif args.action=='verify-metadata':verify_metadata()
    else:
        root=checked_root();root.mkdir(parents=True,exist_ok=True);prepare_files(root);download(root)
