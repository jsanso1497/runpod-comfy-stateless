"""Prepare selected assets, then run private ComfyUI behind the existing gateway."""
import os
from pathlib import Path
import sys
sys.path.insert(0,'/opt/h3-still/runtime')
from bootstrap import prepare
from recovery_launcher import terminal_environment
from manage import download,confirmed


def main():
    root=Path(os.environ.get('H3_STILL_DATA_ROOT','/workspace/h3-still')).resolve()
    if root==Path('/workspace') or not root.is_relative_to(Path('/workspace')):raise ValueError('Data root must be under /workspace.')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. File access remains on port 8888.')
    prop=torch.cuda.get_device_properties(0)
    print('GPU:',prop.name,'VRAM GiB:',round(prop.total_memory/1024**3,2),flush=True)
    if prop.total_memory<70*1024**3:
        print('WARNING: This profile targets A100 80 GB. Smaller devices need reduced settings.',flush=True)
    confirmed()
    # Keep every inherited Studio v1/v2 model and workflow available, without replacing existing user graphs.
    prepare(root)
    # Inherited prepare_turbo is retained in the image under a different module name.
    import importlib.util
    spec=importlib.util.spec_from_file_location('h3_legacy_backend','/opt/h3-still/runtime/media_backend_legacy.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    legacy.prepare_turbo(root)
    preload=os.environ.get('H3V_PRELOAD','hq')
    if preload not in ('hq','all'):raise ValueError('H3V_PRELOAD must be hq or all.')
    download(root,preload)
    env=terminal_environment()
    command=[sys.executable,'/opt/ComfyUI/main.py','--listen','127.0.0.1','--port','8189',
        '--input-directory',str(root/'input'),'--output-directory',str(root/'output'),
        '--user-directory',str(root/'user'),'--temp-directory',str(root/'temp'),
        '--extra-model-paths-config',str(root/'extra_model_paths.yaml'),
        '--preview-method','none','--disable-auto-launch','--use-pytorch-cross-attention','--reserve-vram','4.0']
    # Do not use --highvram: BF16 encoder and diffusion model must be able to offload between stages.
    os.chdir('/opt/ComfyUI');os.execve(sys.executable,command,env)
if __name__=='__main__':main()
