#!/usr/bin/env python3
"""Populate the real ComfyUI user workflow directory and keep PHOTO original startup."""
from __future__ import annotations
import argparse,os,sys,shutil
from pathlib import Path

NATIVE=Path('/opt/qwen21_native')
PHOTO=Path('/opt/qwen21_photo_edit')
PHOTO_WORKFLOWS=('Qwen21_Photo_2K_SAM3_BF16.json','Qwen21_Photo_2K_SAM3_INT8.json')
AUSBOSS_STAGE=Path('/opt/ComfyUI/custom_nodes/ComfyUI-AusBoss')
REQUIRED_AUSBOSS='nodes/node_inpaint_crop_stitch.py'


def install_photo_workflows(user:Path):
    """Copy only the two reviewed editable UI workflows, never overwrite user edits."""
    found={}
    for name in PHOTO_WORKFLOWS:
        candidates=[p for p in PHOTO.rglob(name) if 'api' not in {x.lower() for x in p.parts}]
        if len(candidates)!=1:
            raise RuntimeError(f'Expected one bundled Photo 2K/SAM3 UI workflow {name}, found {len(candidates)}. Verify the Qwen Photo package.')
        found[name]=candidates[0]
    base=user/'default/workflows'
    dest=base/'Qwen21 Photo 2K SAM3'
    for name,src in found.items():
        if any(p.is_file() for p in base.rglob(name)):
            continue  # User may have customized an existing workflow. Never overwrite it.
        dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest/name)
    for name in PHOTO_WORKFLOWS:
        if not any(p.is_file() for p in base.rglob(name)):
            raise RuntimeError('Photo workflow missing from actual ComfyUI user directory: '+name)
    return 2


def ensure_ausboss_for_active_comfy(comfy: Path, staged: Path | None = None) -> Path:
    """Make the build-pinned AusBoss checkout visible to the *active* ComfyUI.

    Qwen Photo can launch /workspace/ComfyUI even though Docker installed
    custom nodes under /opt/ComfyUI. Link only our separately pinned package;
    never replace a pre-existing third-party checkout or edit core files.
    """
    comfy = Path(comfy).resolve()
    staged = Path(staged) if staged is not None else AUSBOSS_STAGE
    staged = staged.resolve()
    destination = comfy / 'custom_nodes' / 'ComfyUI-AusBoss'
    marker = destination / REQUIRED_AUSBOSS
    if marker.is_file() and 'AUSBOSS_NODES_StitchInpaint' in marker.read_text(encoding='utf-8'):
        return destination
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f'Existing AusBoss directory is incomplete/incompatible: {destination}. Refusing to overwrite user files.')
    source_marker = staged / REQUIRED_AUSBOSS
    if not source_marker.is_file() or 'AUSBOSS_NODES_StitchInpaint' not in source_marker.read_text(encoding='utf-8'):
        raise RuntimeError(f'Pinned AusBoss checkout is missing or incomplete: {staged}. Check the unified Docker build layer.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.symlink_to(staged, target_is_directory=True)
    if not marker.is_file():
        destination.unlink()
        raise RuntimeError(f'Unable to expose AusBoss at active ComfyUI: {destination}')
    print(f'Linked pinned AusBoss to ACTIVE ComfyUI: {destination} -> {staged}', flush=True)
    return destination


def install_for(comfy:Path,user:Path,require_photo:bool=False):
    ensure_ausboss_for_active_comfy(comfy)
    sys.path.insert(0,str(NATIVE/'scripts'))
    from install_into_comfy import install as native_install
    import importlib.util
    p=NATIVE/'addons/torso_lock/INSTALL.py'
    spec=importlib.util.spec_from_file_location('qwen21_torso_integration',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    native_install(comfy,user,'identity',False)
    mod.install(comfy,user)
    if require_photo:install_photo_workflows(user)
    return 40 if require_photo else 38


def resolve_candidates() -> list[tuple[Path,Path]]:
    candidates=[]
    explicit_comfy=os.environ.get('UNIFIED_COMFY_DIR')
    explicit_user=os.environ.get('UNIFIED_USER_DIR')
    dirs=[Path(explicit_comfy)] if explicit_comfy else [Path('/opt/ComfyUI'),Path('/workspace/ComfyUI')]
    for comfy in dirs:
        if not (comfy/'comfy_extras/nodes_qwen.py').is_file():continue
        users=[Path(explicit_user)] if explicit_user else [comfy/'user']
        if not explicit_user:
            data=os.getenv('DATA_ROOT')
            if data:users.append(Path(data)/'user')
            users.append(Path('/workspace/ComfyUI/user'))
        for user in users:
            pair=(comfy.resolve(),user.resolve())
            if pair not in candidates:candidates.append(pair)
    return candidates


def main():
    p=argparse.ArgumentParser();p.add_argument('--comfy-dir',type=Path);p.add_argument('--user-dir',type=Path)
    a=p.parse_args()
    pairs=[(a.comfy_dir,a.user_dir)] if a.comfy_dir and a.user_dir else resolve_candidates()
    if not pairs:raise RuntimeError('Cannot identify Qwen Photo ComfyUI. Set UNIFIED_COMFY_DIR and UNIFIED_USER_DIR.')
    for comfy,user in pairs:
        install_for(comfy,user,require_photo=True)
        print(f'Installed Photo (2) + Native (34 default) + Torso (4) workflow files for candidate user directory: {user}',flush=True)
    return len(pairs)

if __name__=='__main__':
    try:main()
    except Exception as e:print('UNIFIED STARTUP INSTALL FAILED: '+str(e),file=sys.stderr);sys.exit(1)
