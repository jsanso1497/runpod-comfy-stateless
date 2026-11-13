"""Assemble a fresh image from immutable core plus the base image's known helpers."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
P=Path(__file__).resolve().parent

def main():
    manifest=json.loads((P/'manifest.json').read_text())
    target=Path('/opt/ComfyUI');fresh=Path('/tmp/h3-a40-core')
    if fresh.exists():raise RuntimeError('Unexpected build directory exists.')
    subprocess.run(['git','init','-q',str(fresh)],check=True)
    def git(*args):return subprocess.check_output(['git','-C',str(fresh),*args],text=True).strip()
    git('remote','add','origin','https://github.com/Comfy-Org/ComfyUI.git')
    git('fetch','--depth','1','origin',manifest['comfy_commit']);git('checkout','--detach','FETCH_HEAD')
    if git('rev-parse','HEAD')!=manifest['comfy_commit']:raise RuntimeError('Comfy source pin mismatch.')
    for name in ('ComfyUI-Fizgig-H3-Still','H3_Still_Studio'):
        source=target/'custom_nodes'/name
        if not source.is_dir():raise RuntimeError('Required base-image helper missing: '+name)
        shutil.copytree(source,fresh/'custom_nodes'/name)
    shutil.copytree(P/'nodepack',fresh/'custom_nodes/H3_A40_Identity')
    shutil.rmtree(fresh/'.git')
    shutil.rmtree(target);shutil.move(str(fresh),str(target))
    subprocess.run([sys.executable,'-m','pip','install','-c','/opt/h3-still/constraints.txt',
                    '-r',str(target/'requirements.txt')],check=True)
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
    old=Path('/opt/h3-still/runtime/media_backend.py')
    saved=old.with_name('media_backend_before_a40.py')
    if saved.exists():raise RuntimeError('Base backend was already patched.')
    shutil.copy2(old,saved)
    old.write_text("import runpy\nimport sys\nsys.path.insert(0,'/opt/h3-still/a40')\n"
                   "sys.argv=[sys.argv[0],'backend']\nrunpy.run_path('/opt/h3-still/a40/runtime.py',run_name='__main__')\n")
    subprocess.run([sys.executable,str(P/'build_workflows.py')],check=True)
    (P/'build-provenance.json').write_text(json.dumps(dict(manifest=manifest,gpu_tested=False,
        auth_gateway_modified=False,experimental_native_video=False),indent=2)+'\n')
if __name__=='__main__':main()
