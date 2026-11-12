"""Build-only assembly. Changes are confined to this new image, not the running Pod."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
P=Path(__file__).resolve().parent

def main():
    manifest=json.loads((P/'manifest.json').read_text())
    lab=Path('/opt/h3-still/lab')
    lab_manifest=lab/'manifest.json'
    value=json.loads(lab_manifest.read_text())
    value['ph3_comfy_commit']=manifest['comfy_commit']
    value['video_studio_build_note']='Modified only in the new H3 Video Still Studio container: pinned shared ComfyUI core.'
    lab_manifest.write_text(json.dumps(value,indent=2)+'\n')
    subprocess.run([sys.executable,str(lab/'assemble.py'),'--profile','ph3'],check=True)
    dest=Path('/opt/ComfyUI/custom_nodes/H3_Video_Still_Studio')
    shutil.copytree(P/'nodepack',dest)
    old=Path('/opt/h3-still/runtime/media_backend.py')
    saved=old.with_name('media_backend_legacy.py')
    if saved.exists():raise RuntimeError('Unexpected prior backend replacement.')
    shutil.copy2(old,saved)
    old.write_text('# Modified by H3 Video Still Studio; original saved as media_backend_legacy.py.\n'
                   'import runpy\nimport sys\n'
                   'sys.path.insert(0, "/opt/h3-still/video_studio")\n'
                   'runpy.run_path("/opt/h3-still/video_studio/backend.py", run_name="__main__")\n')
    subprocess.run([sys.executable,str(P/'build_workflows.py')],check=True)
    (P/'build-provenance.json').write_text(json.dumps(dict(manifest=manifest,
        gpu_tested=False,prompt_enhancer=False,auth_gateway_modified=False),indent=2)+'\n')
    subprocess.run([sys.executable,'-m','compileall','-q',str(P),str(dest)],check=True)
    subprocess.run([sys.executable,'-m','pip','check'],check=True)
if __name__=='__main__':main()
