"""Seed additive workflow folders; preserve the authenticated gateway and file browser."""
import os
from pathlib import Path
import sys
P=Path(__file__).resolve().parent

def main():
    root=Path(os.environ.get('H3_STILL_DATA_ROOT','/workspace/h3-still')).resolve()
    if root==Path('/workspace') or not root.is_relative_to(Path('/workspace')):raise ValueError('Invalid persistent root.')
    for source_dir,name in [(P/'workflows','00_H3_Video_Still_Studio'),
                            (Path('/opt/h3-still/lab/workflows'),'00_H3_Studio_Lab_v1')]:
        target=root/'user/default/workflows'/name;target.mkdir(parents=True,exist_ok=True)
        for source in sorted(source_dir.glob('*.json')):
            dest=target/source.name
            if dest.is_symlink():raise RuntimeError('Refusing a symlinked workflow: '+str(dest))
            try:
                with dest.open('xb') as out:out.write(source.read_bytes())
            except FileExistsError:pass
    for name in ('checkpoints','refmods','loras'):(root/'models'/name).mkdir(parents=True,exist_ok=True)
    os.execv(sys.executable,[sys.executable,'/opt/h3-still/runtime/service_launcher.py'])
if __name__=='__main__':main()
