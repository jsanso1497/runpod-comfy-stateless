#!/usr/bin/env python3
"""Seed only new Lab workflows, then use the existing authenticated Studio launcher."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('H3_STILL_DATA_ROOT','/workspace/h3-still')).resolve()

def seed(root):
    if root==Path('/workspace') or not root.is_relative_to(Path('/workspace')):
        raise ValueError('H3_STILL_DATA_ROOT must be a subfolder of /workspace.')
    folder=root/'user/default/workflows/00_H3_Studio_Lab_v1'
    folder.mkdir(parents=True,exist_ok=True)
    for source in sorted((PACKAGE/'workflows').glob('*.json')):
        target=folder/source.name
        if target.exists():continue
        with target.open('xb') as stream:stream.write(source.read_bytes())
    for name in ('checkpoints','refmods'):(root/'models'/name).mkdir(parents=True,exist_ok=True)
    # Add-on failure does not take away the login or file browser.
    if (root/'lab/providers/spark/enabled.json').is_file():
        try:
            subprocess.run([sys.executable,str(PACKAGE/'manage.py'),'restore-spark'],check=True,timeout=180)
        except Exception as e:print('Optional Spark restore failed; baseline remains usable:',str(e),flush=True)
    print('Lab workflows seeded. Existing v1/v2 graphs and login services are unchanged.',flush=True)

def main():
    seed(ROOT)
    os.execv(sys.executable,[sys.executable,'/opt/h3-still/runtime/service_launcher.py'])
if __name__=='__main__':main()
