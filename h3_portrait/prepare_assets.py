#!/usr/bin/env python3
"""Download H3 base weights plus ALL entries in the shared link-only LoRA list."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import catalog
ROOT=Path('/workspace/h3-portrait/config')
HOME=Path('/workspace/ComfyUI')

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    resolved=ROOT/'resolved'
    resolved.mkdir(exist_ok=True)
    for filename in ('models.json','runtime.json'):
        shutil.copy2(ROOT/filename,resolved/filename)
    # Never consume the obsolete per-portrait numeric-ID catalog after a folder merge.
    (resolved/'loras.json').write_text('[]\n')
    profiles={'h3'}
    catalog.run_downloads(resolved,HOME,profiles)
    subprocess.run([sys.executable,'/opt/shared-loras/sync.py',
        '--links',str(ROOT/'lora_links.txt'),'--comfy-home',str(HOME)],check=True)
    print('H3 PORTRAIT ASSETS READY',flush=True)

if __name__=='__main__':
    main()
