#!/usr/bin/env python3
"""Build-time real ComfyUI CPU import/schema check, without model downloads or inference."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
from validate_live import validate


def main():
    with tempfile.TemporaryDirectory(prefix='moss-cpu-') as folder:
        log=Path(folder)/'comfy.log'
        with log.open('w') as output:
            proc=subprocess.Popen([sys.executable,'/opt/ComfyUI/main.py','--cpu','--listen','127.0.0.1',
                                   '--port','8199','--disable-auto-launch','--user-directory',folder],
                                  cwd='/opt/ComfyUI',stdout=output,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+240
                while time.monotonic()<deadline:
                    if proc.poll() is not None:raise RuntimeError('ComfyUI exited during CPU startup.')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8199/object_info',timeout=5) as r:
                            info=json.load(r)
                        break
                    except (OSError,ValueError):time.sleep(2)
                else:raise RuntimeError('CPU schema startup timed out.')
                errors=validate(info,check_models=False)
                if errors:raise RuntimeError('\n'.join(errors))
                import importlib.metadata
                assert importlib.metadata.version('comfyui-frontend-package')=='1.57.4'
                print('PASS: real ComfyUI CPU startup, required node imports and selectors, frontend pin.')
                print('NOT TESTED: model loading, CUDA inference, image quality, mobile browser rendering.')
            except Exception:
                output.flush()
                print(log.read_text(),file=sys.stderr)
                raise
            finally:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':main()
