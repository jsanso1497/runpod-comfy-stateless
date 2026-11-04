"""Build-time CPU startup/schema check. Does not load weights or claim GPU validation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from .video_process import terminate_process


def main():
    root = Path('/opt/ComfyUI')
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        for x in ('input', 'output', 'user'):
            (d/x).mkdir()
        cmd = [sys.executable, str(root/'main.py'), '--cpu', '--listen', '127.0.0.1', '--port', '8197',
               '--input-directory', str(d/'input'), '--output-directory', str(d/'output'),
               '--user-directory', str(d/'user'), '--disable-auto-launch', '--cache-none',
               '--disable-comfy-compiler', '--disable-partner-nodes']
        env = dict(os.environ, WB_DATA_ROOT=str(d))
        with (d/'smoke.log').open('w') as log:
            proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                schema = None
                for _ in range(180):
                    if proc.poll() is not None:
                        break
                    try:
                        with urlopen('http://127.0.0.1:8197/object_info', timeout=2) as response:
                            schema = json.load(response)
                        break
                    except OSError:
                        time.sleep(1)
                if schema is None:
                    raise RuntimeError((d/'smoke.log').read_text()[-20000:])
                required = {'LoadVideo', 'WBSeedVR2StreamVideo', 'UNETLoader', 'VAELoader', 'KSampler'}
                missing = required - set(schema)
                if missing:
                    raise RuntimeError('Missing node(s): ' + ', '.join(sorted(missing)) + '\n' + (d/'smoke.log').read_text()[-16000:])
                entry = schema['WBSeedVR2StreamVideo']
                inputs = entry['input']['required']
                assert set(inputs) == {'video','mode','model','start_seconds','duration_seconds','seed','crf','keep_intermediates'}
                assert entry['output'] == ['VIDEO', 'STRING'] and entry['output_node']
                for path in Path('/opt/workbench/workflows/restoration/RVID_Streaming').glob('*.json'):
                    graph = json.loads(path.read_text())
                    assert not any(n['type'].startswith('AUSBOSS') for n in graph['nodes'])
                    assert all(n['type'] in schema or n['type'] in ('Note','MarkdownNote') for n in graph['nodes'])
                from .config import read_catalog, requested_groups
                model = next(x for x in read_catalog('models.json')['models'] if x['id']=='qwen')
                assert 'seedvr_stream' in model['local_nodes']
                assert {'qwen','seedvr2','seedvr2_sharp'}.issubset(requested_groups())
                assert not {'ollama','qwen_pe_i2i','qwen_stock_pe','moss_image'} & set(requested_groups())
                print('PASS: ComfyUI CPU startup + streaming schema + Qwen/video catalog. GPU not tested.')
            finally:
                terminate_process(proc)

if __name__ == '__main__':
    main()
