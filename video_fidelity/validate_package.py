"""Offline structural checks; no GPU or model downloads."""
from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent


def validate():
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    graph=json.loads((ROOT/'workflows/01_Faithful_Video.json').read_text())
    assert len(graph['nodes'])==2
    assert [x['type'] for x in graph['nodes']]==['LoadVideo','FaithfulVideoRestore']
    assert graph['links']==[[1,1,0,2,0,'VIDEO']]
    assert graph['nodes'][1]['widgets_values']==['Original size',0.5,0.0,3.0,9,False]
    assert graph['extra']['faithful_video']['gpu_tested'] is False
    api=json.loads((ROOT/'api/01_Faithful_Video.api.json').read_text())
    assert api['2']['inputs']['video']==['1',0]
    assert api['2']['inputs']['restoration_strength']==0.5
    assert api['2']['inputs']['temporal_batch']==9
    locks=json.loads((ROOT/'models.lock.json').read_text())
    assert len(locks['models'])==2
    assert {x['filename'] for x in locks['models']}=={'seedvr2_ema_7b_fp16.safetensors','ema_vae_fp16.safetensors'}
    for row in locks['models']:
        assert re.fullmatch('[0-9a-f]{40}',row['revision'])
        assert re.fullmatch('[0-9a-f]{64}',row['sha256'])
    sources=json.loads((ROOT/'sources.lock.json').read_text())
    for name in ('comfy_commit','seedvr_commit'):
        assert sources[name] in (ROOT/'Dockerfile').read_text()
    node=(ROOT/'nodes/__init__.py').read_text()
    assert 'get_components(' not in node
    assert 'start_new_session=True' in node and 'group=True' in node
    for path in ROOT.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.keyword) and node.arg == 'shell':
                assert not (isinstance(node.value, ast.Constant) and node.value.value is True)
    print('Offline graph, API, two-weight manifest, source-pin and lifecycle contracts passed.')

if __name__=='__main__':validate()
