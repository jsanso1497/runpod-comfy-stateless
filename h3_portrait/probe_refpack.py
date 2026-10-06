#!/usr/bin/env python3
"""Build-only CPU probe of the actual patched dependency; no models or inference."""
import argparse
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser();p.add_argument('--comfy-home',type=Path,required=True);a=p.parse_args()
    home=a.comfy_home
    sys.path.insert(0,str(home))
    sys.path.insert(0,str(home/'custom_nodes/ComfyUI-MiniMaxRefPack'))
    # Avoid CUDA discovery if a future dependency begins importing Comfy model code.
    sys.argv=['probe_refpack.py','--cpu']
    from minimax_refpack.nodes import MiniMaxH3ReferencePack
    from minimax_refpack import local_runtime, prompt
    assert getattr(MiniMaxH3ReferencePack.build,'_h3_handoff_version',None)=='1.5.3'
    assert prompt._LOCAL_HTTP.trust_env is False
    with patch.object(local_runtime,'session',side_effect=AssertionError('A none-provider probe must not contact Ollama')):
        result=MiniMaxH3ReferencePack().build(direction='CPU reference passthrough probe',references_json='',prompt_provider='none')
    assert len(result)==20 and result[18]=='CPU reference passthrough probe'
    assert all(v is None for v in result[:18])
    schema=MiniMaxH3ReferencePack.INPUT_TYPES()
    assert schema['optional']['prompt_provider'][0]==['local','none']
    print('REFERENCE PACK CPU PROBE: actual patched class, local-only schema, none-provider passthrough and 20 sockets verified.',flush=True)


if __name__=='__main__':main()
