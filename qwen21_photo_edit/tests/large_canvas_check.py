"""Explicit CPU geometry stress check. No inference weights are involved."""
from __future__ import annotations
import gc
import importlib.util
import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q21_large_geometry',ROOT/'node/geometry.py')
g=importlib.util.module_from_spec(spec);sys.modules[spec.name]=g;spec.loader.exec_module(g)
results=[]
for width,height in [(7001,4003),(4003,7001)]:
    source=torch.full((1,height,width,3),.231,dtype=torch.float32)
    # Include a one-pixel gradient to expose accidental canvas transforms.
    source[0,:,::2,0]=.319
    mask=torch.zeros((1,height,width));mask[:,600:3100,600:3100]=1.
    mask[:,1500:1532,1500:1532]=0.
    package,work,_=g.crop_photo(source,mask,128,2048,3072,False)
    output,report=g.stitch_photo(package,torch.full_like(work,.875),12,0.)
    if output.shape != source.shape:raise AssertionError('Canvas shape changed')
    for y in range(0,height,128):
        outside=mask[:,y:y+128]==0
        if not torch.equal(output[:,y:y+128][outside],source[:,y:y+128][outside]):
            raise AssertionError(f'Outside-mask pixels changed near row {y}')
    row=json.loads(report)
    row['test']='CPU synthetic 2500x2500 edit with protected 32x32 hole'
    results.append(row)
    print(json.dumps({'source':[width,height],'work':row['model_processing_size'],'outside_mask_changed_channels':0,'protected_hole':'unchanged'}),flush=True)
    del source,mask,package,work,output,report
    gc.collect()
(ROOT/'LARGE_CANVAS_VALIDATION.json').write_text(json.dumps(results,indent=2)+'\n')
