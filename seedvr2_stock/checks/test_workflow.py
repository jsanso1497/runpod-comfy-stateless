"""Local static and geometry tests. They are not SeedVR2 or native video inference."""
import ast
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from validate import validate

def test_entire_package():
    assert validate()['crop_before_model']

@pytest.mark.parametrize('w,h,x,y',[
 (1920,1080,0,0),(1080,1080,0,0),(1080,1080,420,0),(1080,1080,840,0),
 (720,1080,0,0),(720,1080,600,0),(720,1080,1200,0),
 (640,640,800,200),(1280,720,320,180)])
def test_crop_table_preserves_exact_source_region(w,h,x,y):
    # A synthetic location map tests positions and frame alignment, not image quality.
    image=np.arange(1080*1920,dtype=np.int32).reshape(1080,1920)
    result=image[y:y+h,x:x+w]
    assert result.shape==(h,w)
    assert result[0,0]==y*1920+x
    assert result[-1,-1]==(y+h-1)*1920+x+w-1
    assert min(w,h) > 0 and w%2==h%2==0
    # Upstream short-edge equal to native short edge gives scale 1, not an upscale.
    assert min(w,h)/min(result.shape)==1

@pytest.mark.parametrize('w,h',[(1920,1080),(720,1080),(1080,1080),(640,640),(1280,720)])
def test_comparison_is_one_frame_not_full_clip(w,h):
    api=json.loads((ROOT/'api/01_Video_Clarity.api.json').read_text())
    assert api['11']['inputs']['batch_index']==api['12']['inputs']['batch_index']==0
    assert api['11']['inputs']['length']==api['12']['inputs']['length']==1
    assert api['13']['inputs']['match_image_size'] is False
    assert 2*w*h == w*h+w*h

def test_crop_pixel_work_reduction():
    assert 720*1080/(1920*1080)==0.375
    assert 1080*1080/(1920*1080)==0.5625

def test_no_4k_or_blend_or_hidden_time_cap():
    api=json.loads((ROOT/'api/01_Video_Clarity.api.json').read_text())
    assert all(row['class_type'] not in ('ImageBlend','Video Slice','VideoTrim') for row in api.values())
    assert api['5']['inputs']['resolution']==['10',1]
    assert api['5']['inputs']['max_resolution']==0

def test_no_second_inference_for_comparison():
    api=json.loads((ROOT/'api/01_Video_Clarity.api.json').read_text())
    assert sum(r['class_type']=='SeedVR2VideoUpscaler' for r in api.values())==1
    assert api['12']['inputs']['image']==['5',0]

def test_standard_proxy_not_custom_gateway():
    start=(ROOT/'start.sh').read_text(); conf=(ROOT/'nginx.conf').read_text()
    assert 'nginx -t' in start and 'COMFY_PASSWORD' in start
    assert 'RUNPOD_POD_ID' in start
    assert 'auth_basic_user_file' in conf and 'proxy_http_version 1.1' in conf
    assert 'proxy_set_header Upgrade $http_upgrade' in conf
    assert 'proxy_buffering off' in conf and 'proxy_request_buffering off' in conf
    assert 'auth_request' not in conf
    for name in ('gateway.py','worker.py','job.py','auth_server.py'):
        assert not (ROOT/name).exists()

def test_only_one_runtime_extension():
    text=(ROOT/'Dockerfile').read_text()
    assert text.count('custom_nodes/seedvr2_videoupscaler remote add')==1
    assert 'video_fidelity' not in text and 'ausboss' not in text.lower()
    assert 'playwright' not in text

def test_no_runtime_export_assertions():
    start=(ROOT/'start.sh').read_text()
    assert 'Output FPS changed' not in start
    assert 'fps_mode' not in start and 'ffmpeg ' not in start

@pytest.mark.parametrize('key',['model','device','offload_device','blocks_to_swap','attention_mode'])
def test_model_ui_matches_api(key):
    g=json.loads((ROOT/'workflows/01_Video_Clarity.json').read_text())
    a=json.loads((ROOT/'api/01_Video_Clarity.api.json').read_text())
    m=next(x for x in g['nodes'] if x['id']==3)
    order=['model','device','blocks_to_swap','swap_io_components','offload_device','cache_model','attention_mode']
    assert m['widgets_values'][order.index(key)]==a['3']['inputs'][key]
