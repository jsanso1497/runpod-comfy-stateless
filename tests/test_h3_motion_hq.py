"""CPU/offline contracts only. These tests cannot certify motion adherence or GPU VRAM."""
from __future__ import annotations
import ast
import copy
from graphlib import TopologicalSorter
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from workbench import h3_motion_tools as tools
from workbench import h3_motion_contract as contract


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

builder = module('h3_motion_builder_test', ROOT/'scripts/build_h3_motion_hq.py')
nodes = module('h3_motion_nodes_test', ROOT/'src/nodes/h3_motion_hq/__init__.py')


def keypoints(count, valid=True):
    return [{'people':[{'pose_keypoints_2d':[20,20,1. if valid else 0.]*18}]} for _ in range(count)]


def box(x=10,y=10,w=100,h=200):
    return {'x':x,'y':y,'width':w,'height':h,'label':'person','score':.95}


class MotionMathTests(unittest.TestCase):
    def test_frame_grid_never_extends_source(self):
        for n in range(124,600):
            out=tools.aligned_length(n)
            self.assertLessEqual(out,n)
            self.assertLessEqual(out,362)
            self.assertEqual(out%17,5)
        self.assertEqual(tools.aligned_length(126),124)
        self.assertEqual(tools.aligned_length(360),345)
        self.assertEqual(tools.aligned_length(362),362)

    def test_short_clip_rejected(self):
        for n in (0,5,123,-1,True,124.5):
            with self.assertRaises(ValueError):tools.aligned_length(n)

    def test_canvas_respects_source_orientation(self):
        for w,h in [(1920,1080),(1080,1920),(1000,1000),(2000,1000),(1024,1536)]:
            ow,oh=tools.canvas(w,h)
            self.assertEqual(ow%32,0);self.assertEqual(oh%32,0)
            self.assertLessEqual(max(ow,oh),2048)
            self.assertLess(abs(ow/oh-w/h)/(w/h),.05)

    def test_invalid_geometry_and_nonfinite_values(self):
        for args in [(0,100), (1,100), (10000,100)]:
            with self.assertRaises(ValueError):tools.canvas(*args)
        for value in [float('nan'),float('inf'),True,'1',-.5,3]:
            with self.assertRaises(ValueError):tools.canvas(1280,720,value)

    def test_pixel_aspect_and_rotation(self):
        self.assertEqual(tools.source_geometry({'width':640,'height':480,'sample_aspect_ratio':'2:1'}),(1280,480))
        self.assertEqual(tools.source_geometry({'width':640,'height':480,'side_data_list':[{'rotation':90}]}),(480,640))
        with self.assertRaises(ValueError):tools.source_geometry({'width':640,'height':480,'color_transfer':'smpte2084'})

    def test_alignment_requires_identical_timeline_and_dimensions(self):
        self.assertEqual(tools.check_alignment((124,768,1344,3),(124,768,1344,3)),(1344,768,124))
        for shape in [(123,768,1344,3),(124,736,1344,3),(124,768,1330,3)]:
            with self.assertRaises(ValueError):tools.check_alignment((124,768,1344,3),shape)
        with self.assertRaises(ValueError):tools.check_alignment((125,768,1344,3),(125,768,1344,3))

    def test_native_pose_health_check(self):
        self.assertEqual(tools.pose_health(keypoints(124),124)['weak_pose_frames'],[])
        rows=keypoints(124);rows[5]=keypoints(1,False)[0]
        self.assertEqual(tools.pose_health(rows,124)['weak_pose_frames'],[5])
        with self.assertRaises(ValueError):tools.pose_health(keypoints(124,False),124)
        with self.assertRaises(ValueError):tools.pose_health(keypoints(123),124)

    def test_multiple_unselected_people_rejected(self):
        rows=keypoints(124);rows[0]['people']*=2
        with self.assertRaises(ValueError):tools.pose_health(rows,124)

    def test_controlnet_v2_only(self):
        obj=types.SimpleNamespace(model=types.SimpleNamespace(injection_layers=tuple(range(0,50,5))))
        self.assertIs(tools.check_control_patch(obj),obj)
        obj.model.injection_layers=tuple(range(0,50,10))
        with self.assertRaises(ValueError):tools.check_control_patch(obj)


class TrackingTests(unittest.TestCase):
    def test_largest_person_first_then_track_not_largest_each_frame(self):
        frames=[[box(20,20,100,200),box(400,20,50,100)],
                [box(24,20,100,200),box(390,20,170,300)]]
        tracked,report=tools.select_subject(frames,640,480,2)
        self.assertLess(tracked[1][0]['x'],100)
        self.assertEqual(report['multi_person_frames'],2)

    def test_explicit_first_person_left_to_right(self):
        rows=[[box(400,10,100,200),box(10,10,50,100)]]
        out,_=tools.select_subject(rows,640,480,1,0)
        self.assertLess(out[0][0]['x'],100)
        out,_=tools.select_subject(rows,640,480,1,1)
        self.assertGreater(out[0][0]['x'],300)

    def test_short_gap_holds_crop_not_video_frames(self):
        out,report=tools.select_subject([[box()],[],[],[box(15)]],640,480,4,max_gap=2)
        self.assertEqual(len(out),4);self.assertEqual(report['held_crop_frames'],2)
        with self.assertRaises(ValueError):tools.select_subject([[box()],[],[],[]],640,480,4,max_gap=2)

    def test_lost_track_does_not_jump_to_remote_person(self):
        with self.assertRaises(ValueError):tools.select_subject([[box()], [box(500,350,60,70)]],640,480,2,max_gap=0)

    def test_missing_first_frame_and_invalid_index(self):
        for frames,index in [([[]],-1),([[box()]],2),([[box()]],True)]:
            with self.assertRaises(ValueError):tools.select_subject(frames,640,480,1,index)

    def test_bad_input_or_frame_count(self):
        with self.assertRaises(ValueError):tools.select_subject([[box()]],640,480,2)
        with self.assertRaises(ValueError):tools.select_subject([[{'x':float('nan'),'y':1,'width':10,'height':10}]],640,480,1)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.graph,self.api=builder.build().finish(False)

    def test_reciprocal_links_types_and_no_cycles(self):
        for preview in (False,True):
            graph,api=builder.build().finish(preview)
            by={n['id']:n for n in graph['nodes']}
            self.assertEqual(len(by),len(graph['nodes']))
            deps={i:set() for i in by}
            for lid,src,out,dst,inp,typ in graph['links']:
                self.assertIn(lid,by[src]['outputs'][out]['links'])
                self.assertEqual(lid,by[dst]['inputs'][inp]['link'])
                self.assertEqual(typ,by[src]['outputs'][out]['type'])
                self.assertEqual(typ,by[dst]['inputs'][inp]['type'])
                self.assertEqual(api[str(dst)]['inputs'][by[dst]['inputs'][inp]['name']],[str(src),out])
                deps[dst].add(src)
            self.assertEqual(len(list(TopologicalSorter(deps).static_order())),len(by))

    def test_sampler_uses_control_patched_model(self):
        self.assertEqual(self.api['20']['inputs']['model'],['19',0])
        self.assertEqual(self.api['21']['inputs']['model'],['19',0])
        self.assertEqual(self.api['19']['inputs']['control_video'],['11',1])
        self.assertEqual(self.api['19']['inputs']['model_patch'],['16',0])
        self.assertEqual(self.api['16']['inputs']['model_patch'],['15',0])

    def test_single_source_drives_reference_control_and_output_length(self):
        self.assertEqual(self.api['17']['inputs']['length'],['11',4])
        self.assertEqual(self.api['17']['inputs']['ref_videos.ref_video_0'],['11',0])
        self.assertEqual(self.api['9']['inputs']['image'],['4',0])
        self.assertEqual(self.api['11']['inputs']['source_frames'],['4',0])
        self.assertEqual(self.api['11']['inputs']['pose_frames'],['10',0])

    def test_large_loaders_are_after_pose_gate(self):
        for ident,key,slot in [('12','unet_name',5),('13','clip_name',6),('14','vae_name',7),('15','name',8)]:
            self.assertEqual(self.api[ident]['inputs'][key],['11',slot])

    def test_no_ollama_refpack_turbo_or_quantized_weights(self):
        for row in self.api.values():
            self.assertNotIn('Ollama',row['class_type'])
            self.assertNotIn('ReferencePack',row['class_type'])
            self.assertNotIn('Lora',row['class_type'])
        for node in self.graph['nodes']:
            for value in node.get('widgets_values',[]):
                if isinstance(value,str) and value.endswith('.safetensors'):
                    self.assertFalse(any(q in value for q in ('int8','fp8','nvfp4','gguf')))

    def test_no_source_face_landmarks_default(self):
        self.assertFalse(self.api['10']['inputs']['draw_face'])
        self.assertTrue(self.api['10']['inputs']['draw_hands'])
        self.assertEqual(self.api['21']['inputs']['steps'],40)
        self.assertEqual(self.api['19']['inputs']['strength'],1.0)
        self.assertEqual(self.api['19']['inputs']['end_percent'],1.0)

    def test_preview_has_no_h3_diffusion_or_encoder(self):
        graph,api=builder.build().finish(True)
        typeset={v['class_type'] for v in api.values()}
        self.assertNotIn('CLIPLoader',typeset)
        self.assertNotIn('SamplerCustomAdvanced',typeset)
        self.assertNotIn('ModelPatchLoader',typeset)
        self.assertNotIn('MiniMaxH3ReferenceToVideo',typeset)
        self.assertNotIn('LoadImage',typeset)
        loaders=[v for v in api.values() if v['class_type']=='UNETLoader']
        self.assertEqual(len(loaders),1)
        self.assertEqual(loaders[0]['inputs']['unet_name'],'rt_detr_v4-x-hgnet_fp16.safetensors')

    def test_saved_graphs_and_api_match_builder(self):
        for preview,stem in [(True,'00_H3_Motion_Preview'),(False,'01_H3_Character_Swap_Pose_Control')]:
            graph,api=builder.build().finish(preview)
            self.assertEqual(graph,json.loads((ROOT/'workflows/h3/HMOTION_Pose_Control'/(stem+'.json')).read_text()))
            self.assertEqual(api,json.loads((ROOT/'automation/HMOTION'/(stem+'.api.json')).read_text()))

    def test_payload_python_syntax(self):
        for folder in ('src/nodes/h3_motion_hq','src/workbench'):
            for p in (ROOT/folder).glob('*motion*.py'):
                ast.parse(p.read_text())
        ast.parse((ROOT/'src/nodes/h3_motion_hq/__init__.py').read_text())

    def test_model_manifest_precision_and_pins(self):
        data=json.loads((ROOT/'catalog/h3-motion-update.json').read_text())
        self.assertEqual(len(data['assets']),3)
        self.assertEqual(sum(a['estimated_bytes'] for a in data['assets']),10422903562)
        for asset in data['assets']:
            self.assertRegex(asset['revision'],r'^[0-9a-f]{40}$')
            self.assertRegex(asset['sha256'],r'^[0-9a-f]{64}$')
            self.assertEqual(asset['group'],'h3_motion')
        self.assertFalse(data['tasks'][0]['approved'])


class RuntimeContractTests(unittest.TestCase):
    def test_live_schema_catches_missing_assets(self):
        report=contract.inspect_schema({})
        self.assertFalse(report['ok'])
        self.assertIn('MiniMaxH3FunControlNetApply',report['missing_nodes'])
        self.assertTrue(any('ControlNet' in x or 'controlnet' in x for x in report['problems']))

    def test_core_pin_guard(self):
        with self.assertRaises(ValueError):contract.verify_core(Path('/missing'),'0'*40)

    def test_node_output_contract(self):
        self.assertEqual(len(nodes.WBH3MotionReady.RETURN_TYPES),9)
        self.assertEqual(nodes.WBH3MotionTrackPerson.RETURN_TYPES[0],'BOUNDING_BOX')
        self.assertEqual(nodes.WBH3MotionReady.INPUT_TYPES()['required']['mode'][0],['Render video','Preview motion only'])

    def test_preview_gate_blocks_before_gpu_import(self):
        class ShapeOnly:
            shape=(124,768,1344,3)
        class Blocker:
            def __init__(self,value):self.value=value
        pkg=types.ModuleType('comfy_execution')
        sub=types.ModuleType('comfy_execution.graph_utils');sub.ExecutionBlocker=Blocker
        with patch.dict(sys.modules,{'comfy_execution':pkg,'comfy_execution.graph_utils':sub}):
            out=nodes.WBH3MotionReady().ready(ShapeOnly(),ShapeOnly(),keypoints(124),'Preview motion only')
        self.assertEqual(len(out),9)
        self.assertTrue(all(isinstance(x,Blocker) for x in out))

    def test_render_gate_derives_shape_and_releases_preprocessors(self):
        class ShapeOnly:
            shape=(124,768,1344,3)
        actions=[]
        pkg=types.ModuleType('comfy');sub=types.ModuleType('comfy.model_management')
        sub.throw_exception_if_processing_interrupted=lambda:actions.append('check')
        sub.unload_all_models=lambda:actions.append('unload')
        sub.soft_empty_cache=lambda:actions.append('cache')
        pkg.model_management=sub
        with patch.dict(sys.modules,{'comfy':pkg,'comfy.model_management':sub}):
            out=nodes.WBH3MotionReady().ready(ShapeOnly(),ShapeOnly(),keypoints(124),'Render video')
        self.assertEqual(out[2:5],(1344,768,124))
        self.assertEqual(actions,['check','unload','cache'])
        self.assertIn('2.0_pruned_bf16',out[-1])

    def test_local_video_path_safety(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'input';root.mkdir()
            good=root/'test.mp4';good.write_bytes(b'x')
            vid=types.SimpleNamespace(get_stream_source=lambda:str(good))
            self.assertEqual(tools.resolve_video(vid,root)[0],good)
            outside=Path(d)/'outside.mp4';outside.write_bytes(b'x')
            for value in (outside,'https://example.com/video.mp4'):
                bad=types.SimpleNamespace(get_stream_source=lambda v=value:v)
                with self.assertRaises(ValueError):tools.resolve_video(bad,root)
            (root/'link.mp4').symlink_to(outside)
            bad=types.SimpleNamespace(get_stream_source=lambda:root/'link.mp4')
            with self.assertRaises(ValueError):tools.resolve_video(bad,root)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg not installed')
class FFmpegIntegrationTests(unittest.TestCase):
    def test_actual_60fps_clip_becomes_aligned_24fps_without_tail_padding(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','testsrc2=size=96x128:rate=60',
                            '-t','5.3','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(src)],check=True)
            frames,report=tools.decode_source(src,0,5.25,.5)
            self.assertEqual(len(frames),124)
            self.assertEqual(report['fps'],24)
            self.assertAlmostEqual(report['output_seconds'],124/24)
            self.assertLess(report['canvas'][0],report['canvas'][1])
            self.assertEqual(frames.dtype,torch.float32)
            self.assertTrue(torch.isfinite(frames).all())
            self.assertGreater(float((frames[10]-frames[100]).abs().mean()),.01)
            del frames
            with self.assertRaises(ValueError):tools.decode_source(src,5.0,5.25,.5)

    def test_conversion_preserves_elapsed_time_not_just_frame_count(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'ramp.mkv'
            frames=np.zeros((360,64,64,3),dtype=np.uint8)
            for i in range(360):frames[i]=int(i*200/359)
            subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24',
                '-video_size','64x64','-framerate','60','-i','pipe:0','-an','-c:v','ffv1',
                '-threads','2','-pix_fmt','gbrp',str(src)],input=frames.tobytes(),check=True)
            decoded,report=tools.decode_source(src,0,5.25,.5)
            h,w=decoded.shape[1:3]
            last=float(decoded[-1,h//2,w//2,0])*255
            # Frame 123 at 24 fps represents ~5.125 sec, not frame 123 at 60 fps.
            self.assertLessEqual(abs(last-int((123/24)*60*200/359)),2)
            self.assertEqual(report['frames'],124)
            self.assertLess(float(decoded[0,h//2,w//2,0])*255,2)

    def test_comparison_is_frame_aligned(self):
        import torch
        source=torch.zeros((5,64,96,3))
        other=torch.ones_like(source)
        paired=tools.paired_preview(source,other)
        self.assertEqual(paired.shape,(5,88,192,3))
        self.assertEqual(float(paired[:,24:,:96].sum()),0)
        self.assertEqual(float(paired[:,24:,96:].min()),1)
        with self.assertRaises(ValueError):tools.paired_preview(source,other[:4])

if __name__=='__main__':unittest.main()
