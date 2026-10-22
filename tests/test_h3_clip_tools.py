"""CPU-only contracts and real FFmpeg round trips. No ComfyUI/GPU certification."""
from __future__ import annotations
import ast
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from workbench import h3_clip_tools as c


def load_builder():
    spec = importlib.util.spec_from_file_location('h3_clip_builder', ROOT/'scripts/build_h3_clip_workflows.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlanningTests(unittest.TestCase):
    def test_exact_duration(self):
        self.assertEqual(c.DURATION, Fraction(31, 6))
        self.assertEqual(c.AUDIO_SAMPLES, 248000)
        self.assertEqual(c.FRAMES % 17, 5)

    def test_three_minutes_complete_chunks(self):
        clips, report = c.plan_clips(180, mode=c.MODES[1])
        self.assertEqual(len(clips), 34)
        self.assertEqual(clips[-1].start, Fraction(341, 2))
        self.assertAlmostEqual(report['incomplete_tail_seconds'], 13/3)
        self.assertFalse(any(x.padding for x in clips))

    def test_no_rounding_drift_over_many_chunks(self):
        clips, _ = c.plan_clips(3600, mode=c.MODES[1])
        self.assertEqual(len(clips), 696)
        self.assertEqual(clips[-1].start + c.DURATION, 3596)
        for a, b in zip(clips, clips[1:]):
            self.assertEqual(b.start-a.start, c.DURATION)

    def test_specific_chunk_number(self):
        clips, _ = c.plan_clips(180, first_chunk=7)
        self.assertEqual(clips[0].number, 7)
        self.assertEqual(clips[0].start, 31)
        self.assertEqual(len(clips), 1)

    def test_arbitrary_start(self):
        clips, _ = c.plan_clips(180, start_seconds=90.25)
        self.assertEqual(clips[0].start, Fraction(361,4))

    def test_range(self):
        clips, _ = c.plan_clips(180, mode=c.MODES[1], first_chunk=7, chunk_count=4)
        self.assertEqual([x.number for x in clips], [7,8,9,10])
        self.assertEqual(clips[0].start, 31)

    def test_shorter_requested_count_clamps_at_end(self):
        clips, _ = c.plan_clips(180, mode=c.MODES[1], first_chunk=33, chunk_count=10)
        self.assertEqual(len(clips), 2)

    def test_tail_hold_is_explicit(self):
        clips, _ = c.plan_clips(180, mode=c.MODES[1], final_chunk=c.TAIL_POLICIES[1])
        self.assertEqual(len(clips), 35)
        self.assertTrue(clips[-1].padding)
        self.assertEqual(clips[-1].available, Fraction(13,3))

    def test_short_tail_default_fails_for_single(self):
        with self.assertRaisesRegex(ValueError, 'Less than one complete'):
            c.plan_clips(4)

    def test_exact_chunk_probe_rounding(self):
        for duration in [Fraction(31,6), '5.166667', '5.166666']:
            clips, _ = c.plan_clips(duration)
            self.assertEqual(len(clips), 1)
            self.assertFalse(clips[0].padding)

    def test_trim_respected(self):
        clips, _ = c.plan_clips(180, input_start=30, input_duration=20, start_seconds=5, mode=c.MODES[1])
        self.assertEqual(clips[0].start, 35)
        self.assertEqual(len(clips), 2)

    def test_past_end_rejected(self):
        for kwargs in [{'start_seconds':180}, {'first_chunk':100}, {'input_start':200}]:
            with self.assertRaises(ValueError):
                c.plan_clips(180, **kwargs)

    def test_invalid_parameters(self):
        for kwargs in [{'first_chunk':0},{'first_chunk':True},{'chunk_count':-1},{'mode':'typo'},
                       {'final_chunk':'unknown'},{'start_seconds':float('nan')}, {'start_seconds':float('inf')}]:
            with self.assertRaises(ValueError):
                c.plan_clips(180, **kwargs)

    def test_batch_safety_limit(self):
        with self.assertRaisesRegex(ValueError,'over 1000'):
            c.plan_clips(7200, mode=c.MODES[1])
        self.assertEqual(len(c.plan_clips(7200, mode=c.MODES[1],chunk_count=1000)[0]),1000)

    def test_upload_limit(self):
        self.assertEqual(c.upload_cli_args({}), ['--max-upload-size','1024'])
        self.assertEqual(c.upload_cli_args({'WB_H3_MAX_UPLOAD_MB':'1900'}), ['--max-upload-size','1900'])
        for val in ['0','2000','abc','10.5','-1','',None]:
            with self.assertRaises(ValueError):
                c.upload_cli_args({'WB_H3_MAX_UPLOAD_MB':val})


class SafetyTests(unittest.TestCase):
    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for val in ['../x','/tmp/x','a/../b','a//b','a\\b','a\x00b']:
                with self.assertRaises(ValueError):
                    c.safe_child(td,val)

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'outside').mkdir();(root/'alias').symlink_to(root/'outside')
            with self.assertRaises(ValueError):c.safe_child(root,'alias/file')

    def test_resolve_video(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'source.mp4';path.write_bytes(b'not decoded here')
            class Video:
                def get_stream_source(self):return str(path)
                def get_active_trim_window(self):return (4.5,10)
            resolved,start,duration=c.resolve_video(Video(),td)
            self.assertEqual(resolved,path);self.assertEqual(start,Fraction(9,2));self.assertEqual(duration,10)

    def test_remote_and_memory_videos_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for value in ['https://example.com/a.mp4', object()]:
                class Video:
                    def get_stream_source(self):return value
                with self.assertRaises(ValueError):c.resolve_video(Video(),td)
            with self.assertRaises(ValueError):c.resolve_video(object(),td)

    def test_copy_fallback_is_not_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'original.mp4';source.write_bytes(b'clip')
            with patch.object(c.os,'link',side_effect=OSError('cross-device')):
                name,method=c.publish_input(source,root,'new.mp4')
            self.assertEqual(method,'copy');self.assertFalse((root/name).is_symlink())
            self.assertEqual((root/name).read_bytes(),b'clip')
            with self.assertRaises(ValueError):c.publish_input(source,root,'new.mp4')

    def test_hdr_rejected(self):
        for transfer in ['smpte2084','arib-std-b67']:
            with self.assertRaisesRegex(ValueError,'HDR'):
                c.describe_source({'streams':[{'codec_type':'video','color_transfer':transfer}]})

    def test_plan_and_encode_do_not_use_stream_copy_or_rounded_duration(self):
        clip=c.plan_clips(180,first_chunk=7)[0][0]
        info={'video_index':0,'audio_index':1,'width':1920,'height':1080}
        cmd=c.export_command(Path('/input/a.mp4'),Path('/output/new.mp4'),info,clip,True)
        self.assertNotIn('copy',cmd);self.assertIn('libx264',cmd)
        self.assertEqual(cmd[cmd.index('-ss')+1],'31.000000000000')
        self.assertIn('trim=end_frame=124',cmd[cmd.index('-vf')+1])
        self.assertNotIn('tpad',cmd[cmd.index('-vf')+1])
        self.assertIn('248000',cmd[cmd.index('-af')+1])

    def test_cancelled_process_is_terminated(self):
        calls=0
        class Cancelled(Exception):pass
        def cancel():
            nonlocal calls
            calls+=1
            if calls>=3:raise Cancelled('test cancellation')
        with self.assertRaises(Cancelled):
            c.run_media([sys.executable,'-c','import time; time.sleep(10)'],check_cancel=cancel)

    def test_no_gpu_or_model_imports_in_media_engine(self):
        text=(ROOT/'src/workbench/h3_clip_tools.py').read_text()
        imports=[n for n in ast.walk(ast.parse(text)) if isinstance(n,(ast.Import,ast.ImportFrom))]
        names=[a.name for n in imports if isinstance(n,ast.Import) for a in n.names]
        names += [n.module for n in imports if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(name and name.startswith(('torch','comfy','numpy','ollama')) for name in names))


class GraphTests(unittest.TestCase):
    def test_graph_and_api_connections(self):
        builder=load_builder()
        for multiple in [False,True]:
            graph,api=builder.build(multiple)
            nodes={n['id']:n for n in graph['nodes']}
            for lid,src,slot,dst,target,kind in graph['links']:
                self.assertIn(lid,nodes[src]['outputs'][slot]['links'])
                self.assertEqual(nodes[dst]['inputs'][target]['link'],lid)
                self.assertEqual(api[str(dst)]['inputs'][nodes[dst]['inputs'][target]['name']],[str(src),slot])
            self.assertEqual(nodes[2]['widgets_values_named']['mode'],c.MODES[int(multiple)])
            self.assertEqual(set(api),{'1','2','3'})
            self.assertFalse(any('Loader' in n['type'] for n in nodes.values()))

    def test_graphs_are_reproducible(self):
        builder=load_builder()
        files=[('01_H3_Extract_Selected_124_Frame_Clip',False),('02_H3_Split_Long_Video_Into_124_Frame_Clips',True)]
        for name,multiple in files:
            graph,api=builder.build(multiple)
            self.assertEqual(json.loads((ROOT/'workflows/h3/HCLIPS_Video_Preparation'/f'{name}.json').read_text()),graph)
            self.assertEqual(json.loads((ROOT/'automation/HCLIPS'/f'{name}.api.json').read_text()),api)

    def test_ui_widget_order_matches_schema(self):
        tree=ast.parse((ROOT/'src/nodes/h3_clip_tools/__init__.py').read_text())
        schema=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Schema')
        inputs=next(k.value for k in schema.keywords if k.arg=='inputs')
        names=[call.args[0].value for call in inputs.elts]
        graph,_=load_builder().build()
        controls=list(graph['nodes'][1]['widgets_values_named'])
        self.assertEqual(names,['video']+controls)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required for real media tests')
class RealMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name);cls.inp=cls.root/'input';cls.inp.mkdir()
        cls.long=cls.inp/'three_minutes_60fps.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',
            '-f','lavfi','-i','testsrc2=size=96x64:rate=60:duration=180',
            '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=180',
            '-c:v','libx264','-preset','ultrafast','-crf','16','-threads','2',
            '-c:a','aac','-shortest',str(cls.long)],check=True)
        cls.original_size=cls.long.stat().st_size

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.out=self.root/('out_'+self._testMethodName);self.out.mkdir()

    def export(self,source=None,**kwargs):
        return c.export_clips(source or self.long,self.out,self.inp,**kwargs)

    def test_real_three_minute_full_batch(self):
        report=self.export(mode=c.MODES[1],make_zip=True)
        self.assertEqual(len(report['clips']),34)
        self.assertAlmostEqual(report['planning']['unexported_source_seconds'],13/3)
        self.assertEqual(self.long.stat().st_size,self.original_size)
        self.assertTrue(all(x['frames']==124 and x['audio_present'] for x in report['clips']))
        with zipfile.ZipFile(self.out/report['zip']) as archive:
            self.assertEqual(len([x for x in archive.namelist() if x.endswith('.mp4')]),34)
            self.assertIsNone(archive.testzip())
        for row in report['clips']:
            self.assertTrue((self.inp/row['input_file']).is_file())

    def test_real_numbered_range(self):
        report=self.export(mode=c.MODES[1],first_chunk=7,chunk_count=2,audio=c.AUDIO_MODES[1])
        self.assertEqual([x['source_start_exact'] for x in report['clips']],['31','217/6'])
        self.assertFalse(any(x['audio_present'] for x in report['clips']))

    def test_real_selected_start(self):
        report=self.export(start_seconds=93.25,make_zip=False,make_loadable=False)
        self.assertEqual(report['clips'][0]['source_start_seconds'],93.25)
        self.assertNotIn('input_file',report['clips'][0])

    def test_real_explicit_pad_last_chunk(self):
        report=self.export(first_chunk=35,final_chunk=c.TAIL_POLICIES[1],make_zip=False)
        row=report['clips'][0]
        self.assertTrue(row['tail_padded']);self.assertEqual(row['frames'],124)
        self.assertAlmostEqual(row['hold_last_frame_seconds'],5/6)

    def test_exact_124_frame_input_round_trip(self):
        first=self.export(make_zip=False)
        source=self.inp/first['clips'][0]['input_file']
        second=self.export(source,make_zip=False)
        self.assertEqual(len(second['clips']),1)
        self.assertEqual(second['clips'][0]['frames'],124)
        self.assertFalse(second['clips'][0]['tail_padded'])

    def test_original_audio_not_required(self):
        silent=self.inp/'silent.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(self.long),'-t','6','-an','-c:v','copy',str(silent)],check=True)
        result=self.export(silent,make_zip=False)
        self.assertFalse(result['clips'][0]['audio_present'])

    def test_input_trim_window_limits_export(self):
        report=self.export(mode=c.MODES[1],input_start=30,input_duration=12,make_zip=False)
        self.assertEqual(len(report['clips']),2)
        self.assertEqual(report['clips'][0]['source_start_seconds'],30)
        self.assertLessEqual(report['clips'][-1]['source_end_seconds'],42)

    def test_rotation_is_baked_and_square_pixels(self):
        rotated=self.inp/'rotated.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-display_rotation:v:0','90','-i',str(self.long),'-t','7','-c','copy',str(rotated)],check=True)
        report=self.export(rotated,make_zip=False)
        self.assertEqual(report['output_size'],[64,96])
        path=self.out/report['directory']/report['clips'][0]['file']
        data=c.probe(path)
        video=next(s for s in data['streams'] if s['codec_type']=='video')
        self.assertEqual(video['sample_aspect_ratio'],'1:1')
        self.assertTrue(all(int(x.get('rotation',0))==0 for x in video.get('side_data_list',[])))

    def test_variable_frame_rate_source(self):
        vfr=self.inp/'variable_rate.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(self.long),'-t','7',
            '-vf',"select='if(lt(t,3),not(mod(n,2)),1)'",'-vsync','vfr',
            '-c:v','libx264','-preset','ultrafast','-threads','2','-an',str(vfr)],check=True)
        report=self.export(vfr,make_zip=False)
        self.assertEqual(report['clips'][0]['frames'],124)
        self.assertEqual(report['clips'][0]['fps'],24)

    def test_anamorphic_source_becomes_square_pixels(self):
        anamorphic=self.inp/'anamorphic.mp4'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(self.long),'-t','6',
            '-vf','setsar=2','-c:v','libx264','-preset','ultrafast','-threads','2','-an',str(anamorphic)],check=True)
        report=self.export(anamorphic,make_zip=False)
        self.assertEqual(report['output_size'],[192,64])
        path=self.out/report['directory']/report['clips'][0]['file']
        stream=next(x for x in c.probe(path)['streams'] if x['codec_type']=='video')
        self.assertEqual(stream['sample_aspect_ratio'],'1:1')

    def test_selected_start_matches_source_picture(self):
        # Compare decoded pixels, not just the declared timestamp in a manifest.
        import numpy as np
        report=self.export(start_seconds=93.25,make_zip=False)
        exported=self.out/report['directory']/report['clips'][0]['file']
        def frame(path, start):
            raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(start),'-i',str(path),
                '-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','-'])
            return np.frombuffer(raw,dtype=np.uint8).astype(float)
        actual=frame(exported,0)
        expected=frame(self.long,93.25)
        wrong=frame(self.long,0)
        error=np.abs(actual-expected).mean()
        wrong_error=np.abs(actual-wrong).mean()
        self.assertLess(error,8.0)
        self.assertLess(error,wrong_error)

    def test_failure_does_not_publish_invalid_video(self):
        with patch.object(c,'verify_clip',side_effect=ValueError('synthetic invalid export')):
            with self.assertRaisesRegex(ValueError,'synthetic invalid'):
                self.export(make_zip=False)
        runs=list((self.out/'h3_clips').iterdir())
        self.assertEqual(len(runs),1)
        report=json.loads((runs[0]/'run_status.json').read_text())
        self.assertEqual(report['state'],'incomplete')
        self.assertFalse(list(runs[0].glob('*.mp4')))
        self.assertFalse(list(runs[0].glob('.*.part')))


if __name__ == '__main__':unittest.main()
