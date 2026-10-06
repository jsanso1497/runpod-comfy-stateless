"""Role routing, frontend serialization, profile and real final-frame regressions.

No model downloads, personal LoRA URLs, Ollama inference or GPU are used here.
"""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import node
from node import logic, ollama_client as oc, reference_roles as rr
from node.video_export import decode_last_frame,safe_output,H3PortraitExportVideo,H3PortraitSaveLastFrame
from node.native_helpers import H3PortraitCropToAspect
from install_workflows import configured_graph,install
from test_thinking import analysis,director
from test_split_models import pipeline,packet
import test_pipeline as pipeline_fixtures
from smoke_check import validate_graph,validate_registry


def three():
    value=analysis(3)
    for r,role,description in zip(value['references'],['face','body','pose_camera'],
            ['facial identity only','body proportions only','eye-level three-quarter view; torso turned left; one hand on hip; medium framing']):
        r['role']=role;r['use_for']=description
    return value


def uploads(roles=('auto','auto','auto')):
    return [{'filename':f'source_{i}.png','image':object(),'sha256':str(i),'role':role} for i,role in enumerate(roles,1)]


class RoutingTests(unittest.TestCase):
    def test_third_pose_omitted_from_native_images(self):
        a=rr.route_analysis(three(),uploads())
        self.assertEqual(a['_routing']['native_source_indices'],[0,1])
        self.assertEqual(len(a['references']),2)
        self.assertEqual(a['_routing']['ledger'][2]['delivery'],'text_only')
    def test_override_wins_over_model_role(self):
        a=analysis(3);out=rr.route_analysis(a,uploads(('face','body','pose_camera')))
        self.assertEqual(out['_routing']['native_source_indices'],[0,1])
    def test_auto_from_model_uses_role(self):
        self.assertEqual(rr.role_of({'role':'pose_camera'}),'pose_camera')
    def test_legacy_role_fallback(self):
        self.assertEqual(rr.role_of({'use_for':'camera angle and pose only'}),'pose_camera')
    def test_identity_and_pose_combined_not_dropped_without_explicit_scope(self):
        self.assertEqual(rr.role_of({'use_for':'identity and pose'}),'identity')
    def test_all_visual_mode_is_explicit_comparison(self):
        a=rr.route_analysis(three(),uploads(),'All images visual (comparison)')
        self.assertEqual(a['_routing']['native_source_indices'],[0,1,2])
    def test_middle_guide_renumbers_final_subject(self):
        a=three();a['references'][1]['role']='pose_camera';a['references'][2]['role']='body'
        out=rr.route_analysis(a,uploads())
        self.assertEqual(out['_routing']['source_to_h3'],{1:1,2:None,3:2})
        self.assertEqual([r['image'] for r in out['references']],[1,2])
    def test_guide_performer_does_not_create_a_third_subject(self):
        a=three();a['references'][2]['subject']='Source pose performer'
        out=rr.route_analysis(a,uploads())
        self.assertEqual(out['guidance'][0]['target_subject'],'Person A')
    def test_mapping_does_not_modify_source_map(self):
        a=three();before=copy.deepcopy(a);rr.route_analysis(a,uploads());self.assertEqual(a,before)
    def test_role_order_and_filename_ledger(self):
        out=rr.route_analysis(three(),uploads())
        self.assertEqual([x['source_image'] for x in out['_routing']['ledger']],[1,2,3])
    def test_no_visual_identity_fails(self):
        with self.assertRaises(ValueError):rr.route_analysis(analysis(1),uploads(('pose_camera',)))
    def test_ignore_contributes_neither_text_nor_pixels(self):
        out=rr.route_analysis(three(),uploads(('face','body','ignore')))
        self.assertFalse(out['guidance']);self.assertEqual(out['_routing']['ledger'][2]['delivery'],'ignored')
    def test_expression_only_is_text(self):
        out=rr.route_analysis(three(),uploads(('face','body','expression')))
        self.assertEqual(out['guidance'][0]['role'],'expression')
    def test_invalid_override_fails(self):
        with self.assertRaises(ValueError):rr.role_of({},'facee')
    def test_invalid_model_role_fails(self):
        with self.assertRaises(ValueError):rr.role_of({'role':'facee'})
    def test_missing_entry_fails(self):
        with self.assertRaises(ValueError):rr.route_analysis(three(),uploads(('face','body')))
    def test_unknown_routing_mode_fails(self):
        with self.assertRaises(ValueError):rr.route_analysis(three(),uploads(),'fast')
    def test_nonsequential_map_rejected(self):
        a=three();a['references'].reverse()
        with self.assertRaises(ValueError):rr.route_analysis(a,uploads())
    def test_tags_and_plain_numbers_remapped_once(self):
        text=rr.remap_text('<Picture 3> and image 1; Image 2 guides.',{1:1,2:None,3:2})
        self.assertEqual(text,'<Picture 2> and <Picture 1>; text-only guide from source image 2 guides.')
    def test_ordinal_aliases(self):
        self.assertEqual(rr.remap_text('the third photo',{1:1,2:2,3:None}),'the text-only guide from source image 3')
    def test_plural_ranges_are_remapped(self):
        text=rr.remap_text('Images 1-3 show examples.',{1:1,2:None,3:2})
        self.assertIn('<Picture 1> and text-only guide from source image 2 and <Picture 2>',text)
    def test_nonexistent_picture_fails(self):
        with self.assertRaises(ValueError):rr.remap_text('Image 4',{1:1})
    def test_final_prompt_has_no_unsupplied_picture_tag(self):
        a=rr.route_analysis(three(),uploads());o=director(2);o['references']=a['references'];o['_routing']=a['_routing'];o['_guidance']=a['guidance']
        text=logic.format_prompt(o,'Image 3 is pose only. Image 1 is my face.',logic.geometry('9:16','Standard',5))
        self.assertNotIn('<Picture 3>',text);self.assertIn('one hand on hip',text);self.assertIn('text-only guide',text)
    def test_subject_definitions_retain_role_limits(self):
        text=logic.subject_definitions(rr.route_analysis(three(),uploads())['references'])
        self.assertIn('no clothing, jewelry',text);self.assertIn('body proportions',text)
    def test_analysis_role_is_required_in_requested_json(self):
        self.assertIn('role',logic.analysis_schema(3)['properties']['references']['items']['required'])
    def test_new_prompt_excludes_incidental_conflicts(self):
        text=(ROOT/'node/analysis_prompt.txt').read_text().lower()
        for item in ('jewelry','absence','clarification','pose_camera','role'):self.assertIn(item,text)


class PromptPipeline14Tests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'settings.json').read_text());self.refs=uploads(('face','body','pose_camera'))
        self.info=pipeline();self.recipe=logic.geometry('9:16','Standard',5)
    @patch.object(oc,'preview',return_value='encoded')
    @patch.object(oc,'unload')
    def test_all_three_analyzed_but_only_two_pass_to_h3_map(self,unload,preview):
        s=MagicMock();s.post.side_effect=[packet(three()),packet(director(2))]
        out=oc.generate(s,self.cfg,self.info,'system','1/2 my subject; third is pose only',self.refs,self.recipe,'',0)
        first,second=[c.kwargs['json'] for c in s.post.call_args_list]
        self.assertEqual(len(first['messages'][1]['images']),3)
        self.assertIn('user-selected role: pose_camera',first['messages'][1]['content'])
        self.assertFalse(any('images' in m for m in second['messages']))
        self.assertEqual(len(out['references']),2)
        self.assertEqual(out['_routing']['native_source_indices'],[0,1])
        self.assertIn('one hand on hip',second['messages'][1]['content'])
    @patch.object(oc,'preview',return_value='encoded')
    @patch.object(oc,'unload')
    def test_lite_one_helper_and_no_thinking(self,unload,preview):
        cfg=json.loads((ROOT/'profiles/lite/settings.json').read_text())
        info={'model':cfg['ollama_model'],'capabilities':['vision'],'analysis':{'model':cfg['analysis_model'],'capabilities':['vision']}}
        s=MagicMock();s.post.side_effect=[packet(three()),packet(director(2))]
        oc.generate(s,cfg,info,'system','my subject',self.refs,logic.geometry('9:16','Standard',5,cfg),'',0)
        calls=[c.kwargs['json'] for c in s.post.call_args_list]
        self.assertEqual(len({c['model'] for c in calls}),1)
        self.assertTrue(all(not c.get('think',False) for c in calls));self.assertEqual(unload.call_count,1)
        self.assertEqual(calls[0]['keep_alive'],'5m');self.assertEqual(calls[1]['keep_alive'],0)
    @patch.object(oc,'preview',return_value='encoded')
    @patch.object(oc,'unload')
    def test_explicit_roles_get_one_clarification_review(self,unload,preview):
        question=three();question['clarification']='Copy the necklace?'
        s=MagicMock();s.post.side_effect=[packet(question),packet(three()),packet(director(2))]
        out=oc.generate(s,self.cfg,self.info,'system','Face/body/pose only.',self.refs,self.recipe,'',0)
        self.assertEqual(s.post.call_count,3);self.assertEqual(out['_stages'][0]['attempts'],2)
        self.assertIn('ROLE REVIEW',s.post.call_args_list[1].kwargs['json']['messages'][-1]['content'])
    @patch.object(oc,'preview',return_value='encoded')
    @patch.object(oc,'unload')
    def test_genuine_clarification_not_silently_ignored(self,unload,preview):
        question=three();question['clarification']='Which person in the group is the target?'
        s=MagicMock();s.post.side_effect=[packet(question),packet(question)]
        with self.assertRaises(logic.ClarificationNeeded):oc.generate(s,self.cfg,self.info,'system','Face/body/pose only.',self.refs,self.recipe,'',0)
        self.assertEqual(s.post.call_count,2);unload.assert_called_once()
    def test_cache_changes_when_routing_mode_changes(self):
        a=logic.cache_key(['a'],'walk',self.recipe,'d',self.cfg,' ',0)
        b=logic.cache_key(['a'],'walk',self.recipe,'d',dict(self.cfg,reference_mode='All images visual (comparison)'),' ',0)
        self.assertNotEqual(a,b)


class FinalHandoffTests(unittest.TestCase):
    setUp = pipeline_fixtures.DirectorTests.setUp
    tearDown = pipeline_fixtures.DirectorTests.tearDown
    def test_director_job_filters_pixels_and_preserves_tensor_identity(self):
        self.refs=uploads(('face','body','pose_camera'));k=dict(self.kw,references=self.refs)
        a=rr.route_analysis(three(),self.refs);value=director(2);value.update(references=a['references'],_routing=a['_routing'],_guidance=a['guidance'])
        with patch.dict(sys.modules,self.modules),patch.object(logic,'settings',return_value=self.cfg),patch.object(oc,'pipeline_info',return_value=pipeline()),patch.object(oc,'generate',return_value=value),patch.object(oc,'api',return_value={'models':[]}):
            job,text=node.H3PortraitDirector().direct(**k)
        self.assertEqual(len(job['references']),2)
        self.assertIs(job['references'][0]['image'],self.refs[0]['image']);self.assertIs(job['references'][1]['image'],self.refs[1]['image'])
        self.assertNotIn(self.refs[2],job['references']);self.assertIn('Image 3: pose_camera -> text only',text)


class WidgetsAndLiteTests(unittest.TestCase):
    def test_serialized_values_match_actual_ui_including_seed_companion(self):
        fp=types.ModuleType('folder_paths');fp.get_filename_list=lambda name:[]
        with patch.dict(sys.modules,{'folder_paths':fp}):schema=node.H3PortraitDirector.INPUT_TYPES()
        widgets=[]
        for group in ('required','optional'):
            for name,spec in schema[group].items():
                typ=spec[0]
                if isinstance(typ,list) or typ in ('STRING','INT','FLOAT','BOOLEAN'):
                    widgets.append((name,spec))
                    if name in ('seed','noise_seed') and (len(spec)<2 or spec[1].get('control_after_generate',True)):
                        widgets.append(('control_after_generate',(['fixed','increment','decrement','randomize'],)))
        graph=json.loads((ROOT/'workflows/H3_Portrait_Auto.json').read_text())
        values=next(n['widgets_values'] for n in graph['nodes'] if n['type']=='H3PortraitDirector')
        self.assertEqual(len(values),len(widgets));self.assertEqual(len(values),15)
        for (name,spec),v in zip(widgets,values):
            with self.subTest(name=name):
                if isinstance(spec[0],list):self.assertIn(v,spec[0])
                elif spec[0]=='INT':self.assertIs(type(v),int)
                elif spec[0]=='FLOAT':self.assertTrue(type(v) in (int,float) and math.isfinite(v))
                elif spec[0]=='BOOLEAN':self.assertIs(type(v),bool)
                elif spec[0]=='STRING':self.assertIs(type(v),str)
        mapping=dict(zip((w[0] for w in widgets),values))
        self.assertEqual(mapping['control_after_generate'],'fixed');self.assertEqual(mapping['mode'],'Draft only')
        self.assertEqual(mapping['lora_1'],'(none)');self.assertEqual(mapping['lora_2'],'(none)');self.assertEqual(mapping['strength_2'],0)
    def test_profile_lite_model_family_and_geometry(self):
        cfg=json.loads((ROOT/'profiles/lite/settings.json').read_text())
        self.assertNotIn('pruned',cfg['model_files']['diffusion']);self.assertIn('int8_convrot',cfg['model_files']['diffusion'])
        r=logic.geometry('9:16','Standard',5,cfg)
        self.assertEqual((r['width'],r['height'],r['steps'],r['ref_image_size']),(576,1024,16,'match'))
        r=logic.geometry('2:3','High fidelity',5,cfg);self.assertEqual((r['width'],r['height'],r['steps']),(576,864,20))
    def test_lite_settings_load(self):
        with patch.object(logic,'ROOT',ROOT/'profiles/lite'),patch.dict(os.environ,{},clear=True):
            cfg=logic.settings();self.assertFalse(cfg['think']);self.assertEqual(cfg['ollama_model'],cfg['analysis_model'])
    def test_wrong_copied_env_on_lite_stops_with_actionable_error(self):
        with patch.object(logic,'ROOT',ROOT/'profiles/lite'),patch.dict(os.environ,{'OLLAMA_MODEL':logic.DEFAULT_MODEL}):
            with self.assertRaisesRegex(ValueError,'Replace the copied OLLAMA_MODEL'):logic.settings()
    def test_installer_generates_two_workflows_each_profile(self):
        for profile,source in [('Full',ROOT/'settings.json'),('Lite',ROOT/'profiles/lite/settings.json')]:
            with self.subTest(profile=profile),tempfile.TemporaryDirectory() as td:
                cfg=json.loads(source.read_text());install(td,cfg)
                files=sorted((Path(td)/'user/default/workflows').glob('*.json'));self.assertEqual(len(files),2)
                self.assertTrue(all(profile+'_v1_4' in p.name for p in files))
                for p in files:validate_graph(json.loads(p.read_text()))
                native=json.loads(next(p for p in files if 'Standard' in p.name).read_text())
                loader=next(n for n in native['nodes'] if n['type']=='UNETLoader')
                self.assertEqual(loader['widgets_values'][0],cfg['model_files']['diffusion'])
    def test_installer_preserves_user_saved_changes(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=json.loads((ROOT/'settings.json').read_text());install(td,cfg)
            path=Path(td)/'user/default/workflows/H3_Portrait_Full_v1_4.json';path.write_text('user edited')
            install(td,cfg);self.assertEqual(path.read_text(),'user edited')
    def test_native_graph_has_no_ollama_director_or_omni(self):
        g=json.loads((ROOT/'workflows/H3_Ref2VA_Standard.json').read_text());types_={n['type'] for n in g['nodes']}
        self.assertIn('MiniMaxH3ReferenceToVideo',types_);self.assertNotIn('H3PortraitDirector',types_)
        self.assertNotIn('OmniNode',types_);self.assertIn('H3PortraitSaveLastFrame',types_)
    def test_both_graphs_have_explicit_export_dependency(self):
        for p in (ROOT/'workflows').glob('*.json'):
            graph=json.loads(p.read_text());validate_graph(graph)
            export=next(n for n in graph['nodes'] if n['type']=='H3PortraitExportVideo')
            last=next(n for n in graph['nodes'] if n['type']=='H3PortraitSaveLastFrame')
            link=next(l for l in graph['links'] if l[0]==last['inputs'][0]['link'])
            self.assertEqual(link[1],export['id']);self.assertEqual(link[-1],'H3_EXPORTED_VIDEO')
    def test_exact_crop_shapes(self):
        import torch
        for w,h,aspect,ow,oh in [(768,1344,'9:16',756,1344),(576,1024,'9:16',576,1024),(768,1152,'2:3',768,1152)]:
            out=H3PortraitCropToAspect().crop(torch.zeros(1,h,w,3),aspect)[0]
            self.assertEqual(tuple(out.shape),(1,oh,ow,3))


class RealFinalFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.path=cls.root/'fixture.mp4'
        import numpy as np
        frames=np.zeros((6,24,32,3),dtype=np.uint8)
        for i in range(6):frames[i,:,:,i%3]=30+i*35
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','32x24','-r','4','-i','pipe:0','-f','lavfi','-i','sine=frequency=440:sample_rate=32000','-t','1.5','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac',str(cls.path)],input=frames.tobytes(),check=True,timeout=30)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_last_decoded_frame_count_not_guessed_timestamp(self):
        array,meta=decode_last_frame(self.path)
        self.assertEqual(meta['decoded_frames'],6);self.assertEqual(meta['last_frame_index'],5)
        self.assertEqual(array.shape,(24,32,3));self.assertEqual(meta['average_fps'],4)
    def test_pixels_equal_independent_ffmpeg_last_frame(self):
        import numpy as np
        array,_=decode_last_frame(self.path)
        raw=subprocess.run(['ffmpeg','-v','error','-i',str(self.path),'-vf',r'select=eq(n\,5)','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],capture_output=True,check=True,timeout=30).stdout
        self.assertTrue(np.array_equal(array,np.frombuffer(raw,dtype=np.uint8).reshape(24,32,3)))
    def test_decode_cancellation_does_not_hang(self):
        def stop():raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError):decode_last_frame(self.path,stop)
    def test_decode_timeout_does_not_hang(self):
        with self.assertRaises(TimeoutError):decode_last_frame(self.path,timeout=-1)
    def test_missing_video_stops(self):
        with self.assertRaises(subprocess.CalledProcessError):decode_last_frame(self.root/'absent.mp4')
    def test_unsafe_output_paths_rejected(self):
        for path in ('../escape.mp4','/tmp/escape.mp4','a\\b.mp4','\x00',''):
            with self.subTest(path=path),self.assertRaises(ValueError):safe_output(path,self.root)
    def test_export_and_png_from_completed_file_keep_audio(self):
        from PIL import Image
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            fp=types.ModuleType('folder_paths');fp.get_output_directory=lambda:td
            def savepath(prefix,root,w,h):
                folder=Path(root)/'paired';folder.mkdir(exist_ok=True);return str(folder),'clip',1,'paired',prefix
            fp.get_save_image_path=savepath
            mm=types.ModuleType('comfy.model_management');mm.throw_exception_if_processing_interrupted=lambda:None
            comfy=types.ModuleType('comfy');comfy.model_management=mm
            args=types.ModuleType('comfy.cli_args');args.args=types.SimpleNamespace(disable_metadata=False)
            api=types.ModuleType('comfy_api.latest');api.Types=types.SimpleNamespace(VideoContainer=lambda v:v,VideoCodec=lambda v:v)
            modules={'folder_paths':fp,'comfy':comfy,'comfy.model_management':mm,'comfy.cli_args':args,'comfy_api.latest':api}
            class Video:
                def get_dimensions(inner):return (32,24)
                def save_to(inner,path,*,format,codec,metadata=None,crf=None):
                    self.assertEqual((format,codec),('mp4','h264'));self.assertEqual(crf,18)
                    shutil.copyfile(self.path,path)
            with patch.dict(sys.modules,modules):
                export=H3PortraitExportVideo().save(Video(),'paired/clip',18)
                manifest=export['result'][0];result=H3PortraitSaveLastFrame().save(manifest)
                video=Path(td)/manifest['relative_path'];png=Path(td)/result['result'][1]
                expected,_=decode_last_frame(video)
                self.assertTrue(np.array_equal(np.asarray(Image.open(png)),expected))
                self.assertEqual(png.stem,video.stem+'_last');self.assertTrue(png.with_suffix('.json').is_file())
                streams=json.loads(subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','json',str(video)],capture_output=True,text=True,check=True).stdout)['streams']
                self.assertIn('audio',[s['codec_type'] for s in streams])
                video.write_bytes(video.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError,'changed after saving'):H3PortraitSaveLastFrame().save(manifest)


if __name__=='__main__':unittest.main()
