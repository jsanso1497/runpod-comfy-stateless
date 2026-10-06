import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parents[1]
def load(name,path,package=False):
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)] if package else None)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
N=load('_h3_media_tests',HERE/'node/__init__.py',True)
IO=N.media_io
I=load('_h3_media_install_tests',HERE/'install.py')
C=load('_h3_media_check_tests',HERE/'check.py')

def picture(label='face',target='the main subject'):
    return {'kind':'image','image':torch.rand(1,32,48,3),'label':label,'target':target,'use_for':'Distinctive facial identity only.'}
def wave():return {'waveform':torch.sin(torch.arange(48000,dtype=torch.float32)*.03).reshape(1,1,-1),'sample_rate':48000}
def voice():return {'kind':'audio','identity':None,'motion':None,'audio':wave(),'label':'voice','target':'the main subject','audio_role':N.AUDIO_ROLES[0],'use_for':''}
def video(visual=3,audio=True):
    return {'kind':'video','identity':torch.rand(1,32,48,3) if visual in (1,3) else None,
        'motion':torch.rand(39,32,48,3) if visual in (2,3) else None,'audio':wave() if audio else None,
        'label':'clip','target':'the main subject','use_for':'A deliberate hand wave.','audio_role':N.AUDIO_ROLES[0],
        'visual_role':N.VISUAL_ROLES[visual],'metadata':{}}
def plan_args(refs=None):
    vals={'references':[picture(),voice()] if refs is None else refs}
    for key,d in N.H3MediaPlan.INPUT_TYPES()['required'].items():
        if key!='references':vals[key]=d[1]['default']
    return vals

class References(unittest.TestCase):
    def test_bank_does_not_mutate_upstream(self):
        a=[picture()];b=N.H3MediaImageReference().append(torch.rand(1,32,32,3),'new','hair','person',a)[0]
        self.assertEqual(len(a),1);self.assertEqual(len(b),2)
    def test_reference_images_keep_original_pixels(self):
        a=picture();out=N.H3MediaImageReference().append(a['image'],'face','eyes','person')[0][0]['image'];self.assertEqual(out.data_ptr(),a['image'].data_ptr())
    def test_reject_bad_images(self):
        for im in (torch.zeros(2,32,32,3),torch.zeros(1,32,32,1),torch.zeros(1,32,32,3,dtype=torch.uint8)):
            with self.assertRaises(ValueError):N.image(im)
    def test_nine_images_allowed_ten_rejected(self):
        N.check_limits([picture() for _ in range(9)])
        with self.assertRaises(ValueError):N.check_limits([picture() for _ in range(10)])
    def test_three_video_and_audio_limits(self):
        N.check_limits([video() for _ in range(3)])
        with self.assertRaises(ValueError):N.check_limits([video() for _ in range(4)])
        with self.assertRaises(ValueError):N.check_limits([voice() for _ in range(4)])
    def test_selected_identity_frames_count_as_stills(self):
        with self.assertRaises(ValueError):N.check_limits([picture() for _ in range(9)]+[video(1,False)])
    def test_no_active_references_rejected(self):
        with self.assertRaises(ValueError):N.check_limits([])
    def test_stable_independent_tags(self):
        kw,rows=N.compile_references([voice(),picture(),video()])
        self.assertEqual(list(kw['ref_images']),['ref_image_0','ref_image_1'])
        self.assertEqual([r['tag'] for r in rows],['<Picture 1>','<Picture 2>','<Video 1>','<Audio 1>','<Audio 2>'])
        self.assertEqual(kw['ref_video_audios'],{})
    def test_voice_only_sends_no_visual_refs(self):
        kw,rows=N.compile_references([video(0)])
        self.assertEqual(kw['ref_images'],{});self.assertEqual(kw['ref_videos'],{});self.assertEqual(len(kw['ref_audios']),1)
    def test_action_only_does_not_supply_identity_frame(self):
        kw,rows=N.compile_references([video(2,False)]);self.assertEqual(kw['ref_images'],{});self.assertEqual(kw['ref_audios'],{})
        self.assertIn('Do not transfer',rows[0]['purpose'])
    def test_identity_only_sends_single_still(self):
        kw,_=N.compile_references([video(1,False)]);self.assertEqual(len(kw['ref_images']),1);self.assertEqual(kw['ref_videos'],{})
    def test_audio_trim_keeps_sample_rate_and_values(self):
        a=wave();out=N.H3MediaAudioReference().append(a,'voice','person',N.AUDIO_ROLES[0],.25,.5)[0][0]['audio']
        self.assertEqual(out['sample_rate'],48000);torch.testing.assert_close(out['waveform'],a['waveform'][...,12000:36000])
    def test_audio_empty_silent_and_nan_rejected(self):
        for w in (torch.zeros(1,1,48000),torch.full((1,1,48000),float('nan')),torch.ones(1,1,100)):
            with self.assertRaises(ValueError):N.H3MediaAudioReference().append({'waveform':w,'sample_rate':48000},'voice','p',N.AUDIO_ROLES[0],0,1)
    def test_both_video_roles_off_rejected_before_decode(self):
        with self.assertRaisesRegex(ValueError,'Both streams'):
            N.H3MediaVideoReference().append(None,'clip','p',N.VISUAL_ROLES[0],'Ignore audio',0,5,.5,0)

class PromptAndNative(unittest.TestCase):
    def test_new_words_and_audio_reference_not_copy(self):
        a=plan_args();a['new_dialogue']='These exact new words.';p,report=N.H3MediaPlan().prepare(**a)
        self.assertIn('<d>[English] These exact new words.</d>',p['prompt']);self.assertIn(': reference -',p['prompt']);self.assertNotIn('fully_copy',p['prompt'])
        self.assertIn('No input-audio passthrough',report)
    def test_subject_ids_consistent_and_speaker_not_in_retention(self):
        p,_=N.H3MediaPlan().prepare(**plan_args());text=p['prompt'];self.assertIn('<Subject 1>',text);self.assertIn('(S1)',text)
        self.assertNotIn('(S1)',text.split('retention_analysis:')[1].split('detailed_description:')[0])
    def test_six_sections_in_order(self):
        p,_=N.H3MediaPlan().prepare(**plan_args());names=['subject_definitions:','summary:','retention_analysis:','detailed_description:','overall_soundscape:','non_diegetic_music:']
        offsets=[p['prompt'].index(n) for n in names];self.assertEqual(offsets,sorted(offsets))
    def test_empty_dialogue_no_added_speech(self):
        a=plan_args();a['new_dialogue']='';p,_=N.H3MediaPlan().prepare(**a);self.assertNotIn('<d>',p['prompt'])
    def test_unknown_reference_in_advanced_prompt_rejected(self):
        a=plan_args();a['advanced_prompt']='Use <Audio 3>.'
        with self.assertRaises(ValueError):N.H3MediaPlan().prepare(**a)
    def test_advanced_prompt_preserved(self):
        a=plan_args();a['advanced_prompt']='Use <Audio 1> for the voice: <d>[English] Yes.</d>';p,_=N.H3MediaPlan().prepare(**a);self.assertEqual(p['prompt'],a['advanced_prompt'])
    def test_plain_dialogue_rejects_double_markup(self):
        a=plan_args();a['new_dialogue']='<d>Hi</d>'
        with self.assertRaises(ValueError):N.H3MediaPlan().prepare(**a)
    def test_invalid_language(self):
        a=plan_args();a['language']='English] injected ['
        with self.assertRaises(ValueError):N.H3MediaPlan().prepare(**a)
    def test_frame_grid_and_aspects(self):
        for aspect,(w,h) in N.ASPECTS.items():
            for seconds in (3,5,8,15):
                a=plan_args();a.update(aspect=aspect,seconds=seconds);p,_=N.H3MediaPlan().prepare(**a)
                self.assertEqual((p['width'],p['height']),(w,h));self.assertEqual((p['length']-5)%17,0);self.assertGreaterEqual(p['length'],seconds*24)
    def test_invalid_duration(self):
        for seconds in (True,2,16,5.5):
            a=plan_args();a['seconds']=seconds
            with self.assertRaises(ValueError):N.H3MediaPlan().prepare(**a)
    def test_native_receives_all_real_media_inputs(self):
        received={}
        def execute(clip,vae,audio_vae,prompt,width,height,length,ref_image_size,ref_images,ref_videos,ref_video_audios,ref_audios):
            received.update(locals());return ['condition'],{'samples':'latent'}
        module=types.ModuleType('comfy_extras.nodes_minimax_h3');module.MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=execute)
        parent=types.ModuleType('comfy_extras');p,_=N.H3MediaPlan().prepare(**plan_args([picture(),video()]))
        with patch.dict(sys.modules,{'comfy_extras':parent,'comfy_extras.nodes_minimax_h3':module}):out=N.H3MediaConditioning().encode(p,'clip','video-vae','audio-vae')
        self.assertEqual(out[0],['condition']);self.assertEqual(received['audio_vae'],'audio-vae')
        for key in ('ref_images','ref_videos','ref_audios'):self.assertIs(received[key],p[key])
    def test_incompatible_native_api_fails_not_silently_degraded(self):
        module=types.ModuleType('comfy_extras.nodes_minimax_h3');module.MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=lambda **kw:None)
        with patch.dict(sys.modules,{'comfy_extras':types.ModuleType('comfy_extras'),'comfy_extras.nodes_minimax_h3':module}),self.assertRaisesRegex(RuntimeError,'incompatible'):
            N.H3MediaConditioning().encode({},None,None,None)

class LocalVideo:
    def __init__(self,path,trim=(0,0)):self.path=path;self.trim=trim
    def get_stream_source(self):return str(self.path)
    def get_active_trim_window(self):return self.trim

class MP4Decoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)/'reference.mp4';cls.silent=Path(cls.temp.name)/'silent.mp4'
        cmd=['ffmpeg','-v','error','-nostdin','-y','-f','lavfi','-i','testsrc2=size=96x64:rate=30:duration=3','-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=3','-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-c:a','aac','-ac','2','-shortest',str(cls.path)]
        subprocess.run(cmd,check=True,capture_output=True,timeout=30)
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(cls.path),'-c:v','copy','-an',str(cls.silent)],check=True,capture_output=True,timeout=30)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def decode(self,visual=0,audio=0,start=.2,duration=2,path=None,pos=.5):
        return IO.decode_reference(LocalVideo(path or self.path),start,duration,N.VISUAL_ROLES[visual],N.AUDIO_ROLES[audio],pos,0)
    def test_voice_only_never_decodes_pixels(self):
        with patch.object(IO,'read_frames',side_effect=AssertionError('Do not decode video')):im,motion,audio,meta=self.decode()
        self.assertIsNone(im);self.assertIsNone(motion);self.assertEqual(audio['sample_rate'],48000);self.assertEqual(audio['waveform'].shape[1],2)
        self.assertAlmostEqual(audio['waveform'].shape[-1]/48000,2,places=2);self.assertEqual(meta['video_frames'],0)
    def test_identity_extracts_original_frame_not_sequence(self):
        im,motion,audio,meta=self.decode(1,2);self.assertEqual(tuple(im.shape),(1,64,96,3));self.assertIsNone(motion);self.assertIsNone(audio)
        self.assertAlmostEqual(meta['identity_frame_seconds'],1.2)
    def test_motion_rate_conversion_and_grid(self):
        im,frames,audio,meta=self.decode(2,2);self.assertIsNone(im);self.assertEqual(len(frames),39);self.assertEqual(meta['video_fps'],24)
        self.assertEqual((len(frames)-5)%17,0);self.assertLessEqual(meta['video_seconds'],2)
    def test_all_three_reference_roles(self):
        im,frames,audio,meta=self.decode(3,0);self.assertIsNotNone(im);self.assertIsNotNone(frames);self.assertIsNotNone(audio);self.assertFalse(meta['source_audio_is_final_output'])
    def test_no_audio_stream_has_clear_error(self):
        with self.assertRaisesRegex(ValueError,'audio track'):self.decode(path=self.silent)
    def test_silent_clip_valid_for_identity(self):self.assertIsNotNone(self.decode(1,2,path=self.silent)[0])
    def test_end_trim_clamps_without_looping(self):
        _,_,audio,meta=self.decode(start=2,duration=5);self.assertAlmostEqual(meta['selected_seconds'],1);self.assertAlmostEqual(audio['waveform'].shape[-1]/48000,1,places=2)
    def test_past_end_raises(self):
        with self.assertRaisesRegex(ValueError,'end'):self.decode(start=5)
    def test_active_native_trim_respected(self):
        _,_,a,meta=IO.decode_reference(LocalVideo(self.path,(1,1)),.25,5,N.VISUAL_ROLES[0],N.AUDIO_ROLES[0],.5,0)
        self.assertEqual(meta['start_seconds'],1.25);self.assertEqual(meta['selected_seconds'],.75)
    def test_source_file_not_modified(self):
        before=self.path.read_bytes();self.decode(3,0);self.assertEqual(before,self.path.read_bytes())
    def test_identity_at_end_is_valid(self):self.assertEqual(self.decode(1,2,pos=1)[0].shape[0],1)
    def test_reject_nonfile_video(self):
        with self.assertRaisesRegex(ValueError,'locally uploaded'):IO.local_video_source(LocalVideo('https://example.com/private.mp4'))
    def test_orientation_sar_and_hdr(self):
        self.assertEqual(IO.displayed_dimensions({'width':100,'height':50,'sample_aspect_ratio':'2:1','tags':{'rotate':'90'}}),(50,200))
        with self.assertRaisesRegex(ValueError,'HDR'):IO.displayed_dimensions({'width':100,'height':50,'color_transfer':'smpte2084'})

class InstallAndGraph(unittest.TestCase):
    def test_module_schema_matches_functions(self):
        for name,cls in N.NODE_CLASS_MAPPINGS.items():
            with self.subTest(name=name):
                defs={k for group in ('required','optional') for k in cls.INPUT_TYPES().get(group,{})}
                sig=inspect.signature(getattr(cls,cls.FUNCTION));self.assertTrue(defs.issubset(sig.parameters))
    def test_defaults_are_voice_only(self):
        schema=N.H3MediaVideoReference.INPUT_TYPES()['required'];self.assertEqual(schema['visual_role'][1]['default'],N.VISUAL_ROLES[0])
    def test_disabled_runtime_does_not_download(self):
        with patch.dict(os.environ,{'ENABLE_H3_MEDIA':'0'}):self.assertEqual(I.runtime('/not-used','/not-used'),[])
    def test_invalid_flag_fails(self):
        with patch.dict(os.environ,{'ENABLE_H3_MEDIA':'maybe'}),self.assertRaises(ValueError):I.enabled()
    def test_install_preserves_user_edits(self):
        with tempfile.TemporaryDirectory() as d:
            file=I.install_workflows(d)[0];file.write_text('user-edited');I.install_workflows(d);self.assertEqual(file.read_text(),'user-edited')
    def test_full_catalog_has_hashes_and_four_models(self):
        rows=json.loads((HERE/'config/models.json').read_text());self.assertEqual(len(rows),4)
        for r in rows:self.assertEqual(len(r['sha256']),64);self.assertNotIn('int8',r['filename']);self.assertNotIn('turbo',r['filename'])
    def test_graph_static(self):C.static()
    def test_source_audio_passthrough_tamper_rejected(self):
        g=copy.deepcopy(C.static()[0]);n=next(n for n in g['nodes'] if n['type']=='VAEDecodeAudio');n['type']='LoadAudio'
        with self.assertRaises(ValueError):C.validate_graph(g)
    def test_graph_local_widgets_match_schema(self):
        for n in C.static()[0]['nodes']:
            if n['type'] not in N.NODE_CLASS_MAPPINGS:continue
            cls=N.NODE_CLASS_MAPPINGS[n['type']];names=[]
            for group in ('required','optional'):
                for key,d in cls.INPUT_TYPES().get(group,{}).items():
                    if isinstance(d[0],list) or d[0] in ('STRING','INT','FLOAT','BOOLEAN'):names.append(key)
            self.assertEqual(list(n['widgets_values_named']),names)
    @unittest.skipUnless((HERE.parent/'h3_portrait/workflows').is_dir(),'repository-only generator check')
    def test_workflow_reproducible(self):
        file=next((HERE/'workflows').glob('*.json'));before=file.read_bytes();builder=load('_h3_media_graph_builder_test',HERE/'make_workflow.py');builder.make();self.assertEqual(before,file.read_bytes())
    def test_native_schema_guard(self):
        with self.assertRaisesRegex(ValueError,'Missing'):C.validate_native_registry({})
