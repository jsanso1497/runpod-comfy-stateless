import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
import build_workflows as builder
import nodepack.rules as rules
import nodepack.media as media
import nodepack.nodes as custom

class Rules(unittest.TestCase):
    def test_only_requested_engines(self):self.assertEqual(rules.ENGINES,['MiniMax H3','Eros Beta 7 Fix1'])
    def test_h3_reference_checkpoint(self):self.assertIn('_ref2va_',rules.model_names('MiniMax H3','INT8 HQ','References')[0])
    def test_eros_checkpoint(self):self.assertIn('fix1-native_int8_protected_edges_HQ',rules.model_names(rules.ENGINES[1],'INT8 HQ','References')[0])
    def test_encoder_and_vae(self):
        for engine in rules.ENGINES:
            names=rules.model_names(engine,'INT8 HQ','References')
            self.assertIn('32b_minimax_h3_int8_convrot',names[1]);self.assertIn('vae_fp16',names[2])
    def test_no_bf16_stack(self):
        with self.assertRaises(ValueError):rules.model_names('MiniMax H3','BF16 full','References')
    def test_reference_required(self):
        with self.assertRaises(ValueError):rules.resolve_mode('Auto',[],False)
    def test_one_photo_is_reference(self):self.assertEqual(rules.resolve_mode('Auto',[1],False),'References')
    def test_replacement_requires_identity(self):
        with self.assertRaises(ValueError):rules.resolve_mode('Replace person',[1],False)
        self.assertEqual(rules.resolve_mode('Replace person',[1],True),'Replace person')
    def test_native_video_default(self):
        for engine in rules.ENGINES:self.assertFalse(rules.use_frames(engine,'Auto'))
    def test_samplers(self):
        self.assertEqual(rules.recipe('MiniMax H3')['steps'],50)
        e=rules.recipe(rules.ENGINES[1]);self.assertEqual((e['steps'],e['sampler'],e['shift']),(8,'euler',[12.,3.]))
    def test_image_budget(self):
        rules.reference_budget('MiniMax H3',2,7)
        with self.assertRaises(ValueError):rules.reference_budget('MiniMax H3',3,7)
    def test_tags_validate(self):
        self.assertEqual(rules.translate_prompt('Use <Video 1>.',0,0,True),'Use <Video 1>.')
        with self.assertRaises(ValueError):rules.translate_prompt('Use <Picture 2>.',1,0,False)
    def test_crop_and_geometry(self):
        self.assertEqual(rules.crop_box(1920,1080,25,0,25,0),(480,0,1440,1080))
        w,h=rules.dimensions(2,3,2.5);self.assertEqual((w%32,h%32),(0,0))

class ProfileFiles(unittest.TestCase):
    def test_four_assets_only(self):
        m=json.loads((P/'manifest.json').read_text());self.assertEqual(len(m['models']),4)
        self.assertAlmostEqual(sum(x['approx_gb'] for x in m['models']),76.180379616,places=4)
        for x in m['models']:self.assertEqual(len(x['sha256']),64);self.assertEqual(len(x['revision']),40)
    def test_no_bulk_bootstrap(self):
        t=(P/'runtime.py').read_text();self.assertNotIn('prepare(root)',t);self.assertNotIn("download(root,preload)",t)
        self.assertIn("'--cache-classic'",t);self.assertNotIn("'--highvram'",t)
    def test_native_experiment_not_imported(self):
        t=(P/'nodepack/nodes.py').read_text();self.assertNotIn('native_2mp',t);self.assertNotIn('PH3ImageConditioning',t)
        self.assertIn('MiniMaxH3ReferenceToVideo.execute',t)
    def test_stock_video_limits(self):
        with tempfile.TemporaryDirectory() as d:
            fake=types.ModuleType('folder_paths');fake.get_input_directory=lambda:d
            with mock.patch.dict(sys.modules,{'folder_paths':fake}):
                s=custom.H3A40Video.INPUT_TYPES()['required']
        self.assertEqual(s['native_video_megapixels'][1]['max'],1.)
        self.assertEqual(s['duration_seconds'][1]['max'],6.)
        self.assertIn('2*1024**3',(P/'nodepack/media.py').read_text())
    def test_workflow_links_and_api(self):
        for profile in builder.PROFILES:
            gui,api=builder.build(profile);by={n['id']:n for n in gui['nodes']}
            for edge,src,out,dst,inp,kind in gui['links']:
                self.assertIn(edge,by[src]['outputs'][out]['links'])
                self.assertEqual(by[dst]['inputs'][inp]['link'],edge)
                name=by[dst]['inputs'][inp]['name']
                self.assertEqual(api[str(dst)]['inputs'][name],[str(src),out])
            self.assertEqual(api['10']['inputs']['precision'],'INT8 HQ')
            self.assertEqual(api['10']['inputs']['reference_size'],'match')
    def test_seed_not_in_encoding_inputs(self):
        gui,api=builder.build(builder.PROFILES[0])
        self.assertEqual(api['19']['class_type'],'H3A40Encode')
        self.assertEqual(api['19']['inputs'],{'plan':['12',0]})
        self.assertIn('seed',api['15']['inputs']);self.assertEqual(api['15']['inputs']['conditioning'],['19',0])
    def test_default_and_push_differ_only_output(self):
        _,a=builder.build(builder.PROFILES[0]);_,b=builder.build(builder.PROFILES[3])
        for k in ('11','12','15','19'):self.assertEqual(a[k]['inputs'],b[k]['inputs'])
        self.assertEqual(a['10']['inputs']['output_megapixels'],2.5);self.assertEqual(b['10']['inputs']['output_megapixels'],4.)
    def test_no_missing_custom_input_names(self):
        with tempfile.TemporaryDirectory() as d:
            fake=types.ModuleType('folder_paths');fake.get_input_directory=lambda:d;fake.get_filename_list=lambda _:[]
            with mock.patch.dict(sys.modules,{'folder_paths':fake}):
                schemas={name:cls.INPUT_TYPES() for name,cls in custom.NODE_CLASS_MAPPINGS.items()}
        for profile in builder.PROFILES:
            _,api=builder.build(profile)
            for value in api.values():
                if value['class_type'] not in schemas:continue
                schema=schemas[value['class_type']]
                self.assertTrue(set(schema['required']).issubset(value['inputs']))
                self.assertTrue(set(value['inputs']).issubset(set(schema['required'])|set(schema.get('optional',{}))))

class MediaTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg unavailable')
    def test_real_clip_and_original_file_keyframes(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'clip.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=160x96:rate=24','-t','3.2',
                '-c:v','libx264','-pix_fmt','yuv420p','-y',str(p)],check=True)
            clip=media.extract(p,0.,3.1,False)
            self.assertEqual(clip['frames'].shape,(73,96,160,3))
            keys=media.extract(p,0.,3.1,True,count=2)
            self.assertEqual(len(keys['keyframes']),2);self.assertEqual(keys['summary']['representation'],'keyframes')
    def test_preprocessing_guard_restored(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable')
        with mock.patch.object(media,'probe_video',return_value=(1920,1080,20.)):
            with self.assertRaisesRegex(ValueError,'2 GiB'):media.extract(Path('unused.mp4'),0.,15.,False,native_mp=1.)
    def test_background_composite_untouched(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable')
        original=torch.rand(1,64,64,3);mask=torch.zeros(1,64,64);mask[:,20:40,20:40]=1
        candidate=torch.rand_like(original)
        result=custom.finish_image(dict(protect_background=True,original=original,original_mask=mask),candidate)
        self.assertTrue(torch.equal(result[mask==0],original[mask==0]))

class MockEncoding(unittest.TestCase):
    def test_seventy_three_frames_reach_native_conditioning(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable')
        events=[];calls=[]
        class Loader:
            def load_clip(self,*a):return ('clip-fixture',)
            def load_vae(self,*a):return ('vae-fixture',)
        class Native:
            @staticmethod
            def execute(**kwargs):calls.append(kwargs);events.append('encode');return ([[torch.zeros(1,1,8),{}]],{})
        comfy=types.ModuleType('comfy');comfy.__path__=[]
        mm=types.ModuleType('comfy.model_management')
        mm.unload_all_models=lambda:events.append('unload');mm.soft_empty_cache=lambda:None
        extra=types.ModuleType('comfy_extras');extra.__path__=[]
        upstream=types.ModuleType('comfy_extras.nodes_minimax_h3')
        upstream.MiniMaxH3ReferenceToVideo=Native;upstream.adapt_canvas=lambda w,h:(w,h)
        fake_nodes=types.ModuleType('nodes');fake_nodes.CLIPLoader=Loader;fake_nodes.VAELoader=Loader
        fp=types.ModuleType('folder_paths')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture';p.write_bytes(b'no model weights')
            fp.get_full_path_or_raise=lambda *a:str(p)
            modules={'comfy':comfy,'comfy.model_management':mm,'comfy_extras':extra,
                     'comfy_extras.nodes_minimax_h3':upstream,'nodes':fake_nodes,'folder_paths':fp}
            plan=dict(models=('diffusion','encoder','vae'),prompt='Use <Video 1>.',width=160,height=96,
                      conditioning_length=73,reference_size='match',images=[],videos=[torch.zeros(73,96,160,3)])
            with mock.patch.dict(sys.modules,modules),mock.patch.object(custom,'node_call',return_value=({'samples':'single-frame-fixture'},)):
                outputs=custom.H3A40Encode().encode(plan)
        self.assertEqual(calls[0]['ref_videos']['ref_video_0'].shape[0],73)
        self.assertEqual(calls[0]['length'],73);self.assertEqual(events,['encode','unload'])
        self.assertEqual(outputs[2],'vae-fixture');self.assertFalse(json.loads(outputs[3])['video'][0]['native_canvas_override'])

if __name__=='__main__':unittest.main()
