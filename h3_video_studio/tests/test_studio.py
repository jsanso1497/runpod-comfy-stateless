"""Offline routing, graph, geometry and actual FFmpeg tests. No model weights."""
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from nodepack import rules
from nodepack import media
from nodepack.nodes import H3VSettings,H3VPlan,contact_sheet,finish_image
from build_workflows import build,PROFILES

class RuleTests(unittest.TestCase):
    def test_recipes(self):
        self.assertEqual(rules.recipe('MiniMax H3')['steps'],50)
        self.assertEqual(rules.recipe('MiniMax H3')['sampler'],'er_sde')
        self.assertEqual(rules.recipe('Eros Beta 7 Fix1')['shift'],[12.,3.])
        self.assertEqual(len(rules.recipe('P-H3 Image')['sigmas']),9)
        with self.assertRaises(ValueError):rules.recipe('Eros Beta 7 Fix1',eros_steps=50)
    def test_model_families(self):
        for engine in rules.ENGINES:
            for precision in rules.PRECISIONS:
                values=rules.model_names(engine,precision,'References')
                self.assertEqual(len(values),3)
                if engine=='P-H3 Image':self.assertTrue(all('p-h3-image' in n for n in values))
                if precision=='BF16 full':self.assertIn('bf16',values[1])
        self.assertIn('ref2va',rules.model_names('MiniMax H3','INT8 HQ','References')[0])
        self.assertIn('fl2va',rules.model_names('MiniMax H3','INT8 HQ','Generate')[0])
    def test_frame_limit(self):
        rules.reference_budget('P-H3 Image',2,6)
        with self.assertRaises(ValueError):rules.reference_budget('P-H3 Image',3,6)
        rules.reference_budget('MiniMax H3',3,6)
    def test_tag_mapping(self):
        text=rules.translate_prompt('Use <Picture 1> and <Video 1>.',2,4,False)
        self.assertEqual(text,'Use <Picture 1> and <Picture 3>, <Picture 4>, <Picture 5>, <Picture 6>.')
        self.assertEqual(rules.translate_prompt('Use <Video 1>.',0,0,True),'Use <Video 1>.')
        with self.assertRaises(ValueError):rules.translate_prompt('<Picture 9>',2,4,False)
        with self.assertRaises(ValueError):rules.translate_prompt('<Video 2>',2,4,False)
        with self.assertRaises(ValueError):rules.translate_prompt('<Audio 1>',2,4,False)
    def test_ph3_prefix_visible(self):
        self.assertEqual(rules.translate_prompt('Change the shirt.',1,0,False,True),'Edit <Picture 1>. Change the shirt.')
        text='subject_definitions:\nA person from <Video 1>.'
        self.assertTrue(rules.translate_prompt(text,0,3,False,True).startswith('subject_definitions:'))
    def test_modes(self):
        self.assertEqual(rules.resolve_mode('Auto',[],True),'References')
        self.assertEqual(rules.resolve_mode('Auto',[],False),'Generate')
        self.assertEqual(rules.resolve_mode('Auto',[1],False),'Edit')
        self.assertEqual(rules.resolve_mode('Replace person',[1],True),'Replace person')
        with self.assertRaises(ValueError):rules.resolve_mode('Replace person',[2],True)
        with self.assertRaises(ValueError):rules.resolve_mode('Generate',[1],False)
    def test_geometry(self):
        for aspect in rules.ASPECTS:
            a,b=map(int,aspect.split(':'))
            for mp in (.25,2.5,4.,8.):
                w,h=rules.dimensions(a,b,mp)
                self.assertEqual(w%32+h%32,0)
                self.assertLess(abs(w/h-a/b)/(a/b),.06)
        with self.assertRaises(ValueError):rules.dimensions(100,100,float('nan'))
    def test_times(self):
        self.assertEqual(rules.frame_times(4,4),[.5,1.5,2.5,3.5])
        self.assertEqual(rules.frame_times(3.1,4,'0.2, 1.5, 2.4'),[.2,1.5,2.4])
        for text in ('-1','3.1','nan','1,1','abc'):
            with self.assertRaises(ValueError):rules.frame_times(3.1,4,text)
    def test_crop(self):
        self.assertEqual(rules.crop_box(200,100,25,0,25,0),(50,0,150,100))
        with self.assertRaises(ValueError):rules.crop_box(200,100,50,0,50,0)
    def test_pins(self):
        m=json.loads((P/'manifest.json').read_text())
        self.assertFalse(m['gpu_tested'])
        self.assertEqual(len(m['comfy_commit']),40)
        names=set()
        for row in m['models']:
            self.assertRegex(row['revision'],'^[0-9a-f]{40}$')
            if row['sha256']:self.assertRegex(row['sha256'],'^[0-9a-f]{64}$')
            self.assertNotIn(row['destination'],names);names.add(row['destination'])
        for engine in rules.ENGINES:
            for precision in rules.PRECISIONS:
                for mode in ('Generate','References'):
                    for name in rules.model_names(engine,precision,mode):
                        self.assertTrue(any(Path(n).name==name for n in names) or name in (
                            'minimax_h3_fl2va_pruned_int8_convrot.safetensors','minimax_h3_ref2va_pruned_int8_convrot.safetensors'))

class GraphTests(unittest.TestCase):
    def test_graph_links(self):
        for profile in PROFILES:
            gui,api=build(profile);by_id={n['id']:n for n in gui['nodes']}
            for eid,src,slot,dst,index,kind in gui['links']:
                self.assertIn(eid,by_id[src]['outputs'][slot]['links'])
                self.assertEqual(by_id[dst]['inputs'][index]['link'],eid)
                self.assertEqual(api[str(dst)]['inputs'][by_id[dst]['inputs'][index]['name']],[str(src),slot])
            self.assertEqual(gui['last_link_id'],len(gui['links']))
            self.assertEqual(api['10']['inputs']['precision'],'INT8 HQ')
            if not profile.get('preflight'):self.assertFalse(api['15']['inputs']['lora_1_enabled'])
    def test_no_multiple_backends(self):
        for profile in PROFILES:
            gui,api=build(profile)
            self.assertLessEqual(sum(n['class_type']=='H3VRender' for n in api.values()),1)
    def test_api_widgets(self):
        for profile in PROFILES:
            _,api=build(profile)
            for node in api.values():self.assertNotIn('control_after_generate',node['inputs'])
            self.assertEqual(api['14']['inputs']['report'],['12',2])
    def test_all_modes_same_selector(self):
        for profile in PROFILES:
            _,api=build(profile)
            self.assertEqual(api['10']['class_type'],'H3VSettings')
    def test_topological(self):
        for profile in PROFILES:
            _,api=build(profile);seen=set()
            def visit(k,stack):
                self.assertNotIn(k,stack)
                if k in seen:return
                for value in api[k]['inputs'].values():
                    if isinstance(value,list):visit(value[0],stack|{k})
                seen.add(k)
            for key in api:visit(key,set())

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg unavailable')
class VideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.path=cls.root/'fixture.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x192:rate=24',
                        '-t','1.5','-c:v','libx264','-pix_fmt','yuv420p','-y',str(cls.path)],check=True,timeout=30)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_native_video(self):
        v=media.extract(self.path,0,1.5,False,native_mp=.1)
        self.assertEqual(v['frames'].shape[0]%17,5)
        self.assertEqual(v['summary']['fps'],24)
        self.assertEqual(len(v['keyframes']),0)
    def test_source_keyframes(self):
        v=media.extract(self.path,0,1.5,True,count=3,keyframe_mp=2.5)
        self.assertEqual(len(v['keyframes']),3)
        self.assertEqual(v['keyframes'][0].shape,(1,192,320,3))
        self.assertIsNone(v['frames'])
    def test_source_crop(self):
        v=media.extract(self.path,0,1.5,True,count=2,crop=(25,0,25,0))
        self.assertEqual(v['keyframes'][0].shape,(1,192,160,3))
    def test_path_guards(self):
        self.assertEqual(media.checked_path('fixture.mp4',self.root),self.path)
        for name in ('../fixture.mp4','/etc/passwd','fake.mp4'):
            with self.assertRaises(ValueError):media.checked_path(name,self.root)
    def test_past_end(self):
        with self.assertRaises(ValueError):media.extract(self.path,4,1,True)

class PlanTests(unittest.TestCase):
    def test_actual_plan_keyframe_budget_and_order(self):
        import torch
        settings=H3VSettings().configure(engine='P-H3 Image',precision='INT8 HQ',mode='References',aspect_ratio='2:3',
            output_megapixels=.25,photo_cap_megapixels=4.,reference_size='max',video_strategy='Auto',standard_steps=50,eros_steps=8,protect_background=False)[0]
        image=torch.zeros(1,64,64,3)
        v={'frames':None,'keyframes':[image]*4,'summary':{'timestamps_seconds':[0.,1.,2.,3.]}}
        plan,preview,report=H3VPlan().prepare(settings,'subject_definitions:\nPerson from <Video 1>.',video=v,
            reference_2={'image':image,'name':'face.png'},reference_3={'image':image,'name':'body.png'})
        self.assertEqual(len(plan['images']),6)
        self.assertIn('<Picture 3>',plan['prompt'])
        self.assertNotIn('<Video',plan['prompt'])
        self.assertEqual(json.loads(report)['reference_mapping'][0]['slot'],2)
        self.assertEqual(preview.shape[0],1)
    def test_native_ph3_rejected(self):
        import torch
        settings=dict(engine='P-H3 Image',mode='References')
        with self.assertRaises(ValueError):H3VPlan().prepare(settings,'person',video={'frames':torch.zeros(5,64,64,3)})
    def test_exact_background_pixels(self):
        import torch
        original=torch.rand(1,64,64,3); mask=torch.zeros(1,64,64)
        mask[:,:,20:40]=1.;mask[:,:,40:45]=.5
        result=finish_image(dict(protect_background=True,original=original,original_mask=mask),torch.zeros_like(original))
        self.assertTrue(torch.equal(result[mask==0],original[mask==0]))
        self.assertTrue(torch.equal(result[mask==1],torch.zeros_like(result[mask==1])))
        self.assertTrue(torch.equal(result[mask==.5],original[mask==.5]*.5))
    def test_empty_preview(self):
        self.assertEqual(tuple(contact_sheet([],[]).shape),(1,32,32,3))

if __name__=='__main__':unittest.main()
