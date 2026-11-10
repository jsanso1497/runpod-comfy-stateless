import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
import control
import build_workflows as builder
spec=importlib.util.spec_from_file_location('h3_image_helpers_ready',PACKAGE/'nodes/H3ImageWorkspace/__init__.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
HAS_TORCH=importlib.util.find_spec('torch') is not None


class GraphTests(unittest.TestCase):
    def test_six_workflows(self):
        self.assertEqual(len(builder.PROFILES),6)
    def test_blank_literal_prompts(self):
        for profile in builder.PROFILES:
            g,api=builder.build(profile)
            self.assertEqual(api['40']['inputs']['prompt'],'')
            self.assertEqual(h.H3ImageManualConditioning.INPUT_TYPES()['required']['prompt'][1]['dynamicPrompts'],False)
    def test_alternative_actual_order(self):
        g,api=builder.build(builder.PROFILES[5])
        self.assertEqual(api['40']['inputs']['reference_1'],['1',0])
        self.assertEqual(api['40']['inputs']['reference_3'],['3',0])
        self.assertEqual(api['40']['inputs']['aspect_source_slot'],3)
        self.assertTrue(api['40']['inputs']['follow_image_1_aspect'])
        self.assertEqual(api['49']['inputs']['reference'],['3',0])
    def test_before_after_source(self):
        for i in (1,3):
            g,api=builder.build(builder.PROFILES[i])
            self.assertEqual(api['49']['inputs']['reference'],['1',0])
    def test_original_png_independent(self):
        for profile in builder.PROFILES:
            g,api=builder.build(profile)
            self.assertEqual(api['46']['inputs']['images'],['45',0])
    def test_report_waits_for_decode(self):
        for profile in builder.PROFILES:
            g,api=builder.build(profile)
            self.assertEqual(api['47']['inputs']['report'],['45',1])
    def test_all_seed_fixed(self):
        for profile in builder.PROFILES:
            g,_=builder.build(profile)
            n=next(n for n in g['nodes'] if n['id']==43)
            self.assertEqual(n['widgets_values_named']['control_after_generate'],'fixed')
    def test_no_video_pipeline(self):
        for profile in builder.PROFILES:
            _,api=builder.build(profile)
            names=' '.join(n['class_type'] for n in api.values())
            for word in ('Fizgig','PH3Image','Lora','Video','Enhance','PromptBuilder'):
                self.assertNotIn(word,names)
    def test_no_cycles(self):
        for profile in builder.PROFILES:
            _,api=builder.build(profile);seen=set();stack=set()
            def visit(k):
                self.assertNotIn(k,stack)
                if k in seen:return
                stack.add(k)
                for v in api[k]['inputs'].values():
                    if isinstance(v,list) and len(v)==2 and isinstance(v[0],str) and v[0] in api:visit(v[0])
                stack.remove(k);seen.add(k)
            for key in api:visit(key)
    def test_report_strings_are_sockets_not_prompt_widgets(self):
        for cls,key in ((h.H3ImageTimedSampler,'reference_report'),(h.H3ImageTimedDecode,'sampling_report'),(h.H3ImageReport,'report')):
            self.assertTrue(cls.INPUT_TYPES()['required'][key][1]['forceInput'])
    def test_runtime_no_duplicate_launch(self):
        text=(PACKAGE/'runtime/backend.py').read_text()
        self.assertIn('fcntl.LOCK_EX | fcntl.LOCK_NB',text)
        self.assertIn('os.set_inheritable(lock_fd, True)',text)
    def test_manager_and_cache_safety(self):
        text=(PACKAGE/'runtime/backend.py').read_text()
        self.assertIn('--enable-manager',text)
        self.assertIn('--disable-pinned-memory',text)
        self.assertIn('--cache-none',text)
        self.assertNotIn('security_level = weak',text)
    def test_new_workflow_folder(self):
        self.assertIn('00_H3_Image_Manual_v2',(PACKAGE/'runtime/backend.py').read_text())


class ControlTests(unittest.TestCase):
    def test_active_protected(self):
        with self.assertRaises(RuntimeError):control.assert_idle({'queue_running':[1]})
    def test_pending_protected(self):
        with self.assertRaises(RuntimeError):control.assert_idle({'queue_pending':[1]})
    def test_explicit_interrupt(self):
        control.assert_idle({'queue_running':[1],'queue_pending':[2]},True)
    def test_idle_allowed(self):control.assert_idle({'queue_running':[],'queue_pending':[]})
    def test_zip_streaming_no_original_change(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'output/sub';out.mkdir(parents=True)
            (out/'file.png').write_bytes(b'not a real png')
            p=control.zip_outputs(root)
            self.assertEqual((out/'file.png').read_bytes(),b'not a real png')
            with zipfile.ZipFile(p) as z:
                self.assertEqual(z.namelist(),['output/sub/file.png'])
                self.assertEqual(z.read('output/sub/file.png'),b'not a real png')
                self.assertEqual(z.testzip(),None)
    def test_zip_skips_symlinks(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'output';out.mkdir()
            (root/'private.txt').write_text('private')
            (out/'ok.txt').write_text('ok');(out/'no.txt').symlink_to(root/'private.txt')
            p=control.zip_outputs(root)
            with zipfile.ZipFile(p) as z:self.assertEqual(z.namelist(),['output/ok.txt'])
    def test_zip_unique_not_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'output';out.mkdir();(out/'a').write_bytes(b'a')
            a=control.zip_outputs(root);b=control.zip_outputs(root)
            self.assertNotEqual(a,b);self.assertTrue(a.exists() and b.exists())
    def test_zip_empty_stops(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'output').mkdir()
            with self.assertRaises(RuntimeError):control.zip_outputs(root)
    def test_zip_disk_space(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'output';out.mkdir();(out/'a').write_bytes(b'a')
            with patch.object(control.shutil,'disk_usage',return_value=types.SimpleNamespace(free=0)):
                with self.assertRaises(RuntimeError):control.zip_outputs(root)
    def test_zip_rejects_output_symlink(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'elsewhere').mkdir();(root/'output').symlink_to(root/'elsewhere',target_is_directory=True)
            with self.assertRaises(RuntimeError):control.zip_outputs(root)


@unittest.skipUnless(HAS_TORCH,'Numerical tests run after torch installation in image')
class NumericalTests(unittest.TestCase):
    def test_exact_prompt_and_scene_last_geometry(self):
        import torch
        captured={}
        def short_edge(w,hh,low,high):
            scale=min(max(min(w,hh),low),high)/min(w,hh)
            return max(32,round(w*scale/32)*32),max(32,round(hh*scale/32)*32)
        def encode(**kw):
            captured.update(kw)
            return ([[torch.zeros(1,1,1),{}]], {'samples':torch.zeros(1,24,1,kw['height']//16,kw['width']//16)})
        module=types.ModuleType('comfy_extras.nodes_minimax_h3')
        module.MiniMaxH3ImageConditioning=types.SimpleNamespace(execute=encode)
        module.IMAGE_REF_SHORT_EDGE=1024;module.IMAGE_REF_VAE_MAX_SHORT_EDGE=2048;module._size_short_edge=short_edge
        parent=types.ModuleType('comfy_extras');parent.nodes_minimax_h3=module
        refs={'reference_1':{'image':torch.zeros(1,64,32,3),'file':'body.png'},
              'reference_2':{'image':torch.zeros(1,32,32,3),'file':'face.png'},
              'reference_3':{'image':torch.zeros(1,90,160,3),'file':'scene.png'}}
        prompt='  literal {text} <Picture 3>  '
        with patch.dict(sys.modules,{'comfy_extras':parent,'comfy_extras.nodes_minimax_h3':module}):
            cond,latent,report=h.H3ImageManualConditioning().encode(None,None,prompt,'1:1',4.,True,3,aspect_source_slot=3,**refs)
        self.assertEqual(captured['prompt'],prompt)
        self.assertLess(abs(captured['width']/captured['height']-16/9),.03)
        self.assertIs(captured['images']['image_1'],refs['reference_1']['image'])
        data=json.loads(report)
        self.assertEqual(data['references'][1]['vision_size'],[1024,1024])
        self.assertEqual(data['references'][0]['vae_size'],[1024,2048])
        self.assertEqual(data['reference_count'],3)
    def test_missing_aspect_slot_stops(self):
        import torch
        module=types.ModuleType('comfy_extras.nodes_minimax_h3')
        module.MiniMaxH3ImageConditioning=object
        parent=types.ModuleType('comfy_extras');parent.nodes_minimax_h3=module
        with patch.dict(sys.modules,{'comfy_extras':parent,'comfy_extras.nodes_minimax_h3':module}):
            with self.assertRaises(ValueError):
                h.H3ImageManualConditioning().encode(None,None,'text','1:1',4.,True,1,aspect_source_slot=3,
                  reference_1={'image':torch.zeros(1,32,32,3),'file':'x.png'})
    def test_comparison_contain_not_stretch(self):
        import torch
        import torch.nn.functional as F
        utils=types.ModuleType('comfy.utils')
        utils.common_upscale=lambda image,w,hh,method,crop:F.interpolate(image,size=(hh,w),mode='bilinear',align_corners=False)
        comfy=types.ModuleType('comfy');comfy.utils=utils
        original=torch.ones(1,100,50,3);generated=torch.zeros(1,100,100,3)
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.utils':utils}):
            result=h.H3ImageCompare().compare({'image':original},generated)[0]
        self.assertEqual(tuple(result.shape),(1,100,200,3))
        self.assertTrue(torch.all(result[:,:,:25,:]==.5))
        self.assertTrue(torch.all(result[:,:,25:75,:]==1.))
        self.assertTrue(torch.all(result[:,:,100:,:]==0.))
        self.assertTrue(torch.all(original==1.))
    def test_sampler_passes_parameters_unchanged(self):
        import torch
        captured={}
        sample={'samples':torch.zeros(1,24,1,2,2)}
        def execute(noise,guider,sampler,sigmas,latent_image):
            captured.update(noise=noise,guider=guider,sampler=sampler,sigmas=sigmas,latent_image=latent_image)
            return sample,None
        module=types.ModuleType('comfy_extras.nodes_custom_sampler')
        module.SamplerCustomAdvanced=types.SimpleNamespace(execute=execute)
        parent=types.ModuleType('comfy_extras');parent.nodes_custom_sampler=module
        noise=types.SimpleNamespace(seed=123);sigmas=torch.tensor(h.SIGMAS)
        with patch.dict(sys.modules,{'comfy_extras':parent,'comfy_extras.nodes_custom_sampler':module}):
            value,report=h.H3ImageTimedSampler().sample(noise,'guider','sampler',sigmas,sample,'{}','test')
        self.assertIs(value,sample);self.assertIs(captured['sigmas'],sigmas)
        self.assertEqual(json.loads(report)['seed'],123);self.assertEqual(json.loads(report)['steps'],6)
    def test_decode_calls_native_not_fizgig(self):
        import torch
        image=torch.zeros(1,32,64,3)
        class Decode:
            def decode(self,samples,vae):return (image,)
        module=types.ModuleType('nodes');module.VAEDecode=Decode
        with patch.dict(sys.modules,{'nodes':module}):
            out,report=h.H3ImageTimedDecode().decode({},None,'{}')
        self.assertIs(out,image);self.assertEqual(json.loads(report)['decoded_size'],[64,32])
    def test_schedule_rejects_other_family(self):
        model=types.SimpleNamespace(get_model_object=lambda name:types.SimpleNamespace())
        comfy=types.ModuleType('comfy');samplers=types.ModuleType('comfy.samplers');comfy.samplers=samplers
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.samplers':samplers}):
            with self.assertRaises(ValueError):h.H3ImageSixStepSchedule().prepare(model)
    def test_report_saves_json(self):
        with tempfile.TemporaryDirectory() as t:
            paths=types.ModuleType('folder_paths');paths.get_output_directory=lambda:t
            with patch.dict(sys.modules,{'folder_paths':paths}):
                h.H3ImageReport().show('{"seed":42}',True)
            files=list((Path(t)/'H3_Image/reports').glob('*.json'))
            self.assertEqual(len(files),1);self.assertEqual(json.loads(files[0].read_text()),{'seed':42})

if __name__=='__main__':unittest.main()
