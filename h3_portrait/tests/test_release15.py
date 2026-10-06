"""H3 Portrait 1.5 regressions: role routing, H3 stills, serialization and packaging.

No model download, personal LoRA list, live Ollama call or GPU inference is used.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from node import logic,ollama_client as oc,reference_roles as rr,still_portrait as sp
from install_workflows import install
from smoke_check import validate_graph
from test_thinking import analysis,director
from test_split_models import pipeline,packet


def three():
    value=analysis(3)
    roles=['face','body','pose_camera']
    uses=['facial identity only','body proportions only','eye-level three-quarter pose, medium framing, one hand on hip']
    for row,role,use in zip(value['references'],roles,uses):
        row['role']=role;row['use_for']=use
    return value


def uploads(roles=('auto','auto','auto')):
    return [{'filename':f'source_{i}.png','image':object(),'sha256':str(i),'role':r} for i,r in enumerate(roles,1)]


class Routing15(unittest.TestCase):
    def test_video_pose_is_text_only(self):
        out=rr.route_analysis(three(),uploads())
        self.assertEqual(out['_routing']['native_source_indices'],[0,1])
        self.assertEqual(out['_routing']['ledger'][2]['delivery'],'text_only')
    def test_video_scene_remains_visual(self):
        a=three();a['references'][2]['role']='scene'
        out=rr.route_analysis(a,uploads())
        self.assertEqual(out['_routing']['native_source_indices'],[0,1,2])
    def test_still_safe_swap_withholds_pose_and_scene_pixels(self):
        for role in ('pose_camera','scene','expression'):
            with self.subTest(role=role):
                a=three();a['references'][2]['role']=role
                out=rr.route_analysis(a,uploads(),'Still safe swap (pose/scene text-only)')
                self.assertEqual(out['_routing']['native_source_indices'],[0,1])
                self.assertEqual(out['_routing']['ledger'][2]['delivery'],'text_only')
    def test_still_all_visual_comparison_keeps_guide_pixels(self):
        a=three();a['references'][2]['role']='scene'
        out=rr.route_analysis(a,uploads(),'All images visual (comparison)')
        self.assertEqual(out['_routing']['native_source_indices'],[0,1,2])
    def test_face_and_body_role_limits_are_explicit(self):
        text=logic.subject_definitions(rr.route_analysis(three(),uploads())['references']).lower()
        self.assertIn('facial',text);self.assertIn('body proportions',text)
        self.assertIn('jewelry',text)
    def test_guide_is_not_a_new_identity(self):
        a=three();a['references'][2]['subject']='Pose model'
        out=rr.route_analysis(a,uploads(),'Still safe swap (pose/scene text-only)')
        self.assertEqual(out['guidance'][0]['target_subject'],'Person A')
    def test_picture_numbers_remap_after_filter(self):
        self.assertEqual(rr.remap_text('Image 3 guides Image 1',{1:1,2:2,3:None}),
                         'text-only guide from source image 3 guides <Picture 1>')


class H3StillGeometry15(unittest.TestCase):
    def setUp(self):
        import torch
        self.portrait=torch.zeros(1,1600,900,3)
        self.landscape=torch.zeros(1,900,1600,3)
    def test_native_9_16_is_about_one_megapixel_and_five_frames(self):
        g=sp.geometry('9:16','Native detail (~1 MP)','High fidelity',self.portrait)
        self.assertEqual((g['width'],g['height'],g['output_width'],g['output_height']),(768,1344,756,1344))
        self.assertEqual((g['length'],g['steps'],g['ref_image_size'],g['sampler'],g['scheduler']),(5,20,'max','res_multistep','simple'))
    def test_high_res_is_native_h3_canvas_not_post_upscale(self):
        low=sp.geometry('9:16','Native detail (~1 MP)','High fidelity',self.portrait)
        high=sp.geometry('9:16','High-res (~2 MP, experimental)','High fidelity',self.portrait)
        self.assertGreater(high['width']*high['height'],low['width']*low['height']*1.8)
        self.assertEqual(high['width']%32,0);self.assertEqual(high['height']%32,0)
        self.assertAlmostEqual(high['width']*high['height']/1048576,2.0,delta=.08)
    def test_quality_presets(self):
        self.assertEqual(sp.geometry('1:1','Native detail (~1 MP)','Standard',self.portrait)['steps'],16)
        self.assertEqual(sp.geometry('1:1','Native detail (~1 MP)','Fast preview',self.portrait)['steps'],12)
        self.assertEqual(sp.geometry('1:1','Native detail (~1 MP)','Standard',self.portrait)['ref_image_size'],'match')
    def test_match_aspect_prefers_pose_or_scene_guide(self):
        routing={'ledger':[{'source_image':1,'role':'face','delivery':'visual'},{'source_image':2,'role':'pose_camera','delivery':'text_only'}],
                 'native_source_indices':[0]}
        self.assertEqual(sp.aspect_source_index([{'image':self.portrait},{'image':self.landscape}],routing),1)
        g=sp.geometry('Match guide/primary','Native detail (~1 MP)','High fidelity',self.landscape)
        self.assertGreater(g['output_width'],g['output_height'])
    def test_exact_crop_and_stable_selector(self):
        import torch
        g=sp.geometry('9:16','Native detail (~1 MP)','High fidelity',self.portrait)
        frames=torch.zeros(5,g['height'],g['width'],3)
        cropped=sp.crop_frames(frames,g)
        self.assertEqual(tuple(cropped.shape),(5,1344,756,3))
        # Make middle frame sharply alternating while keeping neighbors similar.
        pattern=(torch.arange(1344)[:,None]+torch.arange(756)[None,:])%2
        cropped[2,:,:,0]=pattern
        image,index,score=sp.select_frame(cropped,'Best stable quality')
        self.assertEqual(index,2);self.assertEqual(tuple(image.shape),(1,1344,756,3));self.assertGreater(score,0)
    def test_prompt_is_static_h3_ref2va_instruction(self):
        routed=rr.route_analysis(three(),uploads(),'Still safe swap (pose/scene text-only)')
        obj=director(2);obj['references']=routed['references'];obj['_routing']=routed['_routing'];obj['_guidance']=routed['guidance']
        g=sp.geometry('9:16','Native detail (~1 MP)','High fidelity',self.portrait)
        text=sp.format_prompt(obj,'Image 3 controls the pose; Image 1 is identity.',g)
        self.assertIn('DELIVERY: ONE finished still image',text)
        self.assertIn('five-frame Ref2VA packet',text)
        self.assertNotIn('<Picture 3>',text)
        self.assertIn('STATIC GUIDE TRANSFER',text)


class H3StillOllama15(unittest.TestCase):
    @patch.object(oc,'preview',return_value='encoded')
    @patch.object(oc,'unload')
    def test_all_three_seen_by_ollama_but_guide_removed_before_h3_map(self,unload,preview):
        cfg=json.loads((ROOT/'profiles/lite/settings.json').read_text())
        cfg=dict(cfg,reference_mode='Still safe swap (pose/scene text-only)')
        info={'model':cfg['ollama_model'],'digest':'d','capabilities':['vision'],
              'analysis':{'model':cfg['analysis_model'],'digest':'a','capabilities':['vision']}}
        s=MagicMock();s.post.side_effect=[packet(three()),packet(director(2))]
        out=oc.generate(s,cfg,info,sp.prompt_policy()['still_system_prompt.txt'],'make one still',uploads(('face','body','pose_camera')),sp.analysis_recipe('9:16','High fidelity'),'',0)
        calls=[c.kwargs['json'] for c in s.post.call_args_list]
        self.assertEqual(len(calls[0]['messages'][1]['images']),3)
        self.assertFalse(any('images' in m for m in calls[1]['messages']))
        self.assertEqual(out['_routing']['native_source_indices'],[0,1])
        self.assertIn('text-only',calls[1]['messages'][1]['content'])
        unload.assert_called_once()


class Packaging15(unittest.TestCase):
    def test_lite_assets_are_h3_only(self):
        cfg=json.loads((ROOT/'profiles/lite/settings.json').read_text())
        self.assertNotIn('image_files',cfg)
        rows=json.loads((ROOT/'profiles/lite/models.json').read_text())
        self.assertTrue(rows);self.assertTrue(all(r['profile']=='h3' for r in rows))
        self.assertTrue(all('qwen_image_edit' not in json.dumps(r).lower() for r in rows))
    def test_runtime_maps_all_workflows_to_h3(self):
        runtime=json.loads((ROOT/'runtime.json').read_text())
        self.assertEqual(set(runtime['profiles']),{'h3'})
        self.assertTrue(all(v=='h3' for v in runtime['workflow_profiles'].values()))
    def test_still_workflow_is_native_h3_not_separate_image_model(self):
        g=json.loads((ROOT/'workflows/H3_Portrait_Image_Lite.json').read_text());validate_graph(g)
        types={n['type'] for n in g['nodes']}
        for name in ('H3PortraitStillDirector','H3PortraitModels','H3PortraitConditioning','H3PortraitSampler','VAEDecode','H3PortraitStillOutput','SaveImage'):
            self.assertIn(name,types)
        self.assertFalse(any('QwenGenerate' in x or 'QwenModels' in x or x=='H3PortraitImageDirector' for x in types))
        director_node=next(n for n in g['nodes'] if n['type']=='H3PortraitStillDirector')
        vals=director_node['widgets_values']
        self.assertEqual(vals[2],'High-res (~2 MP, experimental)');self.assertEqual(vals[5],'fixed');self.assertEqual(vals[7],'Draft only');self.assertEqual(vals[-1],'Still safe swap (pose/scene text-only)')
        self.assertFalse(any(isinstance(v,float) and v!=v for v in vals))
    def test_installer_installs_still_only_on_lite(self):
        for profile,path,count in [('Full',ROOT/'settings.json',3),('Lite',ROOT/'profiles/lite/settings.json',4)]:
            with self.subTest(profile=profile),tempfile.TemporaryDirectory() as td:
                install(td,json.loads(path.read_text()))
                files=list((Path(td)/'user/default/workflows').glob('*.json'))
                self.assertEqual(len(files),count)
                names={p.name for p in files}
                self.assertEqual(any('H3_Portrait_Image_' in n for n in names),profile=='Lite')
                for f in files:validate_graph(json.loads(f.read_text()))
    def test_no_removed_qwen_image_code_is_tracked(self):
        for name in ('image_portrait.py','image_analysis_prompt.txt','image_system_prompt.txt'):
            self.assertFalse((ROOT/'node'/name).exists())
        self.assertFalse(hasattr(oc,'generate_image'))
    def test_prepare_assets_uses_h3_profile_only(self):
        text=(ROOT/'prepare_assets.py').read_text()
        self.assertIn("profiles={'h3'}",text);self.assertNotIn("profiles.add('portrait_image')",text)
    def test_release_verifier_accepts_current_source(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('verify',ROOT/'verify_release.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertEqual(m.verify_source(ROOT)['version'],'1.5.3')

if __name__=='__main__':unittest.main()
