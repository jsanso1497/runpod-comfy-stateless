import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

HERE=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
N=load('krea_node_tests',HERE/'node/__init__.py')
C=load('krea_graph_tests',HERE/'check.py')
I=load('krea_install_tests',HERE/'install.py')

def photo(h=64,w=48,value=0.5):return torch.full((1,h,w,3),value,dtype=torch.float32)
def graph(which='01'):return json.loads(next((HERE/'workflows').glob('*_'+which+'_*')).read_text())
def node(g,kind):return next(n for n in g['nodes'] if n['type']==kind)

class ImageHelpers(unittest.TestCase):
    def test_canvas_obeys_mp_cap_and_grid(self):
        for w,h in [(1080,1920),(6000,4000),(2048,2048),(3000,1000)]:
            for mp in [0.25,0.5,1.0,1.5,2.0]:
                ow,oh=N.canvas_size(w,h,mp)
                self.assertEqual(ow%16,0);self.assertEqual(oh%16,0)
                self.assertLessEqual(ow*oh,mp*1_000_000)
                self.assertLess(abs((ow/oh)/(w/h)-1),0.07)
    def test_canvas_rejects_excess_mp(self):
        for val in [0,2.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):N.canvas_size(1080,1920,val)
    def test_panorama_rejected(self):
        with self.assertRaises(ValueError):N.canvas_size(8000,500,1.5)
    def test_single_photo_required(self):
        with self.assertRaises(ValueError):N.image_check(torch.zeros(2,32,32,3))
    def test_bad_channels_rejected(self):
        with self.assertRaises(ValueError):N.image_check(torch.zeros(1,32,32,1))
    def test_nan_pixels_rejected(self):
        a=photo();a[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):N.image_check(a)
    def test_resize_preserves_range(self):
        a=torch.rand(1,32,48,3);r=N.resize(a,73,57)
        self.assertEqual(tuple(r.shape),(1,57,73,3));self.assertGreaterEqual(r.min(),0);self.assertLessEqual(r.max(),1)
    def test_letterbox_does_not_crop(self):
        a=photo(32,64,1);r=N.fit_inside(a,64,64)
        torch.testing.assert_close(r[:,16:48],a)
        self.assertTrue(bool((r[:,:16]==0.5).all()))
    def test_sheet_keeps_both_originals(self):
        face,body=photo(32,32,1),photo(64,32,0)
        r=N.reference_sheet(face,body,'Face + body sheet')
        self.assertEqual(tuple(r.shape),(1,1280,1280,3))
        self.assertEqual(r[0,640,320,0],1);self.assertEqual(r[0,640,960,0],0)
    def test_face_only_returns_original_pixels(self):
        face=photo();self.assertEqual(N.reference_sheet(face,photo(),'Face only').data_ptr(),face.data_ptr())
    def test_body_only_returns_original_pixels(self):
        body=photo();self.assertEqual(N.reference_sheet(photo(),body,'Body only').data_ptr(),body.data_ptr())
    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):N.reference_sheet(photo(),photo(),'inferred identity')
    def test_prepare_has_one_scene_one_subject(self):
        result=N.KreaIdentityPrepare().prepare(photo(),photo(),photo(),'Face only','Keep scene clothing',0.25,'Keep the chair.')
        self.assertEqual(len(result),5);self.assertIn('Keep the chair.',result[2]);self.assertLessEqual(result[3]*result[4],250000)
    def test_prompt_roles_not_reversed(self):
        p=N.instruction('Face + body sheet','Keep scene clothing','')
        self.assertIn('person in image 1 with the person in image 2',p)
        self.assertIn('LEFT panel',p);self.assertIn('RIGHT panel',p);self.assertIn('clothing in image 1',p)
    def test_clothing_toggle_explicit(self):
        p=N.instruction('Face + body sheet','Use body reference clothing','')
        self.assertIn('clothing shown in the body reference',p)
        with self.assertRaisesRegex(ValueError,'does not send a body reference'):
            N.instruction('Face only','Use body reference clothing','')
    def test_pair_requires_instruction(self):
        with self.assertRaises(ValueError):N.KreaIdentityPair().prepare(photo(),photo(),0.25,' ')
    def test_switch_off_is_exact_passthrough(self):
        a,b,c,d=object(),object(),object(),object()
        out=N.KreaIdentityRebalanceControl().select(a,b,c,d,False)
        self.assertIs(out[0],a);self.assertIs(out[1],b)
    def test_switch_on_returns_real_rebalanced_objects(self):
        a,b,c,d=object(),object(),object(),object()
        out=N.KreaIdentityRebalanceControl().select(a,b,c,d,True)
        self.assertIs(out[0],c);self.assertIs(out[1],d)
    def test_string_boolean_rejected(self):
        with self.assertRaises(ValueError):N.KreaIdentityRebalanceControl().select([],[],[],[],'false')
    def test_crop_coordinates(self):
        im=photo(100,200);crop,g=N.KreaIdentityCrop().crop(im,0.25,0.1,0.5,0.5)
        self.assertEqual(tuple(crop.shape),(1,50,100,3));self.assertEqual((g['x'],g['y']),(50,10))
    def test_crop_clamped_to_edges(self):
        _,g=N.KreaIdentityCrop().crop(photo(100,100),0.5,0.5,0.9,0.9)
        self.assertEqual((g['width'],g['height']),(50,50))
    def test_crop_invalid_values(self):
        for vals in [(-1,0,0.5,0.5),(0,0,float('nan'),0.5),(0,0,0,0.5)]:
            with self.assertRaises(ValueError):N.KreaIdentityCrop().crop(photo(),*vals)
    def test_stitch_preserves_outside_exactly(self):
        im=torch.rand(1,128,128,3);before=im.clone();_,g=N.KreaIdentityCrop().crop(im,0.25,0.25,0.5,0.5)
        out=N.KreaIdentityStitch().stitch(im,photo(64,64,1),g,8)[0]
        mask=torch.ones(128,128,dtype=torch.bool);mask[32:96,32:96]=False
        self.assertTrue(torch.equal(out[:,mask],im[:,mask]));self.assertTrue(torch.equal(im,before))
        self.assertEqual(out[0,64,64,0],1)
    def test_stitch_no_feather(self):
        im=photo(128,128,0);_,g=N.KreaIdentityCrop().crop(im,0.25,0.25,0.5,0.5)
        out=N.KreaIdentityStitch().stitch(im,photo(32,32,1),g,0)[0]
        torch.testing.assert_close(out[:,32:96,32:96],torch.ones_like(out[:,32:96,32:96]),rtol=1e-5,atol=1e-6)
    def test_stitch_wrong_original_size_rejected(self):
        _,g=N.KreaIdentityCrop().crop(photo(128,128),0.25,0.25,0.5,0.5)
        with self.assertRaises(ValueError):N.KreaIdentityStitch().stitch(photo(256,256),photo(),g,8)
    def test_upscale_dimensions(self):
        short,cap=N.KreaIdentityUpscaleSize().size(photo(1536,1024),2,4096)
        self.assertEqual((short,cap),(2048,4096))
    def test_upscale_cap(self):
        short,cap=N.KreaIdentityUpscaleSize().size(photo(1536,1024),4,4096)
        self.assertEqual(cap,4096);self.assertEqual(short,2731)
    def test_upscale_invalid_scale(self):
        with self.assertRaises(ValueError):N.KreaIdentityUpscaleSize().size(photo(),float('nan'),4096)

class Workflows(unittest.TestCase):
    def test_all_graphs_validate(self):self.assertEqual(len(C.static()),8)
    def test_broken_graph_rejected(self):
        g=graph();g['links'][0][5]='LATENT'
        with self.assertRaises(ValueError):C.validate_graph(g)
    def test_wrong_reference_order_rejected(self):
        g=graph();n=node(g,'Krea2EditGroundedEncode')
        x=next(i for i in n['inputs'] if i['name']=='image');x['link']=None
        with self.assertRaises(ValueError):C.validate_graph(g)
    def test_wrong_seed_serialization_rejected(self):
        g=graph();node(g,'KSampler')['widgets_values'].pop(1)
        with self.assertRaises(ValueError):C.validate_graph(g)
    def test_rebalance_not_automatically_on(self):
        for which in ['01','02','03']:self.assertFalse(node(graph(which),'KreaIdentityRebalanceControl')['widgets_values_named']['enabled'])
    def test_screenshot_weights_exact(self):
        for which in ['01','02','03']:
            self.assertEqual(node(graph(which),'ConditioningKrea2Rebalance')['widgets_values_named'],{'multiplier':1.0,'per_layer_weights':C.WEIGHTS})
    def test_only_general_editor_lora(self):
        g=graph();ls=[n for n in g['nodes'] if 'LoRA' in n['type'] or 'LoraLoader' in n['type']]
        self.assertEqual(len(ls),1);self.assertEqual(ls[0]['type'],'LoraLoaderModelOnly')
    def test_no_llm_or_video_path_in_edit(self):
        for which in ['01','02','03']:
            for n in graph(which)['nodes']:
                self.assertFalse(any(v in n['type'] for v in ('Ollama','H3','Video','SeedVR')))
    def test_single_image_upscale(self):
        g=graph('04');self.assertEqual(node(g,'SeedVR2VideoUpscaler')['widgets_values_named']['batch_size'],1)
    def test_local_schema_matches_workflow(self):
        registry={name:{'input':cls.INPUT_TYPES(),'output':list(cls.RETURN_TYPES)} for name,cls in N.NODE_CLASS_MAPPINGS.items()}
        for p in (HERE/'workflows').glob('*.json'):
            g=json.loads(p.read_text());g['nodes']=[n for n in g['nodes'] if n['type'] in registry]
            C.validate_schema(g,registry)
    def test_empty_registry_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing installed node'):C.validate_schema(graph(),{})

class Installation(unittest.TestCase):
    def test_runtime_default_does_not_download(self):
        with patch.dict(os.environ,{},clear=True):self.assertEqual(I.runtime('/nonexistent','/nonexistent'),[])
    def test_invalid_env_rejected(self):
        with patch.dict(os.environ,{'ENABLE_KREA_IDENTITY':'yes'}):
            with self.assertRaises(ValueError):I.flag('ENABLE_KREA_IDENTITY')
    def test_upscale_requires_editor(self):
        with patch.dict(os.environ,{'ENABLE_KREA_IDENTITY':'0','KREA_IDENTITY_UPSCALE':'1'}):
            with self.assertRaises(ValueError):I.runtime('/nonexistent','/nonexistent')
    def test_install_respects_user_workflow_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            files=I.install_workflows(tmp,False);self.assertEqual(len(files),7)
            files[0].write_text('user-edited')
            I.install_workflows(tmp,False);self.assertEqual(files[0].read_text(),'user-edited')
    def test_optional_upscale_is_installed_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(I.install_workflows(tmp,True)),8)
    def test_only_two_new_upstream_dependencies(self):
        deps=json.loads((HERE/'config/custom_nodes.json').read_text())
        self.assertEqual({r['name'] for r in deps},{'comfyui-krea2edit','Rebalance-Pack'})
        self.assertTrue(all(len(r['ref'])==40 for r in deps))
    def test_hash_pinned_models_and_full_editor(self):
        models=json.loads((HERE/'config/models.json').read_text())
        loras=json.loads((HERE/'config/loras.json').read_text())
        self.assertEqual(len(loras),1)
        for r in models+loras:
            self.assertEqual(len(r['sha256']),64)
            self.assertNotIn('token=',json.dumps(r))
            self.assertNotIn('fp8',r['filename'])
        turbo=next(r for r in models if 'turbo' in r['filename'])
        self.assertEqual(turbo['sha256'],'78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3')
    def test_shared_lora_list_not_read_or_changed(self):
        text=(HERE/'install.py').read_text()
        self.assertNotIn('lora_links.txt',text)

if __name__=='__main__':unittest.main()
