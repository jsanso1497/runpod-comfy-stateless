from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q21_test_module',ROOT/'node/__init__.py',submodule_search_locations=[str(ROOT/'node')])
q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=q;spec.loader.exec_module(q)
g=sys.modules['q21_test_module.geometry']

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec);sys.modules[name]=out;spec.loader.exec_module(out);return out
builder=load_file('q21_builder',ROOT/'make_workflows.py')
download=load_file('q21_downloader',ROOT/'download_models.py')

class GeometryTests(unittest.TestCase):
    def test_budget_grid(self):
        count=0
        for w,h in [(7001,4003),(4003,7001),(2756,2756),(1756,1756),(64,7000),(7000,64),(513,519),(101,17),(4096,2048)]:
            for edge in (1024,1536,2048):
                for long in (2048,3072,4096):
                    with self.subTest(w=w,h=h,edge=edge,long=long):
                        sw,sh,ww,wh=g.budget_dimensions(w,h,edge,long)
                        self.assertEqual(ww%32,0);self.assertEqual(wh%32,0)
                        self.assertLessEqual(ww*wh,edge**2);self.assertLessEqual(max(ww,wh),long)
                        self.assertLessEqual(sw,w);self.assertLessEqual(sh,h)
                        self.assertGreaterEqual(ww,sw);self.assertGreaterEqual(wh,sh)
                        self.assertLess(ww-sw,32);self.assertLess(wh-sh,32)
                        count+=1
        self.assertEqual(count,81)
    def test_1500_mask_example(self):
        self.assertEqual(g.budget_dimensions(1756,1756,2048,3072),(1756,1756,1760,1760))
    def test_2500_mask_example(self):
        self.assertEqual(g.budget_dimensions(2756,2756,2048,3072),(2048,2048,2048,2048))
    def test_dimensions_bad(self):
        for args in [(0,100,2048,3072),(100,100,2047,3072),(100,100,2048,3071)]:
            with self.assertRaises(ValueError):g.budget_dimensions(*args)
    def test_native_grid_does_not_resize(self):
        source=torch.rand(1,128,160,3); mask=torch.ones(1,128,160)
        data,work,_=g.crop_photo(source,mask,0)
        self.assertEqual(work.shape,source.shape);self.assertFalse(data['plan'].resampled)
    def test_work_does_not_alias_source(self):
        source=torch.rand(1,128,160,3);saved=source.clone();mask=torch.ones(1,128,160)
        data,work,_=g.crop_photo(source,mask,0);work.zero_()
        self.assertTrue(torch.equal(source,saved))
    def test_no_mask(self):
        with self.assertRaisesRegex(ValueError,'Empty mask'):g.mask_check(torch.zeros(1,64,64),100,200)
    def test_bad_mask_shape(self):
        with self.assertRaisesRegex(ValueError,'dimensions'):g.mask_check(torch.ones(1,64,64),100,200)
    def test_invalid_mask_values(self):
        for value in (float('nan'),float('inf'),-1.,2.):
            with self.subTest(value=value),self.assertRaises(ValueError):g.mask_check(torch.full((1,4,4),value),4,4)
    def test_roundtrip_protection(self):
        source=torch.rand(1,161,239,3);mask=torch.zeros(1,161,239);mask[:,25:130,50:190]=1
        mask[:,60:80,90:110]=0
        data,work,_=g.crop_photo(source,mask,19)
        before=source.clone(); generated=torch.ones_like(work)
        out,report=g.stitch_photo(data,generated,12)
        self.assertEqual(out.shape,source.shape)
        self.assertTrue(torch.equal(out[mask==0],source[mask==0]));self.assertTrue(torch.equal(source,before))
        self.assertTrue(torch.any(out[mask>0]!=source[mask>0]))
        self.assertEqual(json.loads(report)['outside_mask_changed_channels'],0)
    def test_frame_edge_and_hole(self):
        source=torch.rand(1,65,97,3);mask=torch.ones(1,65,97);mask[:,20:40,20:40]=0
        data,work,_=g.crop_photo(source,mask,0)
        out,_=g.stitch_photo(data,torch.ones_like(work),0)
        self.assertEqual(float(out[0,0,0,0]),1.)
        self.assertTrue(torch.equal(out[mask==0],source[mask==0]))
    def test_padding_is_removed_before_resize(self):
        source=torch.zeros(1,71,111,3);mask=torch.ones(1,71,111)
        data,work,_=g.crop_photo(source,mask,0)
        generated=torch.ones_like(work);generated[:,:71,:111]=.25
        out,_=g.stitch_photo(data,generated,0)
        self.assertTrue(torch.all(out==.25))
    def test_bad_generated_dimensions(self):
        data,work,_=g.crop_photo(torch.zeros(1,64,64,3),torch.ones(1,64,64),0)
        with self.assertRaises(ValueError):g.stitch_photo(data,torch.ones(1,32,32,3))
    def test_bad_generated_values(self):
        data,work,_=g.crop_photo(torch.zeros(1,64,64,3),torch.ones(1,64,64),0)
        with self.assertRaises(ValueError):g.stitch_photo(data,torch.full_like(work,float('nan')))
    def test_reference_does_not_upscale_or_stretch(self):
        original=torch.rand(1,175,291,3);out=g.prepare_reference(original,1536)
        self.assertEqual(out.shape,(1,192,320,3))
        self.assertTrue(torch.equal(out[:,:175,:291],original))
    def test_source_rejects_video(self):
        with self.assertRaises(ValueError):g.image_check(torch.zeros(2,64,64,3))

class MaskTests(unittest.TestCase):
    def test_manual_never_imports_sam(self):
        source=torch.zeros(1,71,111,3);mask=torch.zeros(1,71,111);mask[:,5:10,20:25]=1
        with patch.dict(sys.modules,{'comfy_extras.nodes_sam3':None}):
            out,report=q.Q21PhotoMask().make_mask(source,manual_mask=mask)
        self.assertTrue(torch.equal(out,mask));self.assertFalse(json.loads(report)['auto_mask'])
    def test_missing_manual_mask(self):
        with self.assertRaisesRegex(ValueError,'SAVE'):q.Q21PhotoMask().make_mask(torch.zeros(1,71,111,3),manual_mask=torch.zeros(1,64,64))
    def test_instance_selection(self):
        masks=torch.zeros(2,8,8);masks[0,:4,:4]=1;masks[1,4:,4:]=1
        self.assertEqual(int(q.choose_instances(masks,-1).sum()),32)
        self.assertEqual(int(q.choose_instances(masks,1).sum()),16)
        with self.assertRaises(ValueError):q.choose_instances(masks,2)
    def test_empty_sam_never_falls_back(self):
        for masks in (torch.zeros(0,8,8),torch.zeros(1,8,8)):
            with self.assertRaises(ValueError):q.choose_instances(masks,-1)
    def test_subtract_after_growth_preserves_protected_region(self):
        selected=torch.ones(1,31,31);manual=torch.zeros_like(selected);manual[:,10:20,10:20]=1
        out=q.combine_masks(selected,manual,'subtract painted area',5)
        self.assertTrue(torch.all(out[manual>0]==0))
    def test_restrict_after_growth(self):
        selected=torch.ones(1,31,31);manual=torch.zeros_like(selected);manual[:,10:20,10:20]=1
        out=q.combine_masks(selected,manual,'restrict to painted area',5)
        self.assertTrue(torch.equal(out,manual))
    def test_missing_correction_mask(self):
        with self.assertRaises(ValueError):q.combine_masks(torch.ones(1,8,8),None,'subtract painted area',0)
    def test_add_manual(self):
        selected=torch.zeros(1,8,8);manual=torch.zeros_like(selected);manual[:,1,1]=1
        self.assertEqual(float(q.combine_masks(selected,manual,'add painted area',0).sum()),1.)
    def test_sam_delegation_and_release(self):
        events=[]
        source=torch.rand(1,71,111,3);saved=source.clone()
        class Detector:
            @classmethod
            def execute(cls,**kwargs):
                events.append(('detect',kwargs['refine_iterations'],kwargs['individual_masks']))
                kwargs['image'].zero_()  # source must stay immutable even if an engine mutates its input
                mask=torch.zeros(1,71,111);mask[:,10:30,20:40]=1
                return (mask,[])
        fake_nodes=types.SimpleNamespace(
            CheckpointLoaderSimple=lambda:types.SimpleNamespace(load_checkpoint=lambda name:('model','clip',None)),
            CLIPTextEncode=lambda:types.SimpleNamespace(encode=lambda clip,text:('cond',)))
        mm=types.ModuleType('comfy.model_management');mm.unload_all_models=lambda:events.append(('unload',))
        comfy=types.ModuleType('comfy');comfy.model_management=mm
        mods={'folder_paths':types.SimpleNamespace(get_full_path=lambda c,n:'/fake/model'), 'nodes':fake_nodes,
              'comfy_extras.nodes_sam3':types.SimpleNamespace(SAM3_Detect=Detector),'comfy':comfy,'comfy.model_management':mm}
        with patch.dict(sys.modules,mods):
            mask,report=q.Q21PhotoMask().make_mask(source,auto_mask=True)
        self.assertTrue(torch.equal(saved,source));self.assertEqual(int(mask.sum()),400)
        self.assertEqual(events,[('detect',2,True),('unload',)])

class NodeTests(unittest.TestCase):
    def test_lazy_preview_requests_no_result(self):
        self.assertEqual(q.Q21PhotoReviewSave().check_lazy_status({},False),[])
        self.assertEqual(q.Q21PhotoReviewSave().check_lazy_status({},True),['edited_result'])
        self.assertEqual(q.Q21PhotoReviewSave().check_lazy_status({},True,edited_result={'image':1}),[])
    def test_disabled_reference_needs_no_files_or_core(self):
        with patch.dict(sys.modules,{'nodes':None}):
            self.assertEqual(q.Q21PhotoReference().load()[0],{'enabled':False})
    def test_reference_gap_rejected(self):
        with self.assertRaises(ValueError):q.ordered_references({'reference_2':{'enabled':True}})
    def test_nine_references(self):
        refs={f'reference_{i}':{'enabled':True,'n':i} for i in range(1,10)}
        self.assertEqual([r['n'] for r in q.ordered_references(refs)],list(range(1,10)))
    def test_loras_off_calls_no_loader(self):
        fake=types.SimpleNamespace(LoraLoaderModelOnly=lambda: (_ for _ in ()).throw(AssertionError('should not load')))
        with patch.dict(sys.modules,{'nodes':fake}): self.assertEqual(q.Q21PhotoLoRAs().apply('base'),('base',))
    def test_lora_chaining(self):
        calls=[]
        fake=types.SimpleNamespace(LoraLoaderModelOnly=lambda:types.SimpleNamespace(load_lora_model_only=lambda m,n,s: (calls.append((m,n,s)) or m+'+',)))
        with patch.dict(sys.modules,{'nodes':fake}):
            out=q.Q21PhotoLoRAs().apply('base',lora_1='one.safetensors',strength_1=.2,lora_2='None',lora_3='three.safetensors',strength_3=.5)
        self.assertEqual(out,('base++',));self.assertEqual(len(calls),2)
    def test_native_qwen_encoder_uses_prepared_images_without_resizing(self):
        data,_,_=g.crop_photo(torch.rand(1,65,97,3),torch.ones(1,65,97),0)
        observed={}
        class Encode:
            @classmethod
            def execute(cls,**kwargs):
                observed.update(kwargs)
                h,w=kwargs['images']['image_1'].shape[1:3]
                return ('positive','negative',{'samples':torch.zeros(1,64,h//16,w//16)})
        ref={'enabled':True,'image':torch.rand(1,128,96,3),'filename':'reference.png','role':'Arm characteristics only; retain source perspective.'}
        with patch.dict(sys.modules,{'comfy_extras.nodes_qwen':types.SimpleNamespace(TextEncodeQwenImage21=Encode)}):
            out=q.Q21PhotoEncode().encode('clip','vae',data,'Edit <image1> using <image2>.',reference_1=ref)
        self.assertEqual(observed['resolution'],0)
        self.assertEqual(list(observed['images']),['image_1','image_2'])
        self.assertIn('<image2>: Arm characteristics',observed['prompt'])
        self.assertEqual(observed['negative_prompt'],'')
        self.assertNotIn('noise_mask',out[2])
    def test_nonexistent_image_tag_rejected(self):
        with patch.dict(sys.modules,{'comfy_extras.nodes_qwen':types.SimpleNamespace(TextEncodeQwenImage21=object)}):
            with self.assertRaises(ValueError):q.Q21PhotoEncode().encode(None,None,{'work_image':torch.zeros(1,64,64,3)},'Use <image2>.')
    def test_rgba_is_composited_on_source_not_black(self):
        data,work,_=g.crop_photo(torch.ones(1,64,64,3)*.4,torch.ones(1,64,64),0);data['mask_report']={}
        rgba=torch.zeros(1,64,64,4)
        out=q.Q21PhotoStitch().stitch(data,rgba,'{}',0)[0]
        self.assertTrue(torch.all(out['image']==.4));self.assertTrue(out['report']['rgba_composited'])
    def test_real_png_roundtrip_and_preview_no_export(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'temp').mkdir();(root/'output').mkdir()
            fake=types.SimpleNamespace(get_temp_directory=lambda:str(root/'temp'),get_output_directory=lambda:str(root/'output'),
                get_save_image_path=lambda prefix,d,w,h:(d,'test',1,'',prefix))
            raw=np.random.default_rng(23).integers(0,256,(65,97,3),dtype=np.uint8)
            source=torch.from_numpy(raw.astype(np.float32)/255)[None];mask=torch.zeros(1,65,97);mask[:,10:40,20:60]=1
            data,work,_=g.crop_photo(source,mask,0);data['mask_report']={'auto_mask':False}
            with patch.dict(sys.modules,{'folder_paths':fake}):
                result=q.Q21PhotoReviewSave().review(data,False)
                self.assertEqual(list((root/'output').iterdir()),[])
                out=q.Q21PhotoStitch().stitch(data,torch.ones_like(work),'{}',0)[0]
                result=q.Q21PhotoReviewSave().review(data,True,edited_result=out)
                file=root/'output'/result['ui']['images'][0]['filename']
                img=Image.open(file);saved=np.asarray(img)
                self.assertEqual(img.size,(97,65));self.assertIn('icc_profile',img.info)
                self.assertTrue(np.array_equal(saved[mask[0].numpy()==0],raw[mask[0].numpy()==0]))
                self.assertEqual(len(list((root/'output').glob('*.verification.json'))),1)
                self.assertEqual(len(list((root/'output').glob('*.mask.png'))),1)

class WorkflowTests(unittest.TestCase):
    def test_graph_connections_and_widgets(self):
        for precision in q.PRESETS:
            wf,api=builder.make(precision);nodes={n['id']:n for n in wf['nodes']}
            for link,source,out,target,inp,typ in wf['links']:
                self.assertEqual(nodes[source]['outputs'][out]['type'],typ)
                self.assertEqual(nodes[target]['inputs'][inp]['type'],typ)
                self.assertEqual(nodes[target]['inputs'][inp]['link'],link)
                self.assertIn(link,nodes[source]['outputs'][out]['links'])
                self.assertEqual(api[str(target)]['inputs'][nodes[target]['inputs'][inp]['name']],[str(source),out])
            self.assertFalse(api['2']['inputs']['auto_mask']);self.assertFalse(api['14']['inputs']['run_edit'])
            self.assertEqual(api['11']['inputs']['steps'],40);self.assertEqual(api['11']['inputs']['cfg'],1.)
            self.assertEqual(api['11']['inputs']['scheduler'],'simple')
            self.assertEqual(api['3']['inputs']['resolution'],2048)
            for i in (6,7,8,9):self.assertEqual(api[str(i)]['inputs']['image'],'None')
    def test_ui_nonserialized_compare(self):
        text=(ROOT/'node/web/review.js').read_text()
        self.assertIn('serialize: false',text)
        self.assertNotIn('preview_long_side',text)
    def test_file_completeness_checks(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'file.safetensors'
            header=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
            data=struct.pack('<Q',len(header))+header+b'1234'
            path.write_bytes(data);self.assertTrue(download.valid_safetensors(path))
            path.write_bytes(data[:-1]);self.assertFalse(download.valid_safetensors(path))
            path.write_bytes(b'<html>Error</html>');self.assertFalse(download.valid_safetensors(path))
    def test_no_automatic_global_upgrader(self):
        text=(ROOT/'install.sh').read_text()
        self.assertNotIn('git pull',text);self.assertNotIn('pip install -U',text)

if __name__=='__main__': unittest.main(verbosity=2)
