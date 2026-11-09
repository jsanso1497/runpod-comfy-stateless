import copy
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
import build_workflows as builder
from graph_checks import validate_graph
import manage

# Local unit tests deliberately use small tensor fixtures, not H3 weights.
try:
    import torch
except ImportError:
    torch=None
folder=types.ModuleType('folder_paths');folder.add_model_folder_path=lambda *a,**kw:None
folder.get_filename_list=lambda kind:[];folder.get_full_path=lambda kind,name:None
sys.modules.setdefault('folder_paths',folder)
node_registry=types.ModuleType('nodes');node_registry.NODE_CLASS_MAPPINGS={}
sys.modules.setdefault('nodes',node_registry)
import nodepack.common as common
import nodepack.nodes as unit
import nodepack.experiments as experiments

class GraphTests(unittest.TestCase):
    def test_all_graphs(self):
        for name,graph in builder.build_all().items():
            with self.subTest(name=name):self.assertEqual(validate_graph(graph),[])
    def test_baseline_unchanged(self):
        self.assertEqual(builder.build_all()['00_Known_Good_Replacement.json'],json.loads((P/'fixtures/known_good_replacement.json').read_text()))
    def test_thirteen_workflows(self):self.assertEqual(len(builder.build_all()),13)
    def test_default_experiment_off(self):
        g=builder.studio('x',replace=True);n=next(n for n in g['nodes'] if n['type']=='H3LabExperiment')
        self.assertEqual(n['widgets_values_named']['accelerator'],'None');self.assertFalse(n['widgets_values_named']['character_swap_adapter'])
    def test_spectrum_only(self):
        g=builder.studio('x',replace=True,experiment='Spectrum');v=next(n['widgets_values_named'] for n in g['nodes'] if n['type']=='H3LabExperiment')
        self.assertEqual(v['accelerator'],'Spectrum');self.assertFalse(v['character_swap_adapter']);self.assertFalse(v['allow_combined_experiment'])
    def test_sam_target_mask_rewired(self):
        g=builder.studio('x',replace=True,sam=True);edge=next(e for e in g['links'] if e[3]==31 and e[4]==1)
        self.assertEqual(edge[1],53);self.assertTrue(next(n for n in g['nodes'] if n['id']==10)['widgets_values_named']['protect_background'])
    def test_sam_shares_selection_text(self):
        g=builder.studio('x',replace=True,sam=True);dest={e[3] for e in g['links'] if e[1]==32}
        self.assertEqual(dest,{33,51})
    def test_same_seed_on_experiments(self):
        seeds=[]
        for name,g in builder.build_all().items():
            if name[:2] in ('02','04','05','06'):seeds.append(next(n['widgets_values_named']['noise_seed'] for n in g['nodes'] if n['type']=='RandomNoise'))
        self.assertEqual(len(set(seeds)),1)
    def test_contact_sheet_has_no_model(self):
        g=builder.contact('x');self.assertFalse(any('Loader' in n['type'] or 'Sampler' in n['type'] for n in g['nodes']))
    def test_ph3_native_decoder_only(self):
        g=builder.ph3('x');self.assertFalse(any(n['type'].startswith('Fizgig') for n in g['nodes']))
        self.assertEqual(next(n for n in g['nodes'] if n['type']=='H3LabTimedDecode')['widgets_values_named']['decoder'],'Native P-H3')
    def test_ph3_eight_inputs(self):
        g=builder.ph3('x');self.assertEqual(len([n for n in g['nodes'] if n['type']=='H3StudioOptionalImage']),8)
    def test_detect_dangling_link(self):
        g=builder.contact('x');g['links'][0][1]=987654;self.assertTrue(validate_graph(g))
    def test_detect_duplicate_id(self):
        g=builder.contact('x');g['nodes'].append(copy.deepcopy(g['nodes'][0]));self.assertTrue(validate_graph(g))
    def test_detect_reciprocity(self):
        g=builder.contact('x');g['nodes'][0]['outputs'][0]['links']=[];self.assertTrue(validate_graph(g))
    def test_generator_deterministic(self):self.assertEqual(builder.build_all(),builder.build_all())
    def test_protected_rgb_is_not_blanked(self):
        g=builder.studio('x',replace=True,sam=True);edge=next(e for e in g['links'] if e[3]==31 and e[4]==0);self.assertEqual(edge[1],1)
    def test_quality_default_50(self):
        g=builder.studio('x',replace=True);v=next(n['widgets_values_named'] for n in g['nodes'] if n['id']==10)
        self.assertEqual(v['preset'],builder.QUALITY);self.assertEqual(v['manual_quality_steps'],0)

class PureTests(unittest.TestCase):
    def test_crop_full(self):self.assertEqual(common.crop_bounds(100,200,0,0,0,0),(0,0,100,200))
    def test_crop_half(self):self.assertEqual(common.crop_bounds(100,200,25,0,25,0),(25,0,75,200))
    def test_crop_invalid(self):
        for vals in ((60,0,40,0),(-1,0,0,0),(float('nan'),0,0,0)):
            with self.assertRaises(ValueError):common.crop_bounds(100,200,*vals)
    def test_dimensions_grid(self):
        w,h=common.dimensions(4000,6000,2.5,True);self.assertEqual(w%32,0);self.assertEqual(h%32,0);self.assertLess(w,4000)
    def test_dimensions_no_upscale(self):self.assertEqual(common.dimensions(320,640,2.5,True),(320,640))
    def test_dimensions_invalid(self):
        with self.assertRaises(ValueError):common.dimensions(100,100,float('nan'))
    def test_names(self):
        for value in ('../bad','a/b','','a'*65):
            with self.assertRaises(ValueError):common.short_name(value)
    def test_valid_name(self):self.assertEqual(common.short_name('person_face_v1'),'person_face_v1')
    def test_safe_path(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):manage.safe_path(root,'../outside')
            self.assertEqual(manage.safe_path(root,'file.txt'),root/'file.txt')
    def test_archive_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../bad.py','bad')
            with self.assertRaises(ValueError):manage.safe_extract(archive,Path(d)/'extract')
    def test_archive_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:
                i=zipfile.ZipInfo('link');i.external_attr=0o120777<<16;z.writestr(i,'outside')
            with self.assertRaises(ValueError):manage.safe_extract(archive,Path(d)/'extract')
    def test_archive_good(self):
        with tempfile.TemporaryDirectory() as d:
            archive=Path(d)/'ok.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('folder/a.txt','ok')
            manage.safe_extract(archive,Path(d)/'extract');self.assertEqual((Path(d)/'extract/folder/a.txt').read_text(),'ok')
    def test_all_downloads_hash_locked(self):
        for row in manage.MANIFEST['downloads']:
            self.assertEqual(len(row['sha256']),64);self.assertEqual(len(row['revision']),40)
    def test_no_bf16_bases(self):
        for row in manage.MANIFEST['downloads']:
            if row['group']=='ph3':self.assertIn('int8',row['file'])
    def test_no_enhancer_download(self):self.assertFalse(any('upsampl' in row['file'] for row in manage.MANIFEST['downloads']))
    def test_ph3_sigmas(self):
        self.assertEqual(len(experiments.PH3_SIGMAS),9);self.assertEqual(experiments.PH3_SIGMAS[-1],0.)
        self.assertTrue(all(a>b for a,b in zip(experiments.PH3_SIGMAS,experiments.PH3_SIGMAS[1:])))
    def test_schema_unknown_parameter(self):
        class C:
            @classmethod
            def INPUT_TYPES(cls):return {'required':{'model':('MODEL',),'mode':(['fidelity','speed'],{'default':'fidelity'})}}
        with self.assertRaises(RuntimeError):common.required_defaults(C,{'model':None,'preset':'fidelity'})
    def test_schema_defaults(self):
        class C:
            @classmethod
            def INPUT_TYPES(cls):return {'required':{'model':('MODEL',),'mode':(['fidelity','speed'],{'default':'fidelity'})}}
        self.assertEqual(common.required_defaults(C,{'model':123}),{'model':123,'mode':'fidelity'})
    def test_outputs_legacy(self):self.assertEqual(common.outputs((1,2)),(1,2))
    def test_outputs_v3(self):self.assertEqual(common.outputs(types.SimpleNamespace(result=(1,2))),(1,2))
    def test_outputs_ui(self):self.assertEqual(common.outputs({'ui':{},'result':(1,2)}),(1,2))

class PromptTests(unittest.TestCase):
    def setUp(self):self.plan={'mode':'Replace person','summary':{'reference_mapping':[{'slot':1,'tag':'<Picture 1>'},{'slot':2,'tag':'<Picture 2>'},{'slot':3,'tag':'<Picture 3>'}]}}
    def test_roles(self):
        text=unit.compose_roles(self.plan,'person on the left',2,3,True,False,'Photorealistic.')
        self.assertIn('exclusive head',text);self.assertIn('<Picture 3> for',text);self.assertNotIn('Match the replacement',text)
    def test_lighting_opt_in(self):self.assertIn('Match the replacement',unit.compose_roles(self.plan,'left person',2,3,False,True,'Photo.'))
    def test_blank_target_rejected(self):
        with self.assertRaises(ValueError):unit.compose_roles(self.plan,'',2,3,True,False,'Photo.')
    def test_missing_role_rejected(self):
        with self.assertRaises(ValueError):unit.compose_roles(self.plan,'left person',4,3,True,False,'Photo.')
    def test_direct_prompt_unchanged(self):self.assertEqual(unit.H3LabPrompt().build(self.plan,False,'',0,0,True,True,'Same words.'),('Same words.',))
    def test_blank_prompt_rejected(self):
        with self.assertRaises(ValueError):unit.H3LabPrompt().build(self.plan,False,'',0,0,False,False,'')
    def test_mapping_uses_active_slot(self):
        self.plan['summary']['reference_mapping'][2]={'slot':8,'tag':'<Picture 3>'}
        self.assertIn('<Picture 3> for',unit.compose_roles(self.plan,'left person',2,8,True,False,'Photo.'))

class ExperimentTests(unittest.TestCase):
    def test_off_returns_same_model(self):
        model=object();out,report=experiments.H3LabExperiment().apply(model,{'adapter':''},False,1.,'None',False)
        self.assertIs(out,model);self.assertEqual(json.loads(report)['accelerator'],'None')
    def test_multiple_changes_guarded(self):
        with self.assertRaises(ValueError):experiments.H3LabExperiment().apply(object(),{'adapter':'turbo'},False,1.,'Spectrum',False)
    def test_wrong_model_for_swap(self):
        with self.assertRaises(ValueError):experiments.H3LabExperiment().apply(object(),{'adapter':'','model':'fl2va'},True,1.,'None',False)
    def test_unknown_accelerator(self):
        with self.assertRaises(ValueError):experiments.H3LabExperiment().apply(object(),{'adapter':''},False,1.,'unknown',False)
    def test_missing_swap_download_clear(self):
        with patch.object(sys.modules['folder_paths'],'get_full_path',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'download swap'):experiments.H3LabExperiment().apply(object(),{'adapter':'','model':'ref2va'},True,1.,'None',False)
    @unittest.skipUnless(torch is not None,'torch tensor tests run inside the image')
    def test_spark_a100_rejected(self):
        with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_capability',return_value=(8,0)):
            with self.assertRaisesRegex(ValueError,'SM86'):experiments.H3LabExperiment().apply(object(),{'adapter':''},False,1.,'Spark SM86',False)
    def test_spectrum_exact_contract(self):
        seen={}
        class S:
            FUNCTION='apply'
            @classmethod
            def INPUT_TYPES(cls):
                return {'required':{'model':('MODEL',),'enabled':('BOOLEAN',{'default':True})},'optional':{k:('STRING',) for k in ('debug','model_aware_mode','offline_smoothing_replay','model_aware_trust_shrinkage','history_storage','audio_blend_weight','degree','warmup_steps','tail_actual_steps')}}
            def apply(self,**kw):seen.update(kw);return (kw['model'],)
        with patch.object(experiments,'provider_node',return_value=S):experiments.H3LabExperiment().apply('m',{'adapter':''},False,1.,'Spectrum',False)
        self.assertEqual(seen['degree'],1);self.assertFalse(seen['offline_smoothing_replay']);self.assertEqual(seen['model_aware_mode'],'full')
    def test_ph3_runtime_guard(self):
        if torch is None:self.skipTest('Torch not installed on host')
        comfy=types.ModuleType('comfy');samplers=types.ModuleType('comfy.samplers');comfy.samplers=samplers
        with patch.dict(sys.modules,{'comfy':comfy,'comfy.samplers':samplers}):
            with self.assertRaisesRegex(RuntimeError,'ph3 build'):experiments.H3LabPH3().prepare('A photo','1:1',4.,True)

@unittest.skipUnless(torch is not None,'Torch tensor tests run inside the image')
class TensorTests(unittest.TestCase):
    def resize(self,img,w,h):return torch.nn.functional.interpolate(img.movedim(-1,1),size=(h,w),mode='bilinear').movedim(1,-1)
    def test_empty_reference_stays_empty(self):
        ref={'image':None,'name':''};out,preview,_=unit.H3LabReference().prepare(ref,'face',2.5,0,0,0,0)
        self.assertIsNone(out['image']);self.assertEqual(tuple(preview.shape),(1,32,32,3))
    def test_face_crop_does_not_mutate_source(self):
        source=torch.rand(1,100,200,3);before=source.clone();ref={'image':source,'mask':torch.ones(1,100,200),'name':'ref'}
        with patch.object(unit,'resize',self.resize):out,preview,_=unit.H3LabReference().prepare(ref,'face',.25,25,0,25,0)
        self.assertTrue(torch.equal(source,before));self.assertNotIn('mask',out);self.assertEqual(out['lab_preprocess']['crop'],[50,0,150,100])
    def test_contact_sheet_actual_images(self):
        image=torch.zeros(1,64,128,3);image[:,:,:,0]=1
        sheet=unit.H3LabContactSheet().build(2,128,reference_1={'image':image,'lab_role':'face'},reference_2={'image':image,'lab_role':'body'})[0]
        self.assertEqual(tuple(sheet.shape),(1,158,256,3));self.assertGreater(float(sheet[0,64,64,0]),.9)
    def test_contact_empty_rejected(self):
        with self.assertRaises(ValueError):unit.H3LabContactSheet().build(3,128)
    def test_batch_reference_rejected(self):
        with self.assertRaises(ValueError):unit.H3LabReference().prepare({'image':torch.zeros(2,64,64,3)},'face',1.,0,0,0,0)
    def test_refmod_empty_rejected(self):
        with self.assertRaises(ValueError):experiments.H3LabRefModSave().save({'image':None},None,'name','Full reference',1024)
    def test_sampler_timing_is_not_gpu_claim(self):
        sampler=types.ModuleType('comfy_extras.nodes_custom_sampler')
        class S:
            @classmethod
            def execute(cls,*args):return ({'samples':torch.zeros(1)}, {'samples':torch.zeros(1)})
        sampler.SamplerCustomAdvanced=S
        with patch.dict(sys.modules,{'comfy_extras.nodes_custom_sampler':sampler}),patch.object(torch.cuda,'is_available',return_value=False):
            out,report=unit.H3LabTimedSampler().sample(types.SimpleNamespace(seed=7),None,None,torch.tensor([1.,0.]),{},'{}','{}','test')
        report=json.loads(report);self.assertFalse(report['executed_on_cuda']);self.assertEqual(report['outer_steps'],1);self.assertEqual(report['noise_seed'],7)
    def test_sampler_restores_log_handler_on_failure(self):
        sampler=types.ModuleType('comfy_extras.nodes_custom_sampler')
        class S:
            @classmethod
            def execute(cls,*args):raise ValueError('synthetic failure')
        sampler.SamplerCustomAdvanced=S;before=list(logging.getLogger().handlers)
        with patch.dict(sys.modules,{'comfy_extras.nodes_custom_sampler':sampler}):
            with self.assertRaises(ValueError):unit.H3LabTimedSampler().sample(None,None,None,torch.tensor([1.,0.]),{},'{}','{}','test')
        self.assertEqual(logging.getLogger().handlers,before)
    def test_decode_native_not_fizgig(self):
        called=[]
        class V:
            @classmethod
            def execute(cls,**kw):return (torch.zeros(1,32,64,3),)
        def reg(name):called.append(name);return V
        with patch.object(unit,'registry',reg):image,report=unit.H3LabTimedDecode().decode({},None,'{}','Native P-H3')
        self.assertEqual(called,['VAEDecode']);self.assertEqual(json.loads(report)['decoded_size'],[64,32])

class DownloadTests(unittest.TestCase):
    def test_checksum_cache_and_replacement(self):
        with tempfile.TemporaryDirectory() as d,patch.object(manage,'ROOT',Path(d)):
            p=Path(d)/'model';p.write_bytes(b'good');row={'id':'test','sha256':hashlib.sha256(b'good').hexdigest()}
            self.assertTrue(manage.verified(p,row));self.assertTrue(manage.verified(p,row));p.write_bytes(b'changed');self.assertFalse(manage.verified(p,row))
    def test_bad_existing_weight_not_overwritten(self):
        row={'group':'swap','id':'test','destination':'loras/x','sha256':'0'*64,'approx_gb':1}
        with tempfile.TemporaryDirectory() as d,patch.object(manage,'ROOT',Path(d)),patch.object(manage,'MANIFEST',{'downloads':[row]}):
            p=Path(d)/'models/loras/x';p.parent.mkdir(parents=True);p.write_bytes(b'mine')
            with self.assertRaises(RuntimeError):manage.download('swap')
            self.assertEqual(p.read_bytes(),b'mine')
    def test_spark_hash_pinned(self):self.assertEqual(manage.MANIFEST['spark']['sha256'],'8997f751296d1ff6c032125f2ce2fbc45a776bec5ac8cf40d5c30f3a6ac14a02')
    def test_group_unknown_rejected(self):
        with self.assertRaises(ValueError):manage.download('unknown')
    def test_provider_pin_lengths(self):
        for row in manage.MANIFEST['providers'].values():self.assertEqual(len(row['revision']),40)



class PlanDelegationTests(unittest.TestCase):
    def setUp(self):
        self.plan={'mode':'Replace person','steps':50,'images':[],
                   'summary':{'steps':50,'reference_mapping':[]}}
        self.calls=[]
        outer=self
        class Legacy:
            def prepare(self,*args,**kwargs):
                outer.calls.append((args,kwargs));return outer.plan,1312,1984
        self.patch=patch.object(unit,'registry',return_value=Legacy);self.patch.start()
    def tearDown(self):self.patch.stop()
    def run_plan(self,**updates):
        values=dict(mode='Replace person',aspect_ratio='2:3',output_megapixels=2.5,
            preset=builder.QUALITY,manual_quality_steps=0,reference_size_mode='match',protect_background=False)
        values.update(updates);return unit.H3LabPlan().prepare(**values)
    def test_unified_size_sent_to_both_legacy_controls(self):
        self.run_plan(output_megapixels=1.);args,_=self.calls[-1]
        self.assertEqual(args[2:4],(1.,1.))
    def test_manual_steps_changes_only_returned_plan(self):
        out,_,_=self.run_plan(manual_quality_steps=20)
        self.assertEqual(out['steps'],20);self.assertEqual(out['summary']['steps'],20)
        self.assertEqual(self.plan['steps'],50);self.assertEqual(self.plan['summary']['steps'],50)
    def test_manual_steps_not_silently_applied_to_turbo(self):
        with self.assertRaises(ValueError):self.run_plan(preset='REF2VA Turbo - 8 steps',manual_quality_steps=20)
        self.assertFalse(self.calls)
    def test_manual_steps_too_low_rejected(self):
        with self.assertRaises(ValueError):self.run_plan(manual_quality_steps=2)
    def test_independent_reference_caps_not_lowered_twice(self):
        self.run_plan();self.assertEqual(self.calls[-1][0][5],8.)
    def test_video_and_background_protection_forwarded(self):
        self.run_plan(protect_background=True,video={'frames':'fixture'})
        args,kwargs=self.calls[-1];self.assertTrue(args[6]);self.assertEqual(kwargs['video'],{'frames':'fixture'})
    def test_reference_mode_reported(self):
        out,_,_=self.run_plan(reference_size_mode='max')
        self.assertEqual(out['lab_reference_size'],'max');self.assertEqual(out['summary']['reference_size_mode'],'max')


class RefModMappingTests(unittest.TestCase):
    def test_zero_strength_refs_do_not_consume_picture_numbers(self):
        a=types.SimpleNamespace(kind='image',name='face')
        b=types.SimpleNamespace(kind='image',name='body')
        class L:
            def load(self,**kwargs):return ([(a,0.),(b,1.)],)
        with patch.object(experiments,'provider_node',return_value=L):
            mods,mapping=experiments.H3LabRefModLoad().load('face','body',0.,1.)
        self.assertEqual(mods,[(b,1.)]);self.assertEqual(mapping,'<Picture 1> = body')
    def test_all_zero_refs_rejected(self):
        a=types.SimpleNamespace(kind='image',name='face')
        class L:
            def load(self,**kwargs):return ([(a,0.)],)
        with patch.object(experiments,'provider_node',return_value=L):
            with self.assertRaises(ValueError):experiments.H3LabRefModLoad().load('face','(none)',0.,0.)


@unittest.skipUnless(torch is not None,'Torch contract fixtures run inside image')
class SparkContractTests(unittest.TestCase):
    def test_uses_mode_and_actual_plan_steps(self):
        seen={}
        class S:
            FUNCTION='apply'
            @classmethod
            def INPUT_TYPES(cls):
                return {'required':{'model':('MODEL',),'mode':(['fidelity','speed'],{'default':'fidelity'}),
                    'steps':('INT',{'default':50}),'strict':('BOOLEAN',{'default':False})},
                    'optional':{'dense_backend':(['inherit','sdpa'],), 'sparse_scope':(['target-video','joint-video'],)}}
            def apply(self,**kwargs):seen.update(kwargs);return (kwargs['model'],)
        provider=types.SimpleNamespace(NODE_CLASS_MAPPINGS={'MiniMaxH3SparkAttentionSM86':S})
        with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_capability',return_value=(8,6)),patch.object(experiments,'provider',return_value=provider):
            experiments.H3LabExperiment().apply('model',{'adapter':'','steps':50},False,1.,'Spark SM86',False)
        self.assertEqual(seen['mode'],'fidelity');self.assertEqual(seen['steps'],50)
        self.assertTrue(seen['strict']);self.assertEqual(seen['sparse_scope'],'joint-video')

if __name__=='__main__':unittest.main()
