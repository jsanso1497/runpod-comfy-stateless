"""Offline tests use model/Comfy stubs, never claim inference or visual quality."""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('h3studio_offline',P/'nodes/studio/__init__.py',submodule_search_locations=[str(P/'nodes/studio')])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
r=sys.modules['h3studio_offline.logic']
from build_studio_workflow import build
from studio_validate import validate_studio


def planned(mode='Auto', count=0, preset=None, source=None):
    return r.plan_values(mode,count,'1:1',4,2.5,preset or r.PRESETS[0],source)

class PlanTests(unittest.TestCase):
    def test_zero_is_generate(self):self.assertEqual(planned()['mode'],'Generate')
    def test_one_is_edit(self):self.assertEqual(planned(count=1,source=(640,960))['mode'],'Edit')
    def test_two_through_nine_are_references(self):
        for n in range(2,10):self.assertEqual(planned(count=n)['mode'],'References')
    def test_single_reference_new_generation(self):
        x=planned('References',1);self.assertEqual(x['mode'],'References');self.assertEqual(x['width'],x['height'])
    def test_generate_rejects_accidental_references(self):
        with self.assertRaises(ValueError):planned('Generate',1)
    def test_edit_requires_exactly_one(self):
        for n in (0,2,9):
            with self.assertRaises(ValueError):planned('Edit',n)
    def test_refs_requires_photo(self):
        with self.assertRaises(ValueError):planned('References',0)
    def test_bounds(self):
        for n in (-1,10,'1'):
            with self.assertRaises(ValueError):planned(count=n)
    def test_generate_selects_fl2va_int8(self):self.assertEqual(planned()['model'],r.FL2VA)
    def test_edit_selects_fl2va_int8(self):self.assertEqual(planned(count=1,source=(640,960))['model'],r.FL2VA)
    def test_references_selects_ref2va_int8(self):self.assertEqual(planned(count=3)['model'],r.REF2VA)
    def test_quality_has_no_turbo(self):
        x=planned();self.assertEqual((x['steps'],x['turbo_strength']),(50,0))
    def test_fast_fl2va_is_developer_recipe(self):
        x=planned(preset=r.PRESETS[1]);self.assertEqual((x['steps'],x['turbo_strength']),(20,.38))
    def test_fast_does_not_cross_to_ref2va(self):
        with self.assertRaises(ValueError):planned(count=2,preset=r.PRESETS[1])
    def test_output_sizes_aligned_and_bounded(self):
        for aspect in r.ASPECTS:
            for mp in (.25,2.5,4,8):
                x=r.plan_values('Generate',0,aspect,mp,2.5,r.PRESETS[0])
                self.assertTrue(all(64<=x[k]<=4096 and x[k]%32==0 for k in ('width','height')))
    def test_edit_uses_original_orientation(self):
        x=planned(count=1,source=(960,640));self.assertGreater(x['width'],x['height'])
    def test_reference_mode_uses_output_canvas_not_source(self):
        x=planned('References',1,source=(640,960));self.assertEqual(x['width'],x['height'])
    def test_impossible_size_rejected(self):
        for x in (float('nan'),float('inf'),0,9):
            with self.assertRaises(ValueError):r.dimensions(1,x)
    def test_small_reference_not_upscaled_for_detail(self):self.assertEqual(r.reference_size(96,160,2.5),(96,160))
    def test_large_reference_capped(self):
        w,h=r.reference_size(6000,4000,2.5);self.assertLess(w*h,2.7*1024*1024)
    def test_prompt_tags_checked(self):
        r.check_prompt('Combine <Picture 1> and <Picture 2>',2)
        for prompt,n in [('Edit <Picture 1>',0),('Use <Picture 3>',2),('Use <Picture 0>',1),('',1)]:
            with self.assertRaises(ValueError):r.check_prompt(prompt,n)

class GraphTests(unittest.TestCase):
    def test_valid_graph(self):self.assertTrue(validate_studio())
    def test_deterministic(self):self.assertEqual(build(),build())
    def test_user_graph_not_mutated(self):
        g=build();before=copy.deepcopy(g);validate_studio(graph=g);self.assertEqual(g,before)
    def test_invalid_connection_detected(self):
        g=build();g['links'][0][1]=999
        with self.assertRaises(ValueError):validate_studio(graph=g)
    def test_only_quantized_asset_downloads(self):
        assets=json.loads((P/'models.lock.json').read_text())['models']
        self.assertEqual(len(assets),5)
        self.assertFalse(any('bf16' in a['file'] for a in assets))
        self.assertTrue(any('ref2va_pruned_int8' in a['file'] for a in assets))
    def test_no_prompt_enhancer_or_ailab(self):
        text=json.dumps(build())
        for kind in ('TextGenerate','AILab_ImageResize','AILab_ColorInput'):self.assertNotIn(kind,text)
    def test_no_old_decode_comparison_branch(self):
        self.assertNotIn('VAEDecode',[n['type'] for n in build()['nodes']])
    def test_nine_blank_slots_are_serialized(self):
        nodes=[n for n in build()['nodes'] if n['type']=='H3StudioOptionalImage']
        self.assertEqual(len(nodes),9)
        self.assertTrue(all(n['widgets_values_named']['image']==r.NONE for n in nodes))

class HelperTests(unittest.TestCase):
    def test_empty_slot_has_no_file_dependency(self):
        self.assertIsNone(m.H3StudioOptionalImage().load(r.NONE)[0]['image'])
    def test_disabled_stale_filename_is_ignored(self):
        self.assertIsNone(m.H3StudioOptionalImage().load('missing.png',False)[0]['image'])
    def test_validation_allows_blank_and_disabled(self):
        self.assertIs(m.H3StudioOptionalImage.VALIDATE_INPUTS(r.NONE),True)
        self.assertIs(m.H3StudioOptionalImage.VALIDATE_INPUTS('missing.png',False),True)
    def test_none_changed_is_stable(self):
        self.assertEqual(m.H3StudioOptionalImage.IS_CHANGED(r.NONE),'empty')
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t, patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace(get_input_directory=lambda:t)}):
            for value in ('../escape.png','/tmp/escape.png','x\\y.png'):
                with self.assertRaises(ValueError):m.image_path(value)
    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'input';root.mkdir();outside=Path(t)/'outside.png';outside.write_bytes(b'x');(root/'escape.png').symlink_to(outside)
            with patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace(get_input_directory=lambda:str(root))}):
                with self.assertRaises(ValueError):m.image_path('escape.png')
    def test_digest_catches_same_filename_changed(self):
        with tempfile.TemporaryDirectory() as t, patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace(get_input_directory=lambda:t)}):
            path=Path(t)/'image.png';path.write_bytes(b'one');a=m.H3StudioOptionalImage.IS_CHANGED(path.name)
            path.write_bytes(b'two');b=m.H3StudioOptionalImage.IS_CHANGED(path.name)
            self.assertNotEqual(a,b)
    def test_plan_compresses_gaps_and_reports_mapping(self):
        fake=types.SimpleNamespace(shape=(1,96,160,3))
        with patch.object(m,'resize_image',lambda img,w,h:img):
            x=m.H3StudioPlan().prepare('Auto','1:1',4,2.5,r.PRESETS[0],2.5,
                  reference_3={'image':fake,'name':'a.png'},reference_9={'image':fake,'name':'b.png'})[0]
        self.assertEqual([z['slot'] for z in x['summary']['reference_mapping']],[3,9])
        self.assertEqual([z['tag'] for z in x['summary']['reference_mapping']],['<Picture 1>','<Picture 2>'])
    def test_native_output_variants(self):
        self.assertEqual(m.native_result(('x','y')),'x')
        self.assertEqual(m.native_result(types.SimpleNamespace(result=('x','y'))),'x')
    def test_lora_disabled_means_no_loading(self):
        obj=object();self.assertIs(m.H3StudioLora().apply(obj,False,'missing.safetensors',1)[0],obj)
    def test_lora_none_and_zero_bypass(self):
        obj=object()
        self.assertIs(m.H3StudioLora().apply(obj,True,r.NONE,1)[0],obj)
        self.assertIs(m.H3StudioLora().apply(obj,True,'missing.safetensors',0)[0],obj)
    def test_lora_pickle_refused(self):
        with self.assertRaises(ValueError):m.H3StudioLora().apply(object(),True,'bad.pt',1)
    def test_lora_turbo_not_double_applied(self):
        with self.assertRaises(ValueError):m.H3StudioLora().apply(object(),True,r.TURBO,1)
    def test_unmatched_lora_fails_clearly(self):
        model=types.SimpleNamespace(patches={})
        loader=types.SimpleNamespace(load_lora_model_only=lambda *args:(model,))
        fake_nodes=types.SimpleNamespace(LoraLoaderModelOnly=lambda:loader)
        fake_paths=types.SimpleNamespace(get_full_path_or_raise=lambda *a:'/unused')
        with patch.dict(sys.modules,{'nodes':fake_nodes,'folder_paths':fake_paths}):
            with self.assertRaisesRegex(ValueError,'No new model patches'):m.H3StudioLora().apply(model,True,'h3.safetensors',1)
    def test_compare_generation_skips(self):
        result=m.H3StudioCompare().save({'mode':'Generate','original':None},None)
        self.assertIn('skipped',result['ui']['text'][0])
    def test_model_cache_only_calls_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as t:
            file=Path(t)/'dummy';file.write_bytes(b'not model weights')
            loader=Mock();loader.load_unet.return_value=(object(),)
            fake_nodes=types.SimpleNamespace(UNETLoader=lambda:loader)
            paths=types.SimpleNamespace(get_full_path_or_raise=lambda *a:str(file))
            with patch.dict(sys.modules,{'nodes':fake_nodes,'folder_paths':paths}):
                selected=m.H3StudioModel();selected.load(planned());selected.load(planned())
                self.assertEqual(loader.load_unet.call_count,1)
                selected.load(planned(count=2))
                self.assertEqual([call.args[0] for call in loader.load_unet.call_args_list],[r.FL2VA,r.REF2VA])
    def test_native_conditioning_receives_no_hidden_prompt(self):
        log=[]
        def execute(**kwargs):log.append(kwargs);return types.SimpleNamespace(result=('cond','unused video latent'))
        native=types.SimpleNamespace(MiniMaxH3ImageToVideo=types.SimpleNamespace(execute=execute),MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=execute))
        with patch.dict(sys.modules,{'comfy_extras':types.ModuleType('comfy_extras'),'comfy_extras.nodes_minimax_h3':native}):
            p=planned();p['images']=[]
            self.assertEqual(m.H3StudioConditioning().encode(p,'clip','vae','Exact prompt'),('cond',))
            p=planned(count=2);p['images']=['a','b']
            m.H3StudioConditioning().encode(p,'clip','vae','Use <Picture 1> and <Picture 2>')
        self.assertEqual(log[0]['prompt'],'Exact prompt')
        self.assertEqual(log[0]['length'],5)
        self.assertEqual(list(log[1]['ref_images'].values()),['a','b'])

if __name__=='__main__':unittest.main()
