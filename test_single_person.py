"""Local regression checks; upstream inference and transport are mocked."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('single_person_test_helpers', ROOT/'scripts/local_nodes/ComfyUI-SinglePersonH3/__init__.py')
p = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(p)
sys.path.insert(0, str(ROOT/'scripts'))
import check_workflows
import catalog


def img(value=0.1, h=32, w=48): return torch.full((1,h,w,3), value)
def refs(): return p.collect_references(img(), 'person', body_1=img(.2,64,32),body_2=img(.3,32,64))
def mod(role, base='person_set', kind='image', tokens=100):
    return types.SimpleNamespace(kind=kind, name=base+'_'+role, description=p.ROLE_MARKER+role+'] one person', token_count=tokens)
def graph(name):return json.loads((ROOT/'config/workflows'/name).read_text())
def nodes(g, kind):return [n for n in g['nodes'] if n['type']==kind]

class PersonReferenceTests(unittest.TestCase):
    def test_head_and_eight_body_views(self):
        r=p.collect_references(img(), 'person', **{f'body_{i}':img(i/10+.1) for i in range(1,9)})
        self.assertEqual(len(r['images']),9);self.assertEqual(r['roles'][0],'primary_headshot')
    def test_unselected_slots_are_not_references(self):
        r=p.collect_references(img(), 'person', body_1=None,body_6=img(.3))
        self.assertEqual(r['roles'],('primary_headshot','body_6'))
        s=p.reference_ledger(r['roles']);self.assertIn('<Picture 2>',s);self.assertNotIn('<Picture 6>',s)
    def test_different_aspect_ratios_unchanged(self):
        r=refs();self.assertEqual([tuple(t.shape[1:3]) for t in r['images']],[(32,48),(64,32),(32,64)])
    def test_original_objects_pass_through(self):
        h,b=img(),img(.5);r=p.collect_references(h,'person',body_1=b)
        self.assertIs(r['images'][0],h);self.assertIs(r['images'][1],b)
    def test_duplicate_image_rejected(self):
        with self.assertRaisesRegex(ValueError,'same image'):p.collect_references(img(),'x',body_1=img())
    def test_headshot_required(self):
        with self.assertRaisesRegex(ValueError,'Primary headshot'):p.collect_references(None,'x')
    def test_image_batch_rejected(self):
        with self.assertRaises(ValueError):p.collect_references(torch.zeros(2,32,32,3),'x')
    def test_nonfinite_rejected(self):
        a=img();a[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):p.collect_references(a,'x')
    def test_label_has_no_path_traversal(self):
        r=p.collect_references(img(),'../../example');self.assertEqual(r['person_label'],'example')
    def test_empty_label_rejected(self):
        with self.assertRaises(ValueError):p.clean_label('...')
    def test_extra_slot_rejected(self):
        with self.assertRaises(ValueError):p.collect_references(img(),'x',body_9=img(.3))
    def test_headshot_only_explicit(self):
        s=p.reference_ledger(['primary_headshot']);self.assertIn('hidden body proportions are not established',s)
    def test_face_and_wardrobe_priorities(self):
        s=p.reference_ledger(refs()['roles']);self.assertIn('headshot takes priority',s)
        self.assertIn('<Picture 2> supplies wardrobe',s);self.assertIn('ONE PERSON ONLY',s)
    def test_multiple_subject_tag_rejected(self):
        with self.assertRaises(ValueError):p.manual_prompt(refs()['roles'],'Make <Subject 1> hug <Subject 2>')
    def test_absent_picture_rejected(self):
        with self.assertRaises(ValueError):p.manual_prompt(refs()['roles'],'Use <Picture 4>')
    def test_no_video_or_audio_reference(self):
        with self.assertRaises(ValueError):p.manual_prompt(refs()['roles'],'Use <Audio 1>')
    def test_manual_prompt_preserves_user_instruction(self):
        s=p.manual_prompt(refs()['roles'],'Walk three slow steps. No dialogue.');self.assertIn('Walk three slow steps. No dialogue.',s)
        self.assertIn('<Picture 3>',s);self.assertNotIn('<Subject 2>',s)
    def test_optional_empty_never_calls_native_loader(self):
        with mock.patch.object(p,'registered',side_effect=AssertionError('no native call')):
            self.assertEqual(p.QualityOptionalBodyImage().load('(none)'),(None,))
            self.assertTrue(p.QualityOptionalBodyImage.VALIDATE_INPUTS('(none)'))
            self.assertEqual(p.QualityOptionalBodyImage.IS_CHANGED('(none)'),'empty')
    def test_optional_upload_uses_native_security_checks(self):
        native=mock.Mock();native.VALIDATE_INPUTS.return_value='Invalid filename'
        with mock.patch.object(p,'registered',return_value=native):
            self.assertEqual(p.QualityOptionalBodyImage.VALIDATE_INPUTS('../bad'),'Invalid filename')
    def test_optional_upload_has_standard_upload_widget(self):
        native=mock.Mock();native.INPUT_TYPES.return_value={'required':{'image':(['a.png'],{'image_upload':True})}}
        with mock.patch.object(p,'registered',return_value=native):
            spec=p.QualityOptionalBodyImage.INPUT_TYPES()['required']['image']
            self.assertEqual(spec[0],['(none)','a.png']);self.assertTrue(spec[1]['image_upload'])

class PersonEncodingTests(unittest.TestCase):
    def fake_call(self,node_type,**kwargs):
        role=kwargs['description'].split(' ',1)[1].split(']',1)[0]
        x=mod(role);x.name=kwargs['name'];x.description=kwargs['description']
        return [(x,1.0)],'details'
    def test_call_adapter_allows_a_name_widget(self):
        class Upstream:
            FUNCTION = "execute"
            def execute(self, name, mode):
                return types.SimpleNamespace(result=(name, mode))
        with mock.patch.object(p, "registered", return_value=Upstream):
            self.assertEqual(p.call_node("MiniMaxH3RefModExtract", name="person", mode="Full Reference"), ("person", "Full Reference"))
    def test_call_adapter_legacy_dictionary_return(self):
        class Legacy:
            FUNCTION = "run"
            def run(self, value):
                return {"ui": {}, "result": (value,)}
        with mock.patch.object(p, "registered", return_value=Legacy):
            self.assertEqual(p.call_node("legacy", value="ok"), ("ok",))
    def test_each_image_encoded_separately(self):
        r=refs()
        with mock.patch.object(p,'call_node',side_effect=self.fake_call) as call:
            out,name,tokens=p.encode_person_bundle(r,object(),2048,131072)
        self.assertEqual(len(out),3);self.assertEqual(tokens,300)
        self.assertEqual(len(call.call_args_list),3)
        for i,c in enumerate(call.call_args_list):
            kw=c.kwargs;self.assertEqual(len(kw['refs_image']),1)
            self.assertIs(kw['refs_image']['ref_image_0'],r['images'][i]);self.assertFalse(kw['save'])
            self.assertEqual(kw['mode'],'Full Reference');self.assertEqual(kw['budget_policy'],'error')
            self.assertEqual((kw['identity'],kw['multiplier'],kw['merge']),(0,1,False))
    def test_token_overflow_is_error(self):
        with mock.patch.object(p,'call_node',side_effect=self.fake_call):
            with self.assertRaisesRegex(ValueError,'no views truncated'):p.encode_person_bundle(refs(),object(),2048,150)
    def test_no_automatic_resolution_fallback(self):
        with mock.patch.object(p,'call_node',side_effect=self.fake_call) as c:
            p.encode_person_bundle(refs(),object(),2048,131072)
        self.assertTrue(all(k.kwargs['ref_resolution']==2048 for k in c.call_args_list))
    def test_invalid_resolution_rejected(self):
        with self.assertRaises(ValueError):p.encode_person_bundle(refs(),object(),2050,1000)
    def test_stacked_video_layout_rejected(self):
        with mock.patch.object(p,'call_node',return_value=([(mod('primary_headshot',kind='video'),1.0)],'')):
            with self.assertRaises(RuntimeError):p.encode_person_bundle(refs(),object(),2048,131072)
    def test_name_deterministic_and_setting_sensitive(self):
        with mock.patch.object(p,'call_node',side_effect=self.fake_call):
            a=p.encode_person_bundle(refs(),object(),2048,131072)[1]
            b=p.encode_person_bundle(refs(),object(),2048,131072)[1]
            c=p.encode_person_bundle(refs(),object(),1024,131072)[1]
        self.assertEqual(a,b);self.assertNotEqual(a,c)
    def test_cancellation_propagates(self):
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            p.encode_person_bundle(refs(),object(),2048,131072,lambda:(_ for _ in ()).throw(RuntimeError('cancel')))
    def test_native_qwen_gets_all_images_without_vae(self):
        r=refs();sentinel=object()
        with mock.patch.object(p,'call_node',return_value=sentinel) as c:
            out=p.QualityPersonNativeConditioning().encode(object(),r,p.manual_prompt(r['roles'],'Stand naturally'),1344,768,124)
        self.assertIs(out,sentinel);self.assertIsNone(c.call_args.kwargs['vae'])
        self.assertEqual(list(c.call_args.kwargs['ref_images']),['ref_image_0','ref_image_1','ref_image_2'])
        self.assertIs(c.call_args.kwargs['ref_images']['ref_image_2'],r['images'][2])
    def test_saved_roles_survive_members(self):
        mods=[(mod(r),1.0) for r in ['primary_headshot','body_1','body_3']]
        self.assertEqual(p.saved_roles(mods),['primary_headshot','body_1','body_3'])
    def test_saved_mixed_people_rejected(self):
        mods=[(mod('primary_headshot'),1.0),(mod('body_1','other_person'),1.0)]
        with self.assertRaisesRegex(ValueError,'different saved sets'):p.saved_roles(mods)
    def test_saved_duplicate_head_rejected(self):
        with self.assertRaises(ValueError):p.saved_roles([(mod('primary_headshot'),1.0)]*2)
    def test_saved_wrong_order_rejected(self):
        with self.assertRaises(ValueError):p.saved_roles([(mod('body_1'),1.0),(mod('primary_headshot'),1.0)])
    def test_saved_video_rejected(self):
        with self.assertRaises(ValueError):p.saved_roles([(mod('primary_headshot',kind='video'),1.0)])
    def test_saved_legacy_roles_rejected(self):
        x=mod('primary_headshot');x.description='old unlabelled reference'
        with self.assertRaises(ValueError):p.saved_roles([(x,1.0)])
    def test_saved_strength_change_rejected(self):
        with self.assertRaises(ValueError):p.saved_roles([(mod('primary_headshot'),.5)])
    def test_saved_unselected_rejected(self):
        with self.assertRaises(ValueError):p.saved_roles([])
    def test_instruction_mode_no_llm(self):
        r=refs()
        with mock.patch.object(p,'registered',side_effect=AssertionError('no llm')):
            text,out,length=p.QualitySinglePersonPrompt().build(r,p.MODES[0],'Stand naturally',124,42,.2,False)
        self.assertIs(out,r);self.assertEqual(length,124);self.assertIn('ONE PERSON ONLY',text)
    def test_invalid_frame_grid_rejected(self):
        with self.assertRaises(ValueError):p.QualitySinglePersonPrompt().build(refs(),p.MODES[0],'Stand',120,42,.2,False)
    def test_llm_receives_every_selected_image(self):
        import threading
        r=refs();fake_mm=mock.Mock();fake_mm.__name__='comfy.model_management'
        fake_comfy=types.ModuleType('comfy');fake_comfy.__path__=[];fake_comfy.model_management=fake_mm
        helper=types.SimpleNamespace(settings=lambda:{'context_length':16384},LOCK=threading.Lock(),generate=mock.Mock(return_value=(p.manual_prompt(r['roles'],'Stand'),{})))
        with mock.patch.object(p,'registered',return_value=type('MockBuilder',(),{'__module__':'test_helper_module'})), \
             mock.patch.object(p.importlib,'import_module',return_value=helper), \
             mock.patch.dict(sys.modules,{'comfy':fake_comfy,'comfy.model_management':fake_mm}):
            out=p.QualitySinglePersonPrompt().build(r,p.MODES[1],'Stand',124,42,.2,False)
        self.assertEqual(len(helper.generate.call_args.args[3]),3)
        self.assertEqual(helper.generate.call_args.args[0]['context_length'],32768)
        self.assertIs(out[1],r);fake_mm.unload_all_models.assert_called_once()

class PersonGraphTests(unittest.TestCase):
    names=['08_H3_Create_Person_RefMod.json','09_H3_One_Person_Prompt_Draft.json','24_H3_One_Person_Multi_Ref_Quality.json','25_H3_One_Person_Saved_RefMod_Quality.json']
    def test_all_new_graphs_valid(self):
        for name in self.names:check_workflows.validate_graph(graph(name))
    def test_one_head_eight_optional_uploads(self):
        for name in self.names[:3]:
            g=graph(name);self.assertEqual(len(nodes(g,'LoadImage')),1);self.assertEqual(len(nodes(g,'QualityOptionalBodyImage')),8)
    def test_creator_does_not_load_diffusion_or_llm(self):
        g=graph(self.names[0]);self.assertFalse(nodes(g,'UNETLoader'));self.assertFalse(nodes(g,'CLIPLoader'));self.assertFalse(nodes(g,'QualitySinglePersonPrompt'))
    def test_one_apply_only(self):
        self.assertEqual(len(nodes(graph(self.names[2]),'MiniMaxH3RefModApply')),1)
        self.assertFalse(nodes(graph(self.names[3]),'MiniMaxH3RefModApply'))
    def test_render_defaults_do_not_call_ollama(self):
        n=nodes(graph(self.names[2]),'QualitySinglePersonPrompt')[0];self.assertEqual(n['widgets_values_named']['mode'],'Use instruction (no Ollama)')
    def test_draft_does_call_ollama_without_video(self):
        g=graph(self.names[1]);self.assertFalse(nodes(g,'UNETLoader'));self.assertEqual(nodes(g,'QualitySinglePersonPrompt')[0]['widgets_values_named']['mode'],'Generate with Ollama')
    def test_all_profiles_mapped(self):
        r=json.loads((ROOT/'config/runtime.json').read_text())
        for n in self.names:self.assertEqual(r['workflow_profiles'][n],'h3')
    def test_existing_krea_and_model_selection_untouched(self):
        self.assertEqual(set(json.loads((ROOT/'config/runtime.json').read_text())['profiles']),{'seedvr2','krea2','h3'})
        for name in self.names:self.assertFalse(any('Rebalance' in n['type'] for n in graph(name)['nodes']))
    def test_duplicate_apply_graph_is_rejected(self):
        g=graph(self.names[2]);a=copy.deepcopy(nodes(g,'MiniMaxH3RefModApply')[0]);a['id']=999
        for i in a['inputs']:i['link']=None
        for o in a['outputs']:o['links']=[]
        g['nodes'].append(a);g['last_node_id']=999
        with self.assertRaisesRegex(ValueError,'exactly once'):check_workflows.validate_graph(g)
    def test_build_runtime_probe_included(self):
        self.assertIn('check_single_person_runtime.py',(ROOT/'Dockerfile').read_text())
        self.assertIn('check_single_person_runtime.py',(ROOT/'scripts/prepare_image.sh').read_text())

if __name__=='__main__':unittest.main()
