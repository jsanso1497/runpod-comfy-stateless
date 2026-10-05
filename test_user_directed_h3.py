"""Regression tests for human-directed H3. Transport/native UI are fixtures, not GPU inference."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import torch

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'scripts/local_nodes/ComfyUI-DirectedH3'
spec=importlib.util.spec_from_file_location('directed_h3_tests_package',DIR/'__init__.py',submodule_search_locations=[str(DIR)])
p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p);d=p.d
sys.path.insert(0,str(ROOT/'scripts'))
import check_workflows


def image(v=0.2,h=32,w=48):return torch.full((1,h,w,3),v)
def ref(owner='Person A',v=.2,h=32,w=48):
    return d.make_reference(image(v,h,w),owner,'My exact description','Primary face reference; ignore the outfit')
def refs():return d.collect({'reference_1':ref(),'reference_2':ref('Person A',.3,64,32),'reference_3':ref('Person B',.4)})
def request(r=None):
    r=r or refs()
    facts,sig,exact=d.check_request(r,'Person A and Person B hug.','Stationary camera. No dialogue or music.',
                                  'Preserve the supplied identities.',124,1344,768)
    return r,facts,sig,exact

def response_obj(n=2):
    subjects=' and '.join(f'<Subject {i}>' for i in range(1,n+1))
    return {'summary':'[reference generation] '+subjects+' follow the human direction.',
            'retention_analysis':subjects+': fully_preserved - the requested reference roles.',
            'detailed_description':'Photorealism. [Shot 1] '+subjects+' perform the requested action.',
            'overall_soundscape':'N/A','non_diegetic_music':'N/A'}

def graph():return json.loads((ROOT/'config/workflows/26_H3_User_Directed_Ollama.json').read_text())
def kind(g,name):return next(n for n in g['nodes'] if n['type']==name)


class HumanMappingTests(unittest.TestCase):
    def test_same_label_not_number_of_photos_determines_identity(self):
        r=refs();self.assertEqual([x['subject'] for x in r['references']],[1,1,2])
    def test_case_and_spaces_normalized(self):
        r=d.collect({'reference_1':ref('Person A'),'reference_2':ref(' person   a ',.3)})
        self.assertEqual(len(r['subjects']),1)
    def test_content_and_role_preserved_verbatim(self):
        r=refs();s=d.definitions(r);self.assertIn('My exact description',s);self.assertIn('Primary face reference; ignore the outfit',s)
    def test_no_assumed_headshot_or_wardrobe_priority(self):
        r=refs();self.assertNotIn('headshot takes priority',d.definitions(r));self.assertNotIn('ONE PERSON ONLY',d.definitions(r))
    def test_scene_is_user_defined_subject(self):
        r=d.collect({'reference_1':ref('Person A'),'reference_2':ref('Room',.4)})
        self.assertIn('Room',d.definitions(r));self.assertEqual(len(r['subjects']),2)
    def test_shapes_and_objects_are_preserved(self):
        a=ref(h=32,w=64);b=ref(h=96,w=32);r=d.collect({'reference_1':a,'reference_2':b})
        self.assertIs(r['references'][0]['image'],a['image']);self.assertEqual(r['references'][1]['image'].shape,(1,96,32,3))
    def test_up_to_nine_refs(self):
        r=d.collect({f'reference_{i}':ref('Person A',i/10) for i in range(1,10)})
        self.assertEqual(len(r['references']),9);self.assertEqual(len(r['subjects']),1)
    def test_extra_slot_rejected(self):
        with self.assertRaises(ValueError):d.collect({'reference_10':ref()})
    def test_gap_rejected_not_silent_renumbering(self):
        with self.assertRaisesRegex(ValueError,'without gaps'):d.collect({'reference_1':ref(),'reference_3':ref()})
    def test_all_empty_rejected(self):
        with self.assertRaisesRegex(ValueError,'at least'):d.collect({})
    def test_blank_definitions_rejected(self):
        for field in ('owner','contains','role'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                d.make_reference(image(),'' if field=='owner' else 'A','' if field=='contains' else 'Face','' if field=='role' else 'Identity')
    def test_invalid_images_rejected(self):
        for bad in (None,torch.zeros(2,32,32,3),torch.zeros(1,16,16,3),torch.full((1,32,32,3),float('nan'))):
            with self.subTest(bad=type(bad)),self.assertRaises(ValueError):d.validate_image(bad)
    def test_same_picture_can_be_deliberately_assigned_to_different_content(self):
        r=d.collect({'reference_1':ref('Person A'),'reference_2':ref('Person B')})
        self.assertEqual(len(r['subjects']),2)  # no face-based deduplication overrides human declarations
    def test_unused_reference_ignores_empty_definition(self):
        self.assertEqual(p.DirectedH3Reference().define(None,'','',''),(None,))
    def test_empty_uploader_does_not_open_file(self):
        with mock.patch.object(p,'registered',side_effect=AssertionError):
            self.assertEqual(p.DirectedH3LoadImage().load('(none)'),(None,))
    def test_native_file_validation_not_bypassed(self):
        native=mock.Mock();native.VALIDATE_INPUTS.return_value='Bad filename'
        with mock.patch.object(p,'registered',return_value=native):
            self.assertEqual(p.DirectedH3LoadImage.VALIDATE_INPUTS('../bad'),'Bad filename')
    def test_owner_tags_rejected(self):
        with self.assertRaises(ValueError):d.owner_name('<Subject 99>')


class PromptContractTests(unittest.TestCase):
    def test_definitions_inserted_by_code_not_generated(self):
        r,_,_,ex=request();out=d.format_response(json.dumps(response_obj()),r,ex)
        self.assertTrue(out.startswith('subject_definitions:\n'+d.definitions(r)))
    def test_llm_cannot_supply_own_definitions(self):
        o=response_obj();o['subject_definitions']='other people';r,_,_,ex=request()
        with self.assertRaises(ValueError):d.format_response(json.dumps(o),r,ex)
    def test_generated_picture_mapping_rejected(self):
        o=response_obj();o['summary']+=' <Picture 1> is Person B.';r,_,_,ex=request()
        with self.assertRaises(ValueError):d.format_response(json.dumps(o),r,ex)
    def test_extra_subject_rejected(self):
        o=response_obj();o['summary']+=' <Subject 3>';r,_,_,ex=request()
        with self.assertRaises(ValueError):d.format_response(json.dumps(o),r,ex)
    def test_missing_subject_rejected(self):
        r,_,_,ex=request()
        with self.assertRaises(ValueError):d.format_response(json.dumps(response_obj(1)),r,ex)
    def test_missing_field_rejected(self):
        o=response_obj();del o['summary'];r,_,_,ex=request()
        with self.assertRaises(ValueError):d.format_response(json.dumps(o),r,ex)
    def test_raw_actions_preserved(self):
        r,_,_,ex=request();out=d.format_response(json.dumps(response_obj()),r,ex)
        self.assertIn(ex,out);self.assertIn('Person A and Person B hug.',out)
    def test_malformed_json_rejected(self):
        r,_,_,ex=request()
        with self.assertRaises(ValueError):d.format_response('not json',r,ex)
    def test_secondary_section_heading_rejected(self):
        o=response_obj();o['detailed_description']+='\nsubject_definitions:\nchange map';r,_,_,ex=request()
        with self.assertRaises(ValueError):d.format_response(json.dumps(o),r,ex)
    def test_unsupported_assets_rejected(self):
        for tag in ['<Video 1>','<Audio 1>','<Subject 0>','<Picture 4>']:
            with self.subTest(tag=tag),self.assertRaises(ValueError):d.validate_labels(tag,3,2)
    def test_bad_duration_or_canvas_rejected(self):
        for length,width,height in [(120,1344,768),(124,1345,768),(363,1344,768)]:
            with self.subTest(length=length,width=width),self.assertRaises(ValueError):
                d.check_request(refs(),'Hug','No sound','Keep identities',length,width,height)
    def test_review_can_edit_narrative(self):
        r,_,sig,ex=request();prompt=d.format_response(json.dumps(response_obj()),r,ex)
        reviewed=d.review_copy(prompt,sig).replace('perform the requested action.','perform the requested action with a brief pause.')
        self.assertIn('brief pause',d.read_review(reviewed,r,sig,ex))
    def test_changed_picture_stops_old_review(self):
        r,_,sig,ex=request();review=d.review_copy(d.format_response(json.dumps(response_obj()),r,ex),sig)
        _,_,new_sig,new_ex=request(d.collect({'reference_1':ref(v=.9),'reference_2':ref(),'reference_3':ref('Person B')}))
        with self.assertRaisesRegex(ValueError,'draft again'):d.read_review(review,r,new_sig,new_ex)
    def test_missing_review_header_rejected(self):
        r,_,sig,ex=request()
        with self.assertRaises(ValueError):d.read_review('some text',r,sig,ex)
    def test_edited_human_map_rejected(self):
        r,_,sig,ex=request();review=d.review_copy(d.format_response(json.dumps(response_obj()),r,ex),sig)
        with self.assertRaises(ValueError):d.read_review(review.replace('My exact description','Different face',1),r,sig,ex)
    def test_removed_human_direction_rejected(self):
        r,_,sig,ex=request();review=d.review_copy(d.format_response(json.dumps(response_obj()),r,ex),sig)
        with self.assertRaises(ValueError):d.read_review(review.replace(ex,''),r,sig,ex)
    def test_reviewed_mode_never_calls_llm(self):
        r,f,sig,ex=request();review=d.review_copy(d.format_response(json.dumps(response_obj()),r,ex),sig)
        with mock.patch.object(p,'ollama_modules',side_effect=AssertionError('no LLM')):
            job,view=p.DirectedH3Prompt().write(r,'Render reviewed prompt',f['actions'],f['scene_camera_audio'],
                f['preservation_rules'],124,1344,768,42,.2,False,False,d.SYSTEM,review)
        self.assertTrue(job['render']);self.assertNotIn('H3_DIRECT_INPUTS',job['prompt'])


class Response:
    def __init__(self,packets,code=200):self.packets=packets;self.status_code=code
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_lines(self,**kw):
        for packet in self.packets:yield json.dumps(packet).encode()

class LocalTransportTests(unittest.TestCase):
    def execute_fake(self,packets=None,include=False,code=200,interrupt=lambda:None,unload_error=None):
        r,f,sig,ex=request();client=mock.MagicMock();session=mock.MagicMock()
        client.BASE_URL='http://127.0.0.1:11434';client.session.return_value.__enter__.return_value=session
        client.model_info.return_value={'model':'m','capabilities':['vision','thinking'],'digest':'a'*64}
        client.encode_image.side_effect=[('one',''),('two',''),('three','')]
        session.post.return_value=Response(packets if packets is not None else [{'message':{'content':json.dumps(response_obj())},'done':True}],code)
        client.unload_and_wait.side_effect=unload_error
        cfg={'enabled':True,'model':'m','expected_model_digest':'','image_max_edge':1536,
             'max_output_tokens':4000,'request_timeout_seconds':900,'unload_timeout_seconds':90}
        try:
            out=d.generate(client,cfg,r,f,ex,d.SYSTEM,42,.2,include,interrupt)
        except BaseException:
            client.unload_and_wait.assert_called_once()
            raise
        client.unload_and_wait.assert_called_once()
        return out,session.post.call_args,client
    def test_default_writer_uses_only_human_descriptions(self):
        _,call,c=self.execute_fake();body=call.kwargs['json']
        self.assertNotIn('images',body['messages'][1]);c.encode_image.assert_not_called()
        self.assertIn('My exact description',body['messages'][1]['content'])
    def test_optional_visual_order_preserved(self):
        _,call,_=self.execute_fake(include=True)
        self.assertEqual(call.kwargs['json']['messages'][1]['images'],['one','two','three'])
    def test_local_redirects_disabled_keepalive_zero(self):
        _,call,_=self.execute_fake();self.assertEqual(call.args[0],'http://127.0.0.1:11434/api/chat')
        self.assertFalse(call.kwargs['allow_redirects']);self.assertEqual(call.kwargs['json']['keep_alive'],0)
        self.assertFalse(call.kwargs['json']['think'])
    def test_truncation_blocks_render_and_unloads(self):
        with self.assertRaises(ValueError):self.execute_fake([{'message':{'content':'{}'},'done':True,'done_reason':'length'}])
    def test_malformed_response_unloads(self):
        with self.assertRaises(ValueError):self.execute_fake([{'message':{'content':'bad'},'done':True}])
    def test_http_error_unloads(self):
        with self.assertRaises(RuntimeError):self.execute_fake(code=500)
    def test_cancel_unloads(self):
        def cancel():raise InterruptedError()
        with self.assertRaises(InterruptedError):self.execute_fake(interrupt=cancel)
    def test_failed_unload_blocks_handoff(self):
        with self.assertRaises(RuntimeError):self.execute_fake(unload_error=RuntimeError('still resident'))


class WorkflowTests(unittest.TestCase):
    def test_graph_passes_link_and_directed_checks(self):check_workflows.validate_graph(graph())
    def test_no_refmod_in_graph(self):
        self.assertFalse(any('RefMod' in n['type'] for n in graph()['nodes']))
    def test_models_and_quality_schedule_unchanged(self):
        g=graph();self.assertEqual(kind(g,'BasicScheduler')['widgets_values_named']['steps'],25)
        self.assertEqual(kind(g,'MiniMaxH3ReferenceToVideo')['widgets_values_named']['ref_image_size'],'max')
        self.assertEqual(p.MODEL_FILES[0],'minimax_h3_ref2va_bf16.safetensors')
    def test_native_video_vae_connection_is_present(self):
        g=graph();self.assertIsNotNone(next(x for x in kind(g,'MiniMaxH3ReferenceToVideo')['inputs'] if x['name']=='vae')['link'])
    def test_draft_gate_returns_only_blockers(self):
        class Blocker:
            def __init__(self,message):self.message=message
        fake=types.ModuleType('comfy_execution.graph_utils');fake.ExecutionBlocker=Blocker
        with mock.patch.dict(sys.modules,{'comfy_execution.graph_utils':fake}):
            out=p.DirectedH3RenderGate().release({'prompt':'valid draft','render':False})
        self.assertEqual(len(out),5);self.assertTrue(all(isinstance(x,Blocker) and x.message is None for x in out))
    def test_approved_gate_and_image_fanout_preserve_identity(self):
        r=refs();job={'references':r,'prompt':'Prompt','length':124,'width':1344,'height':768,'render':True}
        out=p.DirectedH3RenderGate().release(job);self.assertEqual(out[:4],p.MODEL_FILES);self.assertIs(out[4],job)
        imgs=p.DirectedH3OriginalImages().unpack(out[4])
        for i in range(3):self.assertIs(imgs[i],r['references'][i]['image'])
        self.assertEqual(imgs[3:9],(None,)*6);self.assertEqual(imgs[9:],('Prompt',1344,768,124))
    def test_originals_reject_unapproved_job(self):
        with self.assertRaises(ValueError):p.DirectedH3OriginalImages().unpack({'render':False})
    def test_missing_prompt_gate_rejected(self):
        with self.assertRaises(ValueError):p.DirectedH3RenderGate().release({'render':True})
    def test_directed_workflow_profile_is_h3(self):
        rt=json.loads((ROOT/'config/runtime.json').read_text())
        self.assertEqual(rt['workflow_profiles']['26_H3_User_Directed_Ollama.json'],'h3')
    def test_native_images_cannot_be_swapped(self):
        g=graph();h=kind(g,'MiniMaxH3ReferenceToVideo')
        a=next(i for i in h['inputs'] if i['name']=='ref_images.ref_image_0')
        b=next(i for i in h['inputs'] if i['name']=='ref_images.ref_image_1')
        a['link'],b['link']=b['link'],a['link']
        with self.assertRaises(ValueError):check_workflows.validate_graph(g)
    def test_prep_includes_new_suite_and_preserves_explicit_test_fix(self):
        text=(ROOT/'scripts/prepare_image.sh').read_text()
        self.assertIn('test_user_directed_h3',text);self.assertIn('CURRENT_TEST_MODULES',text)
        self.assertNotIn('unittest discover',text)
    def test_default_mode_builds_draft_job_without_render_permission(self):
        import threading
        r,f,sig,ex=request()
        fake_mm=types.ModuleType('comfy.model_management')
        fake_mm.throw_exception_if_processing_interrupted=mock.Mock()
        fake_mm.unload_all_models=mock.Mock();fake_mm.soft_empty_cache=mock.Mock()
        fake_comfy=types.ModuleType('comfy');fake_comfy.model_management=fake_mm
        pkg=types.SimpleNamespace(LOCK=threading.Lock());client=mock.Mock();client.settings.return_value={}
        with mock.patch.dict(sys.modules,{'comfy':fake_comfy,'comfy.model_management':fake_mm}), \
             mock.patch.object(p,'ollama_modules',return_value=(pkg,client)), \
             mock.patch.object(d,'generate',return_value=(d.format_response(json.dumps(response_obj()),r,ex),{})) as gen:
            job,view=p.DirectedH3Prompt().write(r,'Draft prompt only',f['actions'],f['scene_camera_audio'],
                f['preservation_rules'],124,1344,768,42,.2,False,False,d.SYSTEM,'')
        self.assertFalse(job['render']);self.assertIs(job['references'],r)
        self.assertTrue(view.startswith('[H3_DIRECT_INPUTS '));self.assertFalse(gen.call_args.args[8])
        fake_mm.unload_all_models.assert_called_once()

    def test_custom_workflow_widgets_match_actual_node_schemas(self):
        native=mock.Mock();native.INPUT_TYPES.return_value={'required':{'image':(['example.png'],{'image_upload':True})}}
        g=graph()
        with mock.patch.object(p,'registered',return_value=native):
            for n in g['nodes']:
                cls=p.NODE_CLASS_MAPPINGS.get(n['type'])
                if cls is None:continue
                schema=cls.INPUT_TYPES();allowed={**schema.get('required',{}),**schema.get('optional',{})}
                supplied={i['name'] for i in n['inputs']} | set(n['widgets_values_named'])
                self.assertTrue(set(schema['required']) <= supplied,n['type'])
                self.assertTrue((supplied - {'upload'}) <= set(allowed),n['type'])
                self.assertEqual([o['type'] for o in n['outputs']],list(cls.RETURN_TYPES))

    def test_system_instruction_honors_human_source_content(self):
        for word in ('read-only','same belongs_to','source content only','no images','human descriptions'):
            self.assertIn(word,d.SYSTEM)

if __name__=='__main__':unittest.main()
