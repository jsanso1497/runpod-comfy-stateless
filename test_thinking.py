"""Two-pass transport and policy regression tests. Network and generation mocked."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import node
from node import logic, ollama_client as oc
from test_portrait import response

ROOT = Path(__file__).resolve().parents[1]


def analysis(n=3):
    return {'references': response(n)['references'], 'priorities': 'Face from 1; clothes from 2.',
            'conflicts': [], 'clarification': ''}


def director(n=3):
    return {k: v for k, v in response(n).items() if k != 'references'}


class SchemaTests(unittest.TestCase):
    def test_analysis_accepts_all_nine(self):
        self.assertEqual(len(logic.parse_analysis(json.dumps(analysis(9)), 9)['references']), 9)
    def test_same_person_is_one_subject(self):
        text = logic.subject_definitions(analysis()['references'])
        self.assertEqual(text.count('<Subject 1>'), 1)
        self.assertNotIn('<Subject 2>', text)
        for i in range(1, 4):self.assertIn(f'<Picture {i}>', text)
    def test_distinct_subjects_are_not_merged(self):
        obj = analysis(); obj['references'][2]['subject'] = 'Person B'
        self.assertIn('<Subject 2> (Person B)', logic.subject_definitions(obj['references']))
    def test_no_standalone_picture_definitions(self):
        self.assertTrue(all(line.startswith('<Subject ') for line in logic.subject_definitions(analysis()['references']).splitlines()))
    def test_analysis_detects_duplicate_index(self):
        obj = analysis(); obj['references'][2]['image'] = 2
        with self.assertRaises(ValueError):logic.parse_analysis(json.dumps(obj), 3)
    def test_analysis_detects_missing_ref(self):
        obj = analysis(); obj['references'].pop()
        with self.assertRaises(ValueError):logic.parse_analysis(json.dumps(obj), 3)
    def test_analysis_rejects_bool_index(self):
        obj = analysis(1); obj['references'][0]['image'] = True
        with self.assertRaises(ValueError):logic.parse_analysis(json.dumps(obj), 1)
    def test_analysis_clarification_stops(self):
        obj = analysis(); obj['clarification'] = 'Which person in image 2?'
        with self.assertRaises(logic.ClarificationNeeded):logic.parse_analysis(json.dumps(obj), 3)
    def test_director_cannot_return_new_map(self):
        obj = director();obj['references'] = analysis()['references']
        with self.assertRaises(ValueError):logic.parse_director(json.dumps(obj), analysis(), 3)
    def test_director_keeps_map_by_copy(self):
        src = analysis(); value = logic.parse_director(json.dumps(director()), src, 3)
        self.assertEqual(value['references'], src['references'])
        value['references'][0]['subject'] = 'Changed'
        self.assertEqual(src['references'][0]['subject'], 'Person A')
    def test_normal_picture_mentions_valid(self):
        obj = director();obj['shot'] += ' Use <Picture 3> for proportions.'
        logic.parse_director(json.dumps(obj), analysis(), 3)
    def test_nonsupplied_picture_stops(self):
        obj = director();obj['shot'] += ' Use <Picture 4>.'
        with self.assertRaises(ValueError):logic.parse_director(json.dumps(obj), analysis(), 3)
    def test_nonsupplied_subject_stops(self):
        obj = director();obj['shot'] += ' <Subject 2> walks.'
        with self.assertRaises(ValueError):logic.parse_director(json.dumps(obj), analysis(), 3)
    def test_priorities_conflicts_bounded(self):
        for change in ({'priorities': 3}, {'conflicts': ['x']*13}, {'conflicts': [False]}):
            obj = dict(analysis(), **change)
            with self.assertRaises(ValueError):logic.parse_analysis(json.dumps(obj), 3)
    def test_style_introduction_remains_before_shot(self):
        obj = response(1);obj['shot'] = 'Natural light and realistic textures.\n[Shot 1] The person blinks.'
        prompt = logic.format_prompt(obj, 'Blink, nothing else.', logic.geometry('9:16','Standard',5))
        self.assertIn('Natural light and realistic textures.\n[Shot 1]', prompt)
        self.assertEqual(prompt.count('[Shot 1]'), 1)
    def test_freeform_user_text_preserved(self):
        brief = '1-5 are one woman. [SINGLE_PERSON_V1 head] Keep her outfit. Walk!'
        self.assertIn(brief, logic.format_prompt(response(5),brief,logic.geometry('2:3','Standard',5)))


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'settings.json').read_text())
        self.info = {'model': self.cfg['ollama_model'], 'digest':'digest', 'capabilities':['vision','thinking']}
        self.refs = [{'filename': f'image-{i}.png','image':object()} for i in range(1,4)]
        self.previews = patch.object(oc,'preview',side_effect=['img1','img2','img3'])
        self.previews.start(); self.addCleanup(self.previews.stop)
        self.unload_patch = patch.object(oc,'unload')
        self.unload = self.unload_patch.start();self.addCleanup(self.unload_patch.stop)
    def response(self,obj,thinking='do not expose this reasoning',reason='stop',done=True):
        r = MagicMock();r.status_code=200;r.__enter__.return_value=r
        packets=[{'message':{'thinking':thinking, 'content':''}},
                 {'message':{'content':json.dumps(obj)},'done':done,'done_reason':reason,'eval_count':100}]
        r.iter_lines.return_value=[json.dumps(x).encode() for x in packets]
        return r
    def session(self,*values):
        s=MagicMock();s.post.side_effect=[self.response(x) for x in values];return s
    def call(self,s,interrupt=lambda:None):
        return oc.generate(s,self.cfg,self.info,(ROOT/'node/system_prompt.txt').read_text(),
                           'All photos show the same person. She walks.',self.refs,
                           logic.geometry('9:16','Standard',5),'TRIGGER',0,interrupt)
    def test_two_calls_receive_all_images_same_order(self):
        s=self.session(analysis(),director());self.call(s)
        self.assertEqual(s.post.call_count,2)
        for c in s.post.call_args_list:self.assertEqual(c.kwargs['json']['messages'][1]['images'],['img1','img2','img3'])
    def test_think_and_sampling_options_in_both_calls(self):
        s=self.session(analysis(),director());self.call(s)
        for c in s.post.call_args_list:
            b=c.kwargs['json'];self.assertIs(b['think'],True)
            self.assertEqual((b['options']['num_ctx'],b['options']['temperature'],b['options']['top_p'],b['options']['top_k']), (32768,.25,.9,20))
        self.assertEqual(s.post.call_args_list[0].kwargs['json']['options']['num_predict'],8192)
        self.assertEqual(s.post.call_args_list[1].kwargs['json']['options']['num_predict'],12288)
    def test_keepalive_between_passes_unload_after_last(self):
        s=self.session(analysis(),director());self.call(s)
        self.assertEqual([c.kwargs['json']['keep_alive'] for c in s.post.call_args_list],['5m',0])
        self.unload.assert_called_once()
    def test_thinking_not_in_result_or_second_chat(self):
        s=self.session(analysis(),director());obj=self.call(s)
        self.assertNotIn('do not expose this reasoning',json.dumps(obj))
        self.assertNotIn('do not expose this reasoning',json.dumps(s.post.call_args_list[1].kwargs['json']))
        self.assertTrue(all(x['thinking_seen'] for x in obj['_stages']))
    def test_second_chat_has_locked_subject_mapping(self):
        s=self.session(analysis(),director());self.call(s)
        content=s.post.call_args_list[1].kwargs['json']['messages'][1]['content']
        self.assertIn('LOCKED SUBJECT MAP',content);self.assertIn('<Subject 1> (Person A)',content)
        self.assertNotIn('references',s.post.call_args_list[1].kwargs['json']['format']['properties'])
    def test_analysis_clarification_prevents_director(self):
        a=analysis();a['clarification']='Which outfit?';s=self.session(a)
        with self.assertRaises(logic.ClarificationNeeded):self.call(s)
        self.assertEqual(s.post.call_count,1);self.unload.assert_called_once()
    def test_director_clarification_stops(self):
        d=director();d['clarification']='Did you intend Person B?';s=self.session(analysis(),d)
        with self.assertRaises(logic.ClarificationNeeded):self.call(s)
        self.unload.assert_called_once()
    def test_director_json_repair_cannot_mutate_map(self):
        s=self.session(analysis(),dict(director(),references=[]),director());obj=self.call(s)
        self.assertEqual(s.post.call_count,3);self.assertEqual(obj['references'],analysis()['references'])
    def test_bounded_repair_each_pass(self):
        s=self.session({},analysis(),{},director());obj=self.call(s)
        self.assertEqual(s.post.call_count,4);self.assertEqual([s['attempts'] for s in obj['_stages']],[2,2])
    def test_final_failed_director_unloads(self):
        s=self.session(analysis(),{}, {})
        with self.assertRaises(RuntimeError):self.call(s)
        self.assertEqual(s.post.call_count,3);self.unload.assert_called_once()
    def test_length_truncation_no_handoff_or_repair(self):
        s=MagicMock();s.post.return_value=self.response(analysis(),reason='length')
        with self.assertRaisesRegex(RuntimeError,'budget'):self.call(s)
        self.assertEqual(s.post.call_count,1);self.unload.assert_called_once()
    def test_missing_done_is_not_success(self):
        s=MagicMock();s.post.return_value=self.response(analysis(),done=False)
        with self.assertRaisesRegex(RuntimeError,'before completion'):self.call(s)
        self.unload.assert_called_once()
    def test_http_error_unloads(self):
        s=MagicMock();r=self.response(analysis());r.status_code=500;s.post.return_value=r
        with self.assertRaisesRegex(RuntimeError,'HTTP 500'):self.call(s)
        self.unload.assert_called_once()
    def test_stream_error_unloads(self):
        s=MagicMock();r=self.response(analysis());r.iter_lines.return_value=[b'{"error":"out of memory"}'];s.post.return_value=r
        with self.assertRaises(RuntimeError):self.call(s)
        self.unload.assert_called_once()
    def test_cancel_before_encoding_unloads(self):
        s=MagicMock()
        with self.assertRaises(KeyboardInterrupt):self.call(s,MagicMock(side_effect=KeyboardInterrupt))
        self.unload.assert_called_once();s.post.assert_not_called()
    def test_unload_failure_blocks_valid_prompt(self):
        self.unload.side_effect=RuntimeError('still resident');s=self.session(analysis(),director())
        with self.assertRaisesRegex(RuntimeError,'still resident'):self.call(s)
    def test_cleanup_failure_preserves_original_error(self):
        self.unload.side_effect=RuntimeError('still resident');a=analysis();a['clarification']='Which person?';s=self.session(a)
        with self.assertRaises(logic.ClarificationNeeded):self.call(s)
    def test_instruct_rejected_before_chat(self):
        self.info['capabilities']=['vision'];s=MagicMock()
        with self.assertRaisesRegex(RuntimeError,'THINKING'):self.call(s)
        s.post.assert_not_called()
    def test_final_payload_limits_are_enforced(self):
        s=MagicMock();r=self.response(analysis());r.iter_lines.return_value=[json.dumps({'message':{'content':'x'*100001}}).encode()];s.post.return_value=r
        with self.assertRaisesRegex(RuntimeError,'size limit'):self.call(s)
        self.unload.assert_called_once()
    def test_no_true_reasoning_content_saved_in_prompt(self):
        s=self.session(analysis(),director());obj=self.call(s)
        text=logic.format_prompt(obj,'Walk.',logic.geometry('9:16','Standard',5))
        self.assertNotIn('_stages',text);self.assertNotIn('thinking',text)


class ConfigurationTests(unittest.TestCase):
    def test_model_defaults_agree(self):
        cfg=json.loads((ROOT/'settings.json').read_text())
        self.assertEqual(cfg['ollama_model'],logic.DEFAULT_MODEL)
        self.assertIn('OLLAMA_MODEL='+logic.DEFAULT_MODEL,(ROOT/'Dockerfile').read_text())
        self.assertIs(cfg['think'],True)
    def test_no_thinking_false_request_in_code(self):
        source=(ROOT/'node/ollama_client.py').read_text()
        self.assertNotIn("body['think']=False",source)
    def test_show_capability_compatibility(self):
        oc.require_thinking({'capabilities':['vision','thinking']})
        oc.require_thinking({'capabilities':['vision'],'thinking':{'values':[True,False]}})
        with self.assertRaises(RuntimeError):oc.require_thinking({'capabilities':['thinking'],'thinking':{'values':[False]}})
    @patch.object(oc,'api')
    def test_model_info_checks_vision_thinking_and_digest(self,api):
        api.side_effect=[{'models':[{'name':logic.DEFAULT_MODEL,'digest':'abc'}]}, {'capabilities':['vision','thinking']}]
        self.assertEqual(oc.model_info(MagicMock(),logic.DEFAULT_MODEL)['digest'],'abc')
    @patch.object(oc,'api')
    def test_remote_model_rejected(self,api):
        api.side_effect=[{'models':[{'name':'model:latest'}]}, {'capabilities':['vision','thinking'],'remote_host':'x'}]
        with self.assertRaisesRegex(RuntimeError,'local'):oc.model_info(MagicMock(),'model')
    def test_cache_changes_with_policy(self):
        cfg=json.loads((ROOT/'settings.json').read_text());recipe=logic.geometry('9:16','Standard',5)
        before=logic.cache_key(['x'],'walk',recipe,'digest',cfg,'',0)
        with patch.object(logic,'prompt_policy',return_value={'system_prompt.txt':'new','analysis_prompt.txt':'new'}):
            self.assertNotEqual(before,logic.cache_key(['x'],'walk',recipe,'digest',cfg,'',0))
    def test_cache_changes_with_temperature(self):
        cfg=json.loads((ROOT/'settings.json').read_text());recipe=logic.geometry('2:3','Standard',5)
        before=logic.cache_key(['x'],'walk',recipe,'digest',cfg,'',0)
        cfg['temperature']=.3
        self.assertNotEqual(before,logic.cache_key(['x'],'walk',recipe,'digest',cfg,'',0))
    def test_cache_changes_with_model_digest(self):
        cfg=json.loads((ROOT/'settings.json').read_text());recipe=logic.geometry('2:3','Standard',5)
        self.assertNotEqual(logic.cache_key(['x'],'walk',recipe,'old',cfg,'',0),logic.cache_key(['x'],'walk',recipe,'new',cfg,'',0))
    def test_settings_reject_non_thinking(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=json.loads((ROOT/'settings.json').read_text());cfg['think']=False
            (Path(td)/'settings.json').write_text(json.dumps(cfg))
            with patch.object(logic,'ROOT',Path(td)),patch.dict(os.environ,{},clear=True):
                with self.assertRaisesRegex(ValueError,'think=true'):logic.settings()
    def test_prompt_files_read_without_reload(self):
        first=node.H3PortraitDirector
        cfg=json.loads((ROOT/'settings.json').read_text())
        with patch.object(logic,'settings',return_value=cfg):
            key=first.IS_CHANGED()
            with patch.object(logic,'prompt_policy',return_value={'system_prompt.txt':'new'}):
                self.assertNotEqual(key,first.IS_CHANGED())
    def test_mininmax_instruction_terms_present(self):
        text=(ROOT/'node/system_prompt.txt').read_text()
        for phrase in ('350-500','[Shot 1]','fully_preserved','partially_preserved','attribute_transfer','weak_reference','non_diegetic_music'):
            self.assertIn(phrase,text)
    def test_resolutions_steps_unchanged(self):
        for name,steps in [('Preview',12),('Standard',20),('High fidelity',25)]:
            for aspect in ('9:16','2:3'):
                self.assertEqual(logic.geometry(aspect,name,5)['steps'],steps)
    def test_ui_contract_does_not_add_widgets(self):
        import types
        fp=types.ModuleType('folder_paths');fp.get_filename_list=lambda name:[]
        with patch.dict(sys.modules,{'folder_paths':fp}):
            schema=node.H3PortraitDirector.INPUT_TYPES()
        self.assertEqual(list(schema['required']),['references','instruction','aspect','quality','seconds','seed','use_loras','mode','prompt_variation'])
        self.assertEqual(list(schema['optional']),['lora_1','strength_1','lora_2','strength_2','extra_trigger_words'])


if __name__=='__main__':unittest.main()
