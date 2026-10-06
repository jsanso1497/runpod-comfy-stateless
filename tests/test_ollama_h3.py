from __future__ import annotations
import base64
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'scripts/local_nodes/ComfyUI-OllamaH3').is_dir():
    ROOT = Path('/opt/runpod-comfy')
NODE_DIR = ROOT / 'scripts/local_nodes/ComfyUI-OllamaH3'
CONFIG = ROOT / 'config' if (ROOT/'config').is_dir() else ROOT/'default-config'
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('test_ollama_nodepack',NODE_DIR/'__init__.py',submodule_search_locations=[str(NODE_DIR)])
nodes=importlib.util.module_from_spec(spec);sys.modules[spec.name]=nodes;spec.loader.exec_module(nodes)
client=sys.modules[spec.name+'.client']


def content(n=2):
    data={key:'N/A' for key in client.FIELDS}
    data.update(subject_definitions='\n'.join(f'<Subject {i}> is the person in <Picture {i}>.' for i in range(1,n+1)),
                summary='[reference generation] A short meeting.', retention_analysis='<Subject 1>: fully_preserved - intended appearance.',
                detailed_description='Natural photography. [Shot 1] The requested action unfolds.')
    return json.dumps(data)


class Response:
    def __init__(self, body=None, code=200, packets=None):
        self.body=body;self.status_code=code;self.packets=packets or []
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def json(self):return self.body
    def iter_lines(self,**kw):
        for p in self.packets:yield json.dumps(p).encode()


class PromptTests(unittest.TestCase):
    def test_six_sections_in_order(self):
        t=client.format_prompt(content(),2)
        locations=[t.index(k+':') for k in client.FIELDS]
        self.assertEqual(locations,sorted(locations))
    def test_one_reference_valid(self):client.format_prompt(content(1),1)
    def test_missing_picture_rejected(self):
        with self.assertRaises(ValueError):client.format_prompt(content(1),2)
    def test_extra_picture_rejected(self):
        with self.assertRaises(ValueError):client.format_prompt(content(3),2)
    def test_audio_reference_rejected(self):
        with self.assertRaises(ValueError):client.validate_tags('<Picture 1> and <Audio 1>',1)
    def test_video_reference_rejected(self):
        with self.assertRaises(ValueError):client.validate_tags('<Picture 1> and <Video 1>',1)
    def test_negative_picture_index_rejected(self):
        with self.assertRaises(ValueError):client.validate_tags('<Picture 0>',1)
    def test_code_fences_rejected(self):
        with self.assertRaises(ValueError):client.validate_tags('``` <Picture 1> ```',1)
    def test_missing_section_rejected(self):
        obj=json.loads(content());del obj['summary']
        with self.assertRaises(ValueError):client.format_prompt(json.dumps(obj),2)
    def test_nonstring_section_rejected(self):
        obj=json.loads(content());obj['summary']=['wrong']
        with self.assertRaises(ValueError):client.format_prompt(json.dumps(obj),2)
    def test_truncated_json_rejected(self):
        with self.assertRaises(ValueError):client.format_prompt(content()[:100],2)
    def test_shot_marker_required(self):
        obj=json.loads(content());obj['detailed_description']='No shot marker'
        with self.assertRaises(ValueError):client.format_prompt(json.dumps(obj),2)
    def test_no_unrequested_task_type(self):
        obj=json.loads(content());obj['summary']='[video editing] A new video.'
        with self.assertRaises(ValueError):client.format_prompt(json.dumps(obj),2)
    def test_json_schema_exact_fields(self):
        self.assertEqual(set(client.SCHEMA['required']),set(client.FIELDS))
        self.assertFalse(client.SCHEMA['additionalProperties'])


class RuntimeTests(unittest.TestCase):
    def test_loopback_only(self):self.assertEqual(client.BASE_URL,'http://127.0.0.1:11434')
    def test_local_session_ignores_proxy_env(self):
        with client.session() as s:self.assertFalse(s.trust_env)
    def test_redirect_rejected(self):
        s=mock.Mock();s.get.return_value=Response(code=302)
        with self.assertRaises(RuntimeError):client.api(s,'/api/version')
        self.assertFalse(s.get.call_args.kwargs['allow_redirects'])
    def test_invalid_endpoint_rejected(self):
        with self.assertRaises(ValueError):client.api(mock.Mock(),'https://example.com')
    def test_cloud_override_rejected(self):
        with mock.patch.dict(os.environ,{'OLLAMA_MODEL':'qwen3-vl:235b-cloud'}),self.assertRaises(ValueError):client.settings()
    def test_bad_model_digest_rejected(self):
        with mock.patch.dict(os.environ,{'OLLAMA_MODEL_DIGEST':'bad'}),self.assertRaises(ValueError):client.settings()
    def test_disabled_supported(self):
        with mock.patch.dict(os.environ,{'ENABLE_OLLAMA':'0'}):self.assertFalse(client.settings()['enabled'])
    def test_nonvision_rejected(self):
        rows={'models':[{'name':'qwen3-vl:8b-instruct','digest':'a'*64}]}
        with mock.patch.object(client,'api',side_effect=[rows,{'capabilities':['completion']}]),self.assertRaises(RuntimeError):
            client.model_info(mock.Mock(),'qwen3-vl:8b-instruct')
    def test_digest_mismatch_rejected(self):
        rows={'models':[{'name':'qwen3-vl:8b-instruct','digest':'a'*64}]}
        with mock.patch.object(client,'api',return_value=rows),self.assertRaises(RuntimeError):
            client.model_info(mock.Mock(),'qwen3-vl:8b-instruct','b'*64)
    def test_unload_waits_for_absence(self):
        with mock.patch.object(client,'api',side_effect=[{}, {'models':[{'name':'m:1'}]},{'models':[]}]) as a,mock.patch.object(client.time,'sleep'):
            client.unload_and_wait(mock.Mock(),'m:1',5)
        self.assertEqual(a.call_count,3)
        self.assertEqual(a.call_args_list[0].args[2]['keep_alive'],0)
    def run_fake(self, packets, interrupt=lambda:None):
        s=mock.MagicMock();s.__enter__.return_value=s;s.post.return_value=Response(packets=packets)
        cfg=dict(client.DEFAULTS)
        with mock.patch.object(client,'session',return_value=s),mock.patch.object(client,'model_info',return_value={'model':cfg['model'],'digest':'a'*64,'capabilities':['vision']}),mock.patch.object(client,'encode_image',side_effect=[('first','hash1'),('second','hash2')]),mock.patch.object(client,'unload_and_wait') as unload:
            try:result=client.generate(cfg,client.SYSTEM_PROMPT,'A hug', [1,2],124,42,0.2,interrupt)
            except BaseException:
                self.assertTrue(unload.called)
                raise
            self.assertTrue(unload.called)
        return result,s.post.call_args.kwargs['json']
    def test_images_order_and_keepalive(self):
        result,body=self.run_fake([{'message':{'content':content()},'done':True}])
        self.assertEqual(body['messages'][1]['images'],['first','second'])
        self.assertEqual(body['keep_alive'],0)
        self.assertIn('image 2 = <Picture 2>',body['messages'][1]['content'])
        self.assertIn('subject_definitions:',result[0])
    def test_length_limit_stops_handoff_and_unloads(self):
        with self.assertRaises(ValueError):self.run_fake([{'message':{'content':content()},'done':True,'done_reason':'length'}])
    def test_malformed_output_unloads(self):
        with self.assertRaises(ValueError):self.run_fake([{'message':{'content':'invalid'},'done':True}])
    def test_interruption_unloads(self):
        def interrupted():raise InterruptedError('cancelled')
        with self.assertRaises(InterruptedError):self.run_fake([],interrupted)
    def test_manual_mode_never_calls_ollama(self):
        import torch
        a,b=torch.zeros(1,8,8,3),torch.ones(1,8,8,3)
        with mock.patch.object(nodes,'generate',side_effect=AssertionError('no request')):
            result=nodes.EverydayOllamaH3Prompt().build_prompt(a,nodes.MODES[1],'','','<Picture 1> hugs <Picture 2>',124,42,0.2,False,b)
        self.assertIs(result[1],a);self.assertIs(result[2],b);self.assertEqual(result[3],124)
    def test_blank_manual_prompt_blocks_h3(self):
        import torch
        with self.assertRaises(ValueError):nodes.EverydayOllamaH3Prompt().build_prompt(torch.zeros(1,8,8,3),nodes.MODES[1],'','','',124,42,0.2,False)
    def test_single_reference_passthrough_none(self):
        import torch
        a=torch.zeros(1,8,8,3)
        r=nodes.EverydayOllamaH3Prompt().build_prompt(a,nodes.MODES[1],'','','Use <Picture 1>.',124,42,0.2,False)
        self.assertIsNone(r[2])
    def test_image_resize_aspect_and_strip_metadata(self):
        import torch
        from PIL import Image
        raw,sha=client.encode_image(torch.zeros(1,24,48,3),16)
        im=Image.open(io.BytesIO(base64.b64decode(raw)))
        self.assertEqual(im.size,(16,8));self.assertEqual(im.info,{})
        self.assertEqual(len(sha),64)
    def test_video_batch_rejected(self):
        import torch
        with self.assertRaises(ValueError):client.encode_image(torch.zeros(2,8,8,3))
    def test_bad_frame_grid_rejected(self):
        import torch
        with self.assertRaises(ValueError):nodes.EverydayOllamaH3Prompt().build_prompt(torch.zeros(1,8,8,3),nodes.MODES[1],'','','<Picture 1>',120,42,0.2,False)


class GraphTests(unittest.TestCase):
    def test_new_graphs_validate(self):
        import check_workflows
        for name in ['05_Vision_Prompt_Draft.json','21_H3_Ollama_Two_Refs_BF16.json']:
            check_workflows.validate_graph(json.loads((CONFIG/'workflows'/name).read_text()))
    def test_draft_has_no_h3_weights_or_sampling(self):
        g=json.loads((CONFIG/'workflows/05_Vision_Prompt_Draft.json').read_text())
        types={n['type'] for n in g['nodes']}
        self.assertFalse(types & {'UNETLoader','KSampler','SamplerCustomAdvanced','MiniMaxH3ReferenceToVideo'})
    def test_integrated_defaults_reviewed_not_auto(self):
        g=json.loads((CONFIG/'workflows/21_H3_Ollama_Two_Refs_BF16.json').read_text())
        n=next(n for n in g['nodes'] if n['type']=='EverydayOllamaH3Prompt')
        self.assertEqual(n['widgets_values_named']['mode'],nodes.MODES[1])
    def test_model_names_are_combo_sockets(self):
        self.assertEqual(nodes.EverydayH3FilesAfterPrompt.RETURN_TYPES,('COMBO',)*4)
    def test_draft_only_is_shared(self):
        r=json.loads((CONFIG/'runtime.json').read_text())
        self.assertEqual(r['workflow_profiles']['05_Vision_Prompt_Draft.json'],'shared')
        self.assertEqual(r['workflow_profiles']['21_H3_Ollama_Two_Refs_BF16.json'],'h3')


if __name__=='__main__':unittest.main(verbosity=2)
