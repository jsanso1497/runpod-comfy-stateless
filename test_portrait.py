import base64
import copy
import importlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import node
from node import logic,ollama_client as oc
import prepare_assets as assets
from smoke_check import validate_graph,validate_registry


def response(count=5):
    return {'references':[{'image':i,'subject':'Person A','contains':'A view of the same person','use_for':'Face' if i==1 else 'Body and clothing'} for i in range(1,count+1)],
            'summary':'The person walks slowly.', 'retention':'Preserve identity.',
            'shot':'[Shot 1] The person in <Picture 1> walks toward camera. Use <Picture 2> for the outfit.' if count>1 else '[Shot 1] The person in <Picture 1> blinks.',
            'sound':'Quiet room tone, no speech.','music':'N/A','clarification':''}


class LogicTests(unittest.TestCase):
    def test_aspect_ratios(self):
        for aspect in logic.ASPECTS:
            for preset in logic.PRESETS:
                r=logic.geometry(aspect,preset,5);a,b=map(int,aspect.split(':'))
                self.assertEqual(r['output_width']*b,r['output_height']*a)
                self.assertEqual(r['width']%32,0);self.assertEqual(r['height']%32,0)
                self.assertLessEqual(r['width']*r['height'],768*1344)
    def test_20_steps_default(self):self.assertEqual(logic.geometry('9:16','Standard',5)['steps'],20)
    def test_preview_12_steps(self):self.assertEqual(logic.geometry('2:3','Preview',5)['steps'],12)
    def test_quality_25_max(self):
        r=logic.geometry('9:16','High fidelity',5);self.assertEqual((r['steps'],r['ref_image_size']),(25,'max'))
    def test_duration_grid(self):
        for seconds in range(3,16):
            r=logic.geometry('9:16','Standard',seconds)
            self.assertEqual(r['length']%17,5);self.assertGreaterEqual(r['actual_seconds'],seconds)
    def test_not_unlimited_images(self):self.assertEqual(logic.MAX_REFERENCES,9)
    def test_missing_reference(self):
        d=response();d['references'].pop()
        with self.assertRaises(ValueError):logic.parse_llm(json.dumps(d),5)
    def test_duplicate_reference(self):
        d=response();d['references'][2]['image']=2
        with self.assertRaises(ValueError):logic.parse_llm(json.dumps(d),5)
    def test_bool_not_image_index(self):
        d=response(1);d['references'][0]['image']=True
        with self.assertRaises(ValueError):logic.parse_llm(json.dumps(d),1)
    def test_sort_references(self):
        d=response();d['references'].reverse()
        self.assertEqual(logic.parse_llm(json.dumps(d),5)['references'][0]['image'],1)
    def test_picture_mentions_are_allowed(self):logic.parse_llm(json.dumps(response()),5)
    def test_unknown_picture_rejected(self):
        d=response();d['shot']+=' <Picture 6>'
        with self.assertRaises(ValueError):logic.parse_llm(json.dumps(d),5)
    def test_unsupplied_audio_rejected(self):
        d=response();d['shot']+=' <Audio 1>'
        with self.assertRaises(ValueError):logic.parse_llm(json.dumps(d),5)
    def test_clarification_is_actionable(self):
        d=response();d['clarification']='Which person in image 3 should appear?'
        with self.assertRaises(logic.ClarificationNeeded):logic.parse_llm(json.dumps(d),5)
    def test_code_fence_tolerated(self):logic.parse_llm('```json\n'+json.dumps(response())+'\n```',5)
    def test_one_subject_multiple_views(self):
        text=logic.format_prompt(response(),'Images 1-5 are the same woman.',logic.geometry('9:16','Standard',5))
        self.assertIn('<Subject 1>',text);self.assertNotIn('<Subject 2>',text)
        self.assertIn('<Picture 5>',text)
    def test_multiple_subjects(self):
        d=response();d['references'][4]['subject']='Person B'
        self.assertIn('<Subject 2>',logic.format_prompt(d,'A hugs B',logic.geometry('2:3','Standard',5)))
    def test_no_user_grammar(self):
        brief='[SINGLE_PERSON_V1 primary_headshot] Same lady. 2-5 are body pics. Put her walking through a lobby.'
        text=logic.format_prompt(response(),brief,logic.geometry('9:16','Standard',5))
        self.assertIn(brief,text)
    def test_trigger_words_retained(self):
        self.assertTrue(logic.format_prompt(response(),'walk',logic.geometry('9:16','Standard',5),'MYTRIGGER').endswith('MYTRIGGER'))
    def test_new_prompt_variation_changes_cache(self):
        c={'image_max_edge':768,'context_length':32768};r=logic.geometry('9:16','Standard',5)
        self.assertNotEqual(logic.cache_key(['sha'],'walk',r,'d',c,'',0),logic.cache_key(['sha'],'walk',r,'d',c,'',1))
    def test_video_quality_does_not_rerun_same_brief(self):
        c={'image_max_edge':768,'context_length':32768}
        self.assertEqual(logic.cache_key(['sha'],'walk',logic.geometry('9:16','Standard',5),'d',c,'',0),logic.cache_key(['sha'],'walk',logic.geometry('9:16','High fidelity',5),'d',c,'',0))
    def test_empty_selection_does_not_apply_library(self):
        self.assertEqual(logic.active_loras(True,[]),[])
        self.assertEqual(logic.active_loras(False,[('anything',1)]),[])
    def test_select_only_named_lora(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'link-library.json').write_text(json.dumps({'loras':[
                {'lora_name':'a.safetensors','name':'A','trigger_words':'hero'},
                {'lora_name':'b.safetensors','name':'B','trigger_words':'paint'}]}))
            chosen=logic.active_loras(True,[('a.safetensors',.7),('(none)',1)],t)
            self.assertEqual(len(chosen),1)
            self.assertEqual(chosen[0]['strength'],.7)
            self.assertEqual(chosen[0]['trigger_words'],'hero')
    def test_duplicate_selection_rejected(self):
        with self.assertRaises(ValueError):
            logic.active_loras(True,[('a.safetensors',1),('a.safetensors',1)])
    def test_nan_strength_rejected(self):
        with self.assertRaises(ValueError):logic.active_loras(True,[('a',float('nan'))])
    def test_zero_strength_disables(self):
        self.assertEqual(logic.active_loras(True,[('a',0)]),[])

class TransportTests(unittest.TestCase):
    def cfg(self):
        return {'image_max_edge':768,'context_length':32768,
                'analysis_max_output_tokens':4096,'max_output_tokens':8192,
                'request_timeout_seconds':1200,'unload_timeout_seconds':90}
    def refs(self):return [{'image':'tensor','filename':'head.png'}]
    def info(self):return {'model':'test-thinking','capabilities':['vision','thinking'], 'analysis':{'model':'test-instruct','capabilities':['vision']}}
    def analysis(self):
        return {'references':response(1)['references'],'priorities':'Use the face.','conflicts':[],'clarification':''}
    def director(self):
        return {k:v for k,v in response(1).items() if k!='references'}
    def fake(self,objects):
        s=MagicMock();rs=[]
        for obj in objects:
            r=MagicMock();r.__enter__.return_value=r;r.status_code=200
            r.iter_lines.return_value=[json.dumps({'message':{'content':json.dumps(obj)},'done':True,'done_reason':'stop'}).encode()];rs.append(r)
        s.post.side_effect=rs;return s
    def call(self,s):
        return oc.generate(s,self.cfg(),self.info(),'system','whatever',self.refs(),logic.geometry('9:16','Standard',5),'',0)
    def test_loopback_session_not_environment_proxy(self):
        with oc.session() as s:self.assertFalse(s.trust_env)
    def test_endpoint_restricted(self):
        with self.assertRaises(ValueError):oc.api(MagicMock(),'https://elsewhere.test')
    @patch.object(oc,'preview',return_value='encoded-image')
    @patch.object(oc,'unload')
    def test_good_output_unloads(self,unload,preview):
        s=self.fake([self.analysis(),self.director()]);r=self.call(s)
        self.assertEqual(unload.call_count,2);self.assertEqual(r['references'][0]['image'],1)
        body=s.post.call_args.kwargs['json'];self.assertEqual(body['keep_alive'],0)
        self.assertNotIn('images',body['messages'][1])
        self.assertEqual(s.post.call_args_list[0].kwargs['json']['messages'][1]['images'],['encoded-image'])
        self.assertFalse(s.post.call_args.kwargs['allow_redirects'])
        self.assertEqual(s.post.call_count,2)
        self.assertEqual([c.kwargs['json'].get('think', False) for c in s.post.call_args_list],[False,True])
    @patch.object(oc,'preview',return_value='encoded')
    @patch.object(oc,'unload')
    def test_one_repair_then_success(self,unload,preview):
        s=self.fake([{},self.analysis(),self.director()]);self.call(s)
        self.assertEqual(s.post.call_count,3);self.assertEqual(unload.call_count,2)
    @patch.object(oc,'preview',return_value='encoded')
    @patch.object(oc,'unload')
    def test_no_infinite_retries(self,unload,preview):
        s=self.fake([{},{}])
        with self.assertRaises(RuntimeError):self.call(s)
        self.assertEqual(s.post.call_count,2);unload.assert_called_once()
    @patch.object(oc,'preview',return_value='encoded')
    @patch.object(oc,'unload')
    def test_clarification_does_not_retry(self,unload,preview):
        d=self.analysis();d['clarification']='Which outfit?';s=self.fake([d])
        with self.assertRaises(logic.ClarificationNeeded):self.call(s)
        self.assertEqual(s.post.call_count,1);unload.assert_called_once()
    @patch.object(oc,'api',side_effect=[{}, {'models':[]}])
    def test_unload_checks_residency(self,api):oc.unload(MagicMock(),'model');self.assertEqual(api.call_count,2)
    @patch.object(oc,'preview',return_value='encoded')
    @patch.object(oc,'unload',side_effect=RuntimeError('still resident'))
    def test_failed_cleanup_no_handoff(self,unload,preview):
        s=self.fake([self.analysis(),self.director()])
        with self.assertRaises(RuntimeError):self.call(s)


class GraphAndTensorTests(unittest.TestCase):
    def test_graph(self):validate_graph(json.loads((ROOT/'workflows/H3_Portrait_Auto.json').read_text()))
    def test_no_omni_refmod(self):
        graph=json.loads((ROOT/'workflows/H3_Portrait_Auto.json').read_text())
        self.assertFalse(any(x['type'].startswith(('Omni','MiniMaxH3RefMod')) for x in graph['nodes']))
    def test_model_loader_is_gated(self):
        g=json.loads((ROOT/'workflows/H3_Portrait_Auto.json').read_text())
        self.assertIn([2,2,0,3,0,'H3_PORTRAIT_JOB'],g['links'])
    def test_crop_is_not_stretch(self):
        import torch
        r=logic.geometry('9:16','Standard',5);x=torch.zeros(1,1344,768,3);x[:,:,6,:]=1
        y=node.H3PortraitExactAspect().crop(x,{'recipe':r})[0]
        self.assertEqual(tuple(y.shape),(1,1344,756,3));self.assertTrue(torch.all(y[:,:,0,:]==1))
    def test_2_3_no_crop(self):
        import torch
        r=logic.geometry('2:3','Standard',5);x=torch.zeros(1,1152,768,3)
        y=node.H3PortraitExactAspect().crop(x,{'recipe':r})[0];self.assertEqual(y.data_ptr(),x.data_ptr())
    def test_bad_decode_shape_fails(self):
        import torch
        with self.assertRaises(ValueError):node.H3PortraitExactAspect().crop(torch.zeros(1,16,16,3),{'recipe':logic.geometry('9:16','Standard',5)})
    def test_images_preserve_own_shapes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);Image.new('RGB',(30,70)).save(root/'a.png');Image.new('RGB',(60,20)).save(root/'b.png')
            fp=types.ModuleType('folder_paths');fp.get_input_directory=lambda:t
            with patch.dict(sys.modules,{'folder_paths':fp}):
                refs=node.H3PortraitReferences().load('["a.png","b.png"]')[0]
                self.assertEqual(refs[0]['image'].shape[1:3],(70,30));self.assertEqual(refs[1]['image'].shape[1:3],(20,60))
    def test_upload_path_traversal(self):
        fp=types.ModuleType('folder_paths');fp.get_input_directory=lambda:'/tmp'
        with patch.dict(sys.modules,{'folder_paths':fp}):
            with self.assertRaises(ValueError):node.input_paths('["../etc/passwd"]')
    def test_more_than_9_rejected(self):
        with patch.dict(sys.modules,{'folder_paths':types.ModuleType('folder_paths')}):
            with self.assertRaises(ValueError):node.input_paths(json.dumps(['x.png']*10))
    def test_empty_images_rejected(self):
        with patch.dict(sys.modules,{'folder_paths':types.ModuleType('folder_paths')}):
            with self.assertRaises(ValueError):node.input_paths('[]')
    def test_preview_downscale_does_not_mutate_original(self):
        import torch
        from PIL import Image
        x=torch.ones(1,80,40,3)*.5
        data=oc.preview(x,40);im=Image.open(io.BytesIO(base64.b64decode(data)))
        self.assertEqual(im.size,(20,40));self.assertEqual(x.shape[1:3],(80,40))
    def test_build_does_not_download_models(self):
        d=(ROOT/'Dockerfile').read_text();self.assertNotIn('RUN python /opt/h3-portrait/prepare_assets.py',d)
    def test_base_is_pinned(self):
        d=(ROOT/'Dockerfile').read_text()
        self.assertIn('ARG BASE_IMAGE=ghcr.io/jsanso1497/runpod-comfy-stateless@sha256:49d305',d)

if __name__=='__main__':unittest.main()
