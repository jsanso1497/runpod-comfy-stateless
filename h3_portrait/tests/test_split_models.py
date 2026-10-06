"""Model-edition routing regressions. Uses mocked Ollama, never live inference."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from node import logic, ollama_client as oc
from test_thinking import analysis, director


def pipeline():
    return {'model':logic.DEFAULT_MODEL,'digest':'director-v1',
            'capabilities':['vision','thinking'],'thinking':{'values':[True]},
            'analysis':{'model':logic.DEFAULT_ANALYSIS_MODEL,'digest':'analysis-v1',
                        'capabilities':['vision']}}


def packet(value,reason='stop'):
    r=MagicMock(status_code=200)
    r.__enter__.return_value=r
    r.iter_lines.return_value=[json.dumps({'message':{'content':json.dumps(value)},
                                          'done':True,'done_reason':reason}).encode()]
    return r


class SplitModelsTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'settings.json').read_text())
        self.info=pipeline()
        self.refs=[{'filename':str(i)+'.png','image':object()} for i in range(3)]
        self.recipe=logic.geometry('9:16','Standard',5)
    def call(self,s):
        return oc.generate(s,self.cfg,self.info,'system','All images are the same person. Walk.',self.refs,self.recipe,'trigger',0)
    def test_distinct_verified_model_tags(self):
        self.assertEqual(self.cfg['analysis_model'],logic.DEFAULT_ANALYSIS_MODEL)
        self.assertIn('32b-instruct-q4_K_M',self.cfg['analysis_model'])
        self.assertIn('32b-thinking-q4_K_M',self.cfg['ollama_model'])
        self.assertNotEqual(self.cfg['analysis_model'],self.cfg['ollama_model'])
    def test_thinking_only_support_is_accepted_for_director(self):
        oc.require_thinking(self.info)
        oc.require_control(self.info,True)
    def test_analysis_never_uses_a_thinking_edition(self):
        with self.assertRaisesRegex(RuntimeError,'Instruct'):
            oc.require_non_thinking(self.info)
    def test_pipeline_digest_tracks_analysis_version(self):
        changed=copy.deepcopy(self.info);changed['analysis']['digest']='analysis-v2'
        self.assertNotEqual(oc.pipeline_digest(self.info),oc.pipeline_digest(changed))
    def test_pipeline_digest_tracks_director_version(self):
        changed=copy.deepcopy(self.info);changed['digest']='director-v2'
        self.assertNotEqual(oc.pipeline_digest(self.info),oc.pipeline_digest(changed))
    @patch.object(oc,'model_info')
    def test_pipeline_info_checks_each_role(self,get):
        get.side_effect=[{k:v for k,v in self.info.items() if k!='analysis'},self.info['analysis']]
        info=oc.pipeline_info(MagicMock(),self.cfg)
        self.assertEqual([c.kwargs['role'] for c in get.call_args_list],['director','analysis'])
        self.assertEqual(info['analysis']['model'],logic.DEFAULT_ANALYSIS_MODEL)
    @patch.object(oc,'api')
    def test_instruct_show_does_not_need_thinking_capability(self,api):
        api.side_effect=[{'models':[{'name':logic.DEFAULT_ANALYSIS_MODEL,'digest':'i'}]}, {'capabilities':['vision']}]
        self.assertEqual(oc.model_info(MagicMock(),logic.DEFAULT_ANALYSIS_MODEL,role='analysis')['digest'],'i')
    @patch.object(oc,'preview',return_value='preview')
    def test_model_switch_happens_after_analysis_unload(self,preview):
        events=[];responses=iter([packet(analysis()),packet(director())]);s=MagicMock()
        def send(*a,**k):
            events.append(('chat',k['json']['model']));return next(responses)
        def unload(*a,**k):events.append(('unload',a[1]))
        s.post.side_effect=send
        with patch.object(oc,'unload',side_effect=unload):self.call(s)
        self.assertEqual(events,[('chat',logic.DEFAULT_ANALYSIS_MODEL),('unload',logic.DEFAULT_ANALYSIS_MODEL),('chat',logic.DEFAULT_MODEL),('unload',logic.DEFAULT_MODEL)])
        first,last=[c.kwargs['json'] for c in s.post.call_args_list]
        self.assertNotIn('think',first)
        self.assertIs(last['think'],True)
        self.assertFalse(any('images' in m for m in last['messages']))
    @patch.object(oc,'preview',return_value='preview')
    def test_director_truncation_uses_instruct_repair_not_unsupported_toggle(self,preview):
        events=[];responses=iter([packet(analysis()),packet({},'length'),packet(director())]);s=MagicMock()
        def send(*a,**k):
            events.append(('chat',k['json']['model']));return next(responses)
        s.post.side_effect=send
        with patch.object(oc,'unload',side_effect=lambda *a,**k:events.append(('unload',a[1]))):
            obj=self.call(s)
        self.assertEqual(events,[('chat',logic.DEFAULT_ANALYSIS_MODEL),('unload',logic.DEFAULT_ANALYSIS_MODEL),('chat',logic.DEFAULT_MODEL),('unload',logic.DEFAULT_MODEL),('chat',logic.DEFAULT_ANALYSIS_MODEL),('unload',logic.DEFAULT_ANALYSIS_MODEL)])
        repair=s.post.call_args_list[-1].kwargs['json']
        self.assertNotIn('think',repair)
        self.assertTrue(all('images' not in m for m in repair['messages']))
        self.assertEqual(obj['references'],analysis()['references'])
        self.assertEqual(obj['_stages'][1]['model'],logic.DEFAULT_ANALYSIS_MODEL)
    @patch.object(oc,'preview',return_value='preview')
    @patch.object(oc,'unload')
    def test_analysis_failure_never_loads_director(self,unload,preview):
        s=MagicMock();s.post.side_effect=[packet({},'length'),packet({},'length')]
        with self.assertRaises(RuntimeError):self.call(s)
        self.assertEqual([c.kwargs['json']['model'] for c in s.post.call_args_list],[logic.DEFAULT_ANALYSIS_MODEL]*2)
        unload.assert_called_once_with(s,logic.DEFAULT_ANALYSIS_MODEL,self.cfg['unload_timeout_seconds'])
    @patch.object(oc,'api')
    def test_cached_draft_can_clear_instruct_residency(self,api):
        api.side_effect=[{'models':[{'name':logic.DEFAULT_ANALYSIS_MODEL}]},{},{'models':[]}]
        oc.clear_owned_residency(MagicMock(),self.info)
        self.assertEqual(api.call_args_list[1].args[2]['model'],logic.DEFAULT_ANALYSIS_MODEL)
    @patch.object(oc,'api')
    def test_cached_draft_refuses_unknown_resident_model(self,api):
        api.return_value={'models':[{'name':'unrelated:latest'}]}
        with self.assertRaisesRegex(RuntimeError,'Another'):
            oc.clear_owned_residency(MagicMock(),self.info)
        self.assertEqual(api.call_count,1)
    def test_settings_reject_same_model_in_both_roles(self):
        with tempfile.TemporaryDirectory() as td:
            cfg=copy.deepcopy(self.cfg);cfg['analysis_model']=cfg['ollama_model']
            (Path(td)/'settings.json').write_text(json.dumps(cfg))
            with patch.object(logic,'ROOT',Path(td)),patch.dict(logic.os.environ,{},clear=True):
                with self.assertRaisesRegex(ValueError,'separate'):logic.settings()
    def test_service_downloads_both_models_sequentially_and_keeps_one_resident(self):
        spec=importlib.util.spec_from_file_location('test_portrait_service',ROOT/'ollama_service.py')
        service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
        proc=MagicMock();proc.poll.return_value=None;proc.wait.side_effect=[KeyboardInterrupt(),0];proc.pid=1234
        sess=MagicMock();sess.__enter__.return_value=sess
        with tempfile.TemporaryDirectory() as td,patch.object(service,'ROOT',Path(td)),patch.object(service,'settings',return_value=self.cfg),patch.object(service.oc,'session',return_value=sess),patch.object(service.oc,'api',side_effect=[RuntimeError('not up'),{}]),patch.object(service.oc,'pipeline_info',return_value=self.info),patch.object(service.subprocess,'Popen',return_value=proc) as popen,patch.object(service.subprocess,'run') as pull,patch.object(service.signal,'signal'),patch.object(service.os,'killpg'):
            service.main()
            models=[c.args[0][-1] for c in pull.call_args_list]
            self.assertEqual(models,[logic.DEFAULT_ANALYSIS_MODEL,logic.DEFAULT_MODEL])
            self.assertEqual(popen.call_args.kwargs['env']['OLLAMA_MAX_LOADED_MODELS'],'1')
            status=json.loads((Path(td)/'ollama-status.json').read_text())
            self.assertEqual(status['analysis']['model'],logic.DEFAULT_ANALYSIS_MODEL)
            self.assertEqual(status['model'],logic.DEFAULT_MODEL)


if __name__=='__main__':unittest.main()
