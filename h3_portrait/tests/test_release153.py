"""Regression tests for missing models, prompt handoff and memory configuration."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod);return mod

rt=load(ROOT/'node/local_runtime.py','release153_runtime')
svc=load(ROOT/'ollama_service.py','release153_service')
opts=load(ROOT/'launch_options.py','release153_options')
installer=load(ROOT/'install_workflows.py','release153_installer')
MODEL='huihui_ai/qwen3-vl-abliterated:32b-instruct-q4_K_M'


def response(messages):
    r=MagicMock();r.status_code=200;r.__enter__.return_value=r
    r.iter_lines.return_value=iter(json.dumps(x).encode() for x in messages)
    return r


class StartupTests(unittest.TestCase):
    def test_cached_model_is_not_pulled(self):
        s=MagicMock()
        with patch.object(svc,'present',return_value=True):svc.pull_model(s,MODEL,emit=MagicMock())
        s.post.assert_not_called()

    def test_pull_requires_success_and_registered_tag(self):
        s=MagicMock();s.post.return_value=response([{'status':'pulling','total':100,'completed':91},{'status':'success'}])
        with patch.object(svc,'present',side_effect=[False,True]):svc.pull_model(s,MODEL,emit=MagicMock())
        self.assertEqual(s.post.call_args.kwargs['json'],{'model':MODEL,'stream':True})
        self.assertFalse(s.post.call_args.kwargs['allow_redirects'])

    def test_incomplete_pull_does_not_become_ready(self):
        s=MagicMock();s.post.return_value=response([{'status':'pulling','completed':91,'total':100}])
        with patch.object(svc,'present',return_value=False),self.assertRaisesRegex(RuntimeError,'Could not finish'):
            svc.pull_model(s,MODEL,attempts=1,emit=MagicMock())

    def test_retry_reuses_same_model_without_deleting_parts(self):
        s=MagicMock();s.post.side_effect=[response([{'error':'temporary download failure'}]),response([{'status':'success'}])]
        with patch.object(svc,'present',side_effect=[False,True]):
            svc.pull_model(s,MODEL,emit=MagicMock(),sleep=lambda t:None)
        self.assertEqual([c.kwargs['json']['model'] for c in s.post.call_args_list],[MODEL,MODEL])
        s.delete.assert_not_called()

    def test_success_without_tag_is_failure(self):
        s=MagicMock();s.post.return_value=response([{'status':'success'}])
        with patch.object(svc,'present',return_value=False),self.assertRaises(RuntimeError):
            svc.pull_model(s,MODEL,attempts=1,emit=MagicMock())

    def test_pull_stalls_are_bounded(self):
        s=MagicMock();msg={'status':'pulling','completed':91,'total':100}
        s.post.return_value=response([msg,msg])
        # deadline, attempt, progress start, message1, message2
        values=iter([0,0,0,0,1,200])
        with patch.object(svc,'present',return_value=False),self.assertRaisesRegex(RuntimeError,'stalled'):
            svc.pull_model(s,MODEL,attempts=1,stall_seconds=180,clock=lambda:next(values),emit=MagicMock())

    def test_models_dir_is_configurable_and_secrets_not_forwarded(self):
        env=svc.daemon_env({'context_length':32768},{'OLLAMA_MODELS':'/workspace/cache/ollama','HF_TOKEN':'secret','HTTP_PROXY':'secret'})
        self.assertEqual(env['OLLAMA_MODELS'],'/workspace/cache/ollama')
        self.assertEqual(env['OLLAMA_CONTEXT_LENGTH'],'32768')
        self.assertEqual(env['OLLAMA_KEEP_ALIVE'],'0')
        self.assertNotIn('HF_TOKEN',env);self.assertNotIn('HTTP_PROXY',env)

    def test_bad_cache_path_fails(self):
        with self.assertRaises(ValueError):svc.daemon_env({'context_length':32768},{'OLLAMA_MODELS':'/'})

    def test_status_is_atomic_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'status.json';svc.atomic_json(p,{'phase':'ready'})
            self.assertEqual(json.loads(p.read_text()),{'phase':'ready'})
            self.assertFalse(p.with_name('status.json.tmp').exists())

    def test_startup_wait_is_before_comfy_process(self):
        text=(ROOT/'start.sh').read_text()
        self.assertLess(text.index('wait_ready.py'),text.index('python main.py'))
        self.assertIn('"${MEMORY_FLAGS[@]}"',text)

    def test_default_memory_path_avoids_dynamic_allocator(self):
        self.assertEqual(opts.options({}),['--disable-dynamic-vram','--reserve-vram','4.0'])

    def test_dynamic_memory_is_explicit_opt_in(self):
        self.assertEqual(opts.options({'H3_DYNAMIC_VRAM':'1','H3_RESERVE_VRAM':'5'})[0],'--enable-dynamic-vram')

    def test_invalid_memory_settings_are_not_ignored(self):
        for env in ({'H3_DYNAMIC_VRAM':'auto'},{'H3_RESERVE_VRAM':'nan'},{'H3_RESERVE_VRAM':'-1'}):
            with self.assertRaises(ValueError):opts.options(env)


class HandoffTests(unittest.TestCase):
    def original(self, prompt_provider='local', api_base='http://127.0.0.1:11434/v1', local_model_slug=MODEL, openrouter_api_key=''):
        return 'prompt'

    def test_private_base_has_no_remote_or_proxy_escape(self):
        self.assertEqual(rt.local_base('http://localhost:11434/v1'),rt.BASE+'/v1')
        for base in ('https://example.com/v1','http://127.0.0.1:11434/v1?key=secret','http://user:pass@127.0.0.1:11434/v1','http://127.0.0.1:11434/not-v1'):
            with self.assertRaises(ValueError):rt.local_base(base)

    def test_session_ignores_environment_proxies(self):
        with rt.session() as s:self.assertFalse(s.trust_env)

    def test_missing_model_fails_before_comfy_release_or_inference(self):
        with patch.object(rt,'session'),patch.object(rt,'ensure_installed',side_effect=RuntimeError('not ready')),patch.object(rt,'release_comfy') as release:
            with self.assertRaisesRegex(RuntimeError,'not ready'):rt.guard_refpack(self.original)()
            release.assert_not_called()

    def test_success_unloads_after_original_returns(self):
        events=[]
        def original(prompt_provider='local',api_base=rt.BASE+'/v1',local_model_slug=MODEL):events.append('write');return 'prompt'
        with patch.object(rt,'session'),patch.object(rt,'ensure_installed'),patch.object(rt,'api',return_value={'models':[]}),patch.object(rt,'release_comfy',side_effect=lambda:events.append('free_comfy')),patch.object(rt,'unload',side_effect=lambda *a:events.append('unload_ollama')):
            self.assertEqual(rt.guard_refpack(original)(),'prompt')
        self.assertEqual(events,['free_comfy','write','unload_ollama'])

    def test_exception_still_unloads(self):
        def original(prompt_provider='local',api_base=rt.BASE+'/v1',local_model_slug=MODEL):raise ValueError('writer failed')
        with patch.object(rt,'session'),patch.object(rt,'ensure_installed'),patch.object(rt,'api',return_value={'models':[]}),patch.object(rt,'release_comfy'),patch.object(rt,'unload') as unload:
            with self.assertRaisesRegex(ValueError,'writer failed'):rt.guard_refpack(original)()
            unload.assert_called_once()

    def test_unload_failure_blocks_h3_even_after_good_prompt(self):
        with patch.object(rt,'session'),patch.object(rt,'ensure_installed'),patch.object(rt,'api',return_value={'models':[]}),patch.object(rt,'release_comfy'),patch.object(rt,'unload',side_effect=RuntimeError('still resident')):
            with self.assertRaisesRegex(RuntimeError,'still resident'):rt.guard_refpack(self.original)()

    def test_manual_prompt_passthrough_never_calls_ollama(self):
        with patch.object(rt,'session') as session:
            self.assertEqual(rt.guard_refpack(self.original)(prompt_provider='none'),'prompt')
            session.assert_not_called()

    def test_hosted_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'hosted'):rt.guard_refpack(self.original)(prompt_provider='openrouter')

    def test_explicit_unload_and_confirmation(self):
        with patch.object(rt,'api',side_effect=[{}, {'models':[{'name':MODEL}]}, {'models':[]}]) as api:
            rt.unload(MagicMock(),MODEL,sleep=lambda t:None)
            self.assertEqual(api.call_args_list[0].args[2]['keep_alive'],0)
            self.assertEqual(api.call_args_list[-1].args[1],'/api/ps')

    def test_unrelated_resident_model_is_not_silently_stopped(self):
        with patch.object(rt,'api',return_value={'models':[{'name':'unrelated:latest'}]}),self.assertRaisesRegex(RuntimeError,'still using GPU'):
            rt.require_idle(MagicMock())

    def test_cloud_model_is_not_accepted_as_local(self):
        with patch.object(rt,'api',side_effect=[{'models':[{'name':MODEL}]},{'capabilities':['vision'],'remote_host':'somewhere'}]),self.assertRaises(ValueError):
            rt.ensure_installed(MagicMock(),MODEL)


class WorkflowTests(unittest.TestCase):
    def test_full_models_and_high_quality_preserved(self):
        cfg=json.loads((ROOT/'settings.json').read_text())
        graph=json.loads((ROOT/'workflows/H3_Reference_Video_Swap_Local.json').read_text())
        got=installer.configured_graph(graph,cfg)
        types={n['type']:n for n in got['nodes']}
        self.assertEqual(types['SafeH3CLIPLoader']['widgets_values'],['qwen3vl_32b_minimax_h3_bf16.safetensors','minimax','default'])
        self.assertEqual(types['SafeH3AudioVAELoader']['widgets_values'],['minimax_h3_audio_vae_fp32.safetensors'])
        self.assertEqual(types['BasicScheduler']['widgets_values'][1],25)
        self.assertEqual(types['MiniMaxH3ReferencePack']['widgets_values'][8],MODEL)
        self.assertEqual(types['MiniMaxH3ReferenceToVideo']['widgets_values'][4],'max')
        self.assertEqual(graph['links'],got['links'])

    def test_gate_runs_again_for_cached_reference_pack(self):
        mod=load(ROOT/'node/reference_video.py','release153_video')
        import math
        self.assertTrue(math.isnan(mod.H3ReferenceVideoDraftGate.IS_CHANGED()))
        self.assertIn('_require_ollama_idle()', (ROOT/'node/reference_video.py').read_text())


if __name__=='__main__':unittest.main()


class DraftGateTests(unittest.TestCase):
    def gate(self):
        from node.reference_video import H3ReferenceVideoDraftGate
        return H3ReferenceVideoDraftGate()

    def graph_utils(self):
        class Blocker:
            def __init__(self,x): self.value=x
        return types.SimpleNamespace(ExecutionBlocker=Blocker)

    def test_draft_only_does_not_check_or_load_h3(self):
        with patch.dict(sys.modules,{'comfy_execution.graph_utils':self.graph_utils()}),patch('node.reference_video._require_ollama_idle') as idle:
            got=self.gate().review('prompt','debug',object(),object(),'Draft only')
            self.assertNotIsInstance(got['result'][0],str);idle.assert_not_called()

    def test_generate_checks_live_residency(self):
        with patch.dict(sys.modules,{'comfy_execution.graph_utils':self.graph_utils()}),patch('node.reference_video._require_ollama_idle') as idle:
            got=self.gate().review('prompt','debug',object(),object(),'Generate video')
            self.assertEqual(got['result'],('prompt',));idle.assert_called_once()

    def test_generate_refuses_cached_prompt_with_live_ollama(self):
        with patch.dict(sys.modules,{'comfy_execution.graph_utils':self.graph_utils()}),patch('node.reference_video._require_ollama_idle',side_effect=RuntimeError('still resident')):
            with self.assertRaisesRegex(RuntimeError,'still resident'):
                self.gate().review('cached prompt','debug',object(),object(),'Generate video')

    def test_generate_requires_visual_inputs(self):
        with patch.dict(sys.modules,{'comfy_execution.graph_utils':self.graph_utils()}),self.assertRaisesRegex(ValueError,'subject image'):
            self.gate().review('prompt','debug',None,object(),'Generate video')
