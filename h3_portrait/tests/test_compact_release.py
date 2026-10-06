"""Compact prompt-stage and release identity regression tests. No live inference."""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from node import logic, ollama_client as oc
from test_thinking import analysis, director
spec=importlib.util.spec_from_file_location('portrait_release',ROOT/'verify_release.py')
release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'settings.json').read_text())
        self.info={'model':logic.DEFAULT_MODEL,'capabilities':['vision','thinking'],
                   'thinking':{'values':[True]},
                   'analysis':{'model':logic.DEFAULT_ANALYSIS_MODEL,'digest':'a-digest','capabilities':['vision']}}
        self.refs=[{'filename':'head.png','image':object()},{'filename':'body.png','image':object()},{'filename':'outfit.png','image':object()}]
        self.recipe=logic.geometry('9:16','Standard',5)
        patcher=patch.object(oc,'preview',return_value='image')
        patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(oc,'unload');self.unload=patcher.start();self.addCleanup(patcher.stop)
    def response(self,obj=None,raw=None,reason='stop',thinking=''):
        r=MagicMock(status_code=200);r.__enter__.return_value=r
        content=json.dumps(obj) if raw is None else raw
        r.iter_lines.return_value=[json.dumps({'message':{'thinking':thinking,'content':content},'done':True,'done_reason':reason}).encode()]
        return r
    def call(self,s):
        return oc.generate(s,self.cfg,self.info,'system','All three photos show one person. Walk.',self.refs,self.recipe,'',0)
    def test_incomplete_json_regenerates_without_feeding_fragment_back(self):
        s=MagicMock();s.post.side_effect=[self.response(raw='{"references": [UNFINISHED'),self.response(analysis()),self.response(director())]
        result=self.call(s)
        self.assertEqual(len(result['references']),3)
        retry=s.post.call_args_list[1].kwargs['json']
        self.assertIs(retry.get('think', False),False)
        self.assertNotIn('UNFINISHED',json.dumps(retry))
        self.assertFalse(any(m['role']=='assistant' for m in retry['messages']))
        self.assertEqual(self.unload.call_count,2)
    def test_analysis_token_cutoff_short_retry_then_success(self):
        s=MagicMock();s.post.side_effect=[self.response(raw='{',reason='length'),self.response(analysis()),self.response(director())]
        result=self.call(s)
        self.assertEqual(result['_stages'][0]['attempts'],2)
        self.assertEqual([c.kwargs['json'].get('think', False) for c in s.post.call_args_list],[False,False,True])
    def test_director_cutoff_repair_does_not_repeat_thinking(self):
        s=MagicMock();s.post.side_effect=[self.response(analysis()),self.response(raw='{',reason='length'),self.response(director())]
        result=self.call(s)
        self.assertEqual([c.kwargs['json'].get('think', False) for c in s.post.call_args_list],[False,True,False])
        self.assertFalse(result['_stages'][1]['think_requested'])
        self.assertEqual(s.post.call_args_list[2].kwargs['json']['options']['num_predict'],4096)
    def test_wrong_model_that_cannot_disable_thinking_fails_before_chat(self):
        self.info['analysis']['model']=logic.DEFAULT_MODEL;s=MagicMock()
        with self.assertRaisesRegex(RuntimeError,'Instruct'):self.call(s)
        s.post.assert_not_called()
    def test_model_ignoring_no_thinking_stops_early(self):
        s=MagicMock();s.post.return_value=self.response(analysis(),thinking='x'*2049)
        with self.assertRaisesRegex(RuntimeError,'despite think=false'):self.call(s)
        self.assertEqual(s.post.call_count,1);self.unload.assert_called_once()
    def test_pass_two_never_receives_base64_images(self):
        s=MagicMock();s.post.side_effect=[self.response(analysis()),self.response(director())];self.call(s)
        for m in s.post.call_args_list[1].kwargs['json']['messages']:
            self.assertNotIn('images',m)
    def test_every_original_image_is_still_present_in_source_refs(self):
        original=list(self.refs)
        s=MagicMock();s.post.side_effect=[self.response(analysis()),self.response(director())];self.call(s)
        for a,b in zip(original,self.refs):self.assertIs(a['image'],b['image'])
    def test_nine_refs_all_preserved_and_compact(self):
        data=analysis(9)
        self.assertEqual(len(logic.parse_analysis(json.dumps(data),9)['references']),9)
        schema=logic.analysis_schema(9)
        self.assertEqual(schema['properties']['references']['minItems'],9)
        self.assertEqual(schema['properties']['references']['maxItems'],9)
    def test_overlong_field_is_rejected_not_truncated(self):
        for key,limit in [('subject',64),('contains',160),('use_for',240)]:
            with self.subTest(key=key):
                data=analysis();data['references'][0][key]='x'*(limit+1)
                with self.assertRaises(ValueError):logic.parse_analysis(json.dumps(data),3)
    def test_boundary_lengths_are_valid(self):
        data=analysis()
        for row in data['references']:
            row['subject']='x'*64;row['contains']='x'*160;row['use_for']='x'*240
        logic.parse_analysis(json.dumps(data),3)
    def test_priorities_and_conflicts_have_real_parser_limits(self):
        for key,value in [('priorities','x'*513),('clarification','x'*241),('conflicts',['x'*161]),('conflicts',['x']*6)]:
            data=analysis();data[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):logic.parse_analysis(json.dumps(data),3)
    def test_analysis_deadline_checked_before_request(self):
        with patch.object(oc.time,'monotonic',return_value=2):
            s=MagicMock()
            with self.assertRaisesRegex(RuntimeError,'time budget'):
                oc._stream_final(s,self.cfg,self.info,'TEST',{'think':False},1,lambda:None)
            s.post.assert_not_called()
    def test_diagnostics_do_not_contain_thinking_text(self):
        s=MagicMock();s.post.side_effect=[self.response(analysis()),self.response(director(),thinking='SECRET_THOUGHT')]
        result=self.call(s)
        self.assertNotIn('SECRET_THOUGHT',json.dumps(result))
    def test_reference_map_is_deep_copied_on_director_validation(self):
        data=analysis();out=logic.parse_director(json.dumps(director()),data,3)
        out['references'][0]['contains']='changed'
        self.assertNotEqual(data['references'][0]['contains'],'changed')
    def test_human_instruction_is_not_parsed(self):
        text='Photos one through five are the same person, no colon labels. Walk and smile!'
        obj=logic.parse_director(json.dumps(director()),analysis(),3)
        self.assertIn(text,logic.format_prompt(obj,text,self.recipe))
    def test_analysis_prompt_describes_bounds(self):
        policy=logic.prompt_policy()['analysis_prompt.txt']
        for phrase in ('64 characters','160 characters','240 characters','ALL image entries'):
            self.assertIn(phrase,policy)
    def test_cache_tracks_analysis_settings(self):
        before=logic.cache_key(['x'],'walk',self.recipe,'digest',self.cfg,'',0)
        other=copy.deepcopy(self.cfg);other['analysis_temperature']=.12
        self.assertNotEqual(before,logic.cache_key(['x'],'walk',self.recipe,'digest',other,'',0))
    def test_actual_geometry_presets_unchanged(self):
        self.assertEqual(logic.geometry('9:16','Standard',5)['output_width'],756)
        self.assertEqual(logic.geometry('2:3','Standard',5)['output_width'],768)
        self.assertEqual(logic.geometry('9:16','Standard',5)['steps'],20)


class ReleaseTests(unittest.TestCase):
    def test_current_source_verified(self):
        self.assertEqual(release.verify_source(ROOT)['version'],'1.4.0')
    def test_all_node_files_hashed(self):
        self.assertEqual(set(release.node_hashes(ROOT/'node')),set(release.NODE_FILES))
    def test_missing_analysis_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(ROOT,Path(td)/'h3')
            (Path(td)/'h3/node/analysis_prompt.txt').unlink()
            with self.assertRaisesRegex(ValueError,'missing'):release.verify_source(Path(td)/'h3')
    def test_old_client_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(ROOT,Path(td)/'h3')
            (Path(td)/'h3/node/ollama_client.py').write_text('PIPELINE_REVISION="old"\n')
            with self.assertRaisesRegex(ValueError,'Older'):release.verify_source(Path(td)/'h3')
    def test_old_node_init_cannot_pass_with_new_client(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(ROOT,Path(td)/'h3')
            (Path(td)/'h3/node/__init__.py').write_text('PACKAGE_VERSION="1.1.0"\n')
            with self.assertRaisesRegex(ValueError,'Older node'):release.verify_source(Path(td)/'h3')
    def test_copy_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(ROOT/'node',Path(td)/'node')
            (Path(td)/'node/system_prompt.txt').write_text('old system prompt')
            with self.assertRaisesRegex(ValueError,'does not match'):release.compare_node(ROOT,Path(td)/'node')
    def test_matching_node_copy_valid(self):
        release.compare_node(ROOT,ROOT/'node')
    def test_source_settings_old_version_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            shutil.copytree(ROOT,Path(td)/'h3')
            p=Path(td)/'h3/settings.json';cfg=json.loads(p.read_text());cfg['version']='1.1.0';p.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):release.verify_source(Path(td)/'h3')
    def test_real_manifest_roundtrip_and_changed_node_detection(self):
        with tempfile.TemporaryDirectory() as td:
            manifest=Path(td)/'release.json'
            cmd=[sys.executable,str(ROOT/'verify_release.py'),'--source',str(ROOT),'--source-revision','a'*40]
            subprocess.run(cmd+['--write-manifest',str(manifest)],check=True,stdout=subprocess.PIPE)
            subprocess.run(cmd+['--check-manifest',str(manifest)],check=True,stdout=subprocess.PIPE)
            saved=json.loads(manifest.read_text());saved['node_sha256']['logic.py']='0'*64;manifest.write_text(json.dumps(saved))
            result=subprocess.run(cmd+['--check-manifest',str(manifest)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertNotEqual(result.returncode,0)
    def test_runtime_checks_happen_before_any_model_download(self):
        text=(ROOT/'start.sh').read_text()
        self.assertLess(text.index('--check-manifest'),text.index('prepare_assets.py'))
        self.assertIn('--node-copy "$COMFY_HOME/custom_nodes/ComfyUI-H3Portrait"',text)
    def test_docker_source_and_bundle_check(self):
        text=(ROOT/'Dockerfile').read_text()
        self.assertIn('COPY h3_portrait/ /opt/h3-portrait/',text)
        self.assertIn('--write-manifest /opt/h3-portrait/release-manifest.json',text)
        self.assertIn('ENV H3_SOURCE_REVISION=${SOURCE_REVISION}',text)
    def test_no_lora_link_list_in_release_fingerprints(self):
        self.assertFalse(any('lora_links' in name for name in release.NODE_FILES))


if __name__=='__main__':unittest.main()
