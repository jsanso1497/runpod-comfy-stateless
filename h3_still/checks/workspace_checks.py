from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(PACKAGE/'runtime'))
from fixtures import graph
from workflow_tools import portable_graph, get_node, set_widget, validate_graph, blob_sha, PRESETS
from validate_package import validate_manifest
from validate_live import validate as validate_live
from bootstrap import safe_child, verified, seed_workflows, atomic_json
from launch import validate_password, clean_environment, NGINX

MODELS = json.loads((PACKAGE/'models.lock.json').read_text())['models']


def converted(profile='turbo'):
    return portable_graph(graph(profile.startswith('edit'), profile=='hq'), profile+'.json',profile,MODELS)


class WorkflowTests(unittest.TestCase):
    def test_four_presets(self):
        for _, name, profile in PRESETS:
            validate_graph(converted(profile),profile)
    def test_source_graph_not_mutated(self):
        source=graph();before=copy.deepcopy(source)
        portable_graph(source,'a.json','turbo',MODELS)
        self.assertEqual(source,before)
    def test_filename_and_metadata_are_consistent(self):
        node=get_node(converted(),'LoraLoaderModelOnly')
        self.assertNotIn('\\',node['widgets_values'][0])
        self.assertEqual(node['widgets_values'][0],node['widgets_values_named']['lora_name'])
        self.assertIn('larryvrh',node['properties']['models'][0]['url'])
        self.assertNotIn('/main/',node['properties']['models'][0]['url'])
    def test_vae_metadata_not_fp16(self):
        node=get_node(converted(),'VAELoader')
        self.assertIn('int8_convrot',node['properties']['models'][0]['name'])
    def test_hq_50_steps_no_turbo(self):
        g=converted('hq')
        self.assertEqual(get_node(g,'BasicScheduler')['widgets_values_named']['steps'],50)
        self.assertEqual(get_node(g,'LoraLoaderModelOnly')['widgets_values_named']['strength_model'],0)
        self.assertEqual(get_node(g,'ResolutionSelector')['widgets_values_named']['multiple'],32)
    def test_both_edit_resolutions_preserved(self):
        self.assertEqual(get_node(converted('edit_readme'),'ImageScaleToTotalPixels')['widgets_values_named']['megapixels'],2.5)
        self.assertEqual(get_node(converted('edit_original'),'ImageScaleToTotalPixels')['widgets_values_named']['megapixels'],2)
    def test_topology_preserved(self):
        original=graph(); result=portable_graph(original,'t.json','turbo',MODELS)
        self.assertEqual(result['links'],original['links'])
        self.assertEqual([n['type'] for n in result['nodes']],[n['type'] for n in original['nodes']])
    def test_unique_stable_ids(self):
        a=converted('turbo');b=converted('hq')
        self.assertNotEqual(a['id'],b['id'])
        self.assertEqual(a['id'],converted('turbo')['id'])
    def test_unknown_model_refused(self):
        source=graph();set_widget(get_node(source,'UNETLoader'),'unet_name','wrong.safetensors')
        with self.assertRaises(ValueError):portable_graph(source,'a.json','turbo',MODELS)
    def test_widget_array_mismatch_refused(self):
        n=get_node(graph(),'LoraLoaderModelOnly');n['widgets_values'][0]='wrong'
        with self.assertRaises(ValueError):set_widget(n,'lora_name','valid')
    def test_wrong_sampler_rejected(self):
        g=converted();set_widget(get_node(g,'KSamplerSelect'),'sampler_name','euler')
        with self.assertRaises(ValueError):validate_graph(g,'turbo')
    def test_disconnected_latent_rejected(self):
        g=converted();get_node(g,'SamplerCustomAdvanced')['inputs'][-1]['link']=None
        with self.assertRaises(ValueError):validate_graph(g,'turbo')
    def test_standard_decode_cannot_be_saved(self):
        g=converted();n=get_node(g,'FizgigH3StillDecode');n['type']='VAEDecode'
        with self.assertRaises(ValueError):validate_graph(g,'turbo')
    def test_duplicate_link_rejected(self):
        g=converted();g['links'].append(g['links'][0])
        with self.assertRaises(ValueError):validate_graph(g,'turbo')
    def test_prompt_and_seed_unchanged(self):
        source=graph();g=portable_graph(source,'t.json','turbo',MODELS)
        for kind in ('MiniMaxH3ImageToVideo','RandomNoise'):
            self.assertEqual(get_node(source,kind)['widgets_values'],get_node(g,kind)['widgets_values'])


class ManifestTests(unittest.TestCase):
    def test_exact_developer_assets(self):validate_manifest({'models':MODELS})
    def test_unpinned_revision_rejected(self):
        rows=copy.deepcopy(MODELS);rows[0]['revision']='main'
        with self.assertRaises(ValueError):validate_manifest({'models':rows})
    def test_extra_enhancer_rejected(self):
        rows=copy.deepcopy(MODELS);r=copy.deepcopy(rows[0]);r.update(id='enhancer',destination='text_encoders/extra.safetensors');rows.append(r)
        with self.assertRaises(ValueError):validate_manifest({'models':rows})
    def test_duplicate_destination_rejected(self):
        rows=copy.deepcopy(MODELS);rows[1]['destination']=rows[0]['destination']
        with self.assertRaises(ValueError):validate_manifest({'models':rows})
    def test_blob_hash_matches_git_empty_blob(self):
        self.assertEqual(blob_sha(b''),'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')
    def test_sources_are_pinned(self):
        lock=json.loads((PACKAGE/'sources.lock.json').read_text())
        for source in lock['sources'].values():self.assertRegex(source['commit'],r'^[0-9a-f]{40}$')


class StorageTests(unittest.TestCase):
    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            for path in ('../outside','/absolute','x\\bad'):
                with self.assertRaises(ValueError):safe_child(Path(t),path)
    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'root';root.mkdir();(root/'escape').symlink_to(Path(t))
            with self.assertRaises(ValueError):safe_child(root,'escape/file')
    def test_legitimate_internal_symlink_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'source').mkdir();(root/'alias').symlink_to(root/'source')
            self.assertEqual(safe_child(root,'alias/test'),root/'alias/test')
    def test_checksum_change_detected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'asset';p.write_bytes(b'good');marker=Path(t)/'verified.json'
            row={'id':'tiny','revision':'a'*40,'sha256':hashlib.sha256(b'good').hexdigest()}
            self.assertTrue(verified(p,row,marker))
            self.assertTrue(verified(p,row,marker))
            p.write_bytes(b'wrong')
            self.assertFalse(verified(p,row,marker))
    def test_wrong_hash_not_marked(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'asset';p.write_bytes(b'bad');m=Path(t)/'marker'
            self.assertFalse(verified(p,{'id':'x','revision':'a'*40,'sha256':'0'*64},m))
            self.assertFalse(m.exists())
    def test_atomic_metadata_symlink_refused(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'link';p.symlink_to(Path(t)/'elsewhere')
            with self.assertRaises(ValueError):atomic_json(p,{})
    def test_user_workflow_edits_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);package=root/'package';(package/'workflows').mkdir(parents=True)
            (package/'workflows/example.json').write_text('original')
            data=root/'data';data.mkdir()
            with patch('bootstrap.PACKAGE',package):
                seed_workflows(data)
                target=next((data/'user').rglob('example.json'));target.write_text('my edit')
                seed_workflows(data)
            self.assertEqual(target.read_text(),'my edit')


class AuthTests(unittest.TestCase):
    def test_valid_password(self):validate_password('this-is-a-long-test-password')
    def test_unsafe_passwords_rejected(self):
        for value in ('short','x'*73,'x'*20+'\n','{{ RUNPOD_SECRET_comfy_password }}','x'*20+'\x00'):
            with self.assertRaises(ValueError):validate_password(value)
    def test_secrets_removed_from_child_environment(self):
        with patch.dict(os.environ,{'COMFY_PASSWORD':'secret','HF_TOKEN':'token','WB_PASSWORD':'old','WB_LORA_URLS':'private'}):
            env=clean_environment()
            for name in ('COMFY_PASSWORD','HF_TOKEN','WB_PASSWORD','WB_LORA_URLS'):self.assertNotIn(name,env)
    def test_auth_all_paths_no_query_bypass(self):
        self.assertLess(NGINX.index('auth_basic '),NGINX.index('location /'))
        self.assertNotIn('utm_source',NGINX)
        self.assertIn('Cache-Control "no-store"',NGINX)
        self.assertIn('proxy_set_header Upgrade $http_upgrade',NGINX)


class LiveSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.info={}
        for _,name,profile in PRESETS:
            g=converted(profile);(self.root/name).write_text(json.dumps(g))
            for node in g['nodes']:
                fields={}
                for key,val in node['widgets_values_named'].items():
                    if isinstance(val,str):fields[key]=['STRING',{}] if key in ('prompt','filename_prefix','image','upload','control_after_generate') else [[val],{}]
                    else:fields[key]=['FLOAT',{}]
                if node['type']=='LoadImage':fields['image']=[[],{}]
                self.info[node['type']]={'input':{'required':fields}}
    def tearDown(self):self.temp.cleanup()
    def test_real_schema_checker_accepts_compatible_fixture(self):
        self.assertEqual(validate_live(self.info,self.root),[])
    def test_missing_resize_node_detected(self):
        del self.info['ImageScaleToTotalPixels']
        self.assertTrue(any('ImageScaleToTotalPixels' in x for x in validate_live(self.info,self.root)))
    def test_missing_model_detected(self):
        self.info['UNETLoader']['input']['required']['unet_name']=[[],{}]
        self.assertTrue(any('unet_name' in x for x in validate_live(self.info,self.root)))
    def test_unsupported_sampler_detected(self):
        self.info['KSamplerSelect']['input']['required']['sampler_name']=[['euler'],{}]
        self.assertTrue(any('sampler_name' in x for x in validate_live(self.info,self.root)))


if __name__=='__main__':unittest.main()
