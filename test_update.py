from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import catalog
import check_workflows
import install_nodes
spec = importlib.util.spec_from_file_location('everyday_nodes', ROOT / 'scripts/local_nodes/ComfyUI-Everyday/__init__.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ROOT / 'config'
    def test_default_profiles(self):
        with mock.patch.dict(os.environ, {'MODEL_PROFILES': ''}):
            self.assertEqual(catalog.selected_profiles(self.cfg), {'seedvr2', 'krea2'})
    def test_h3_is_explicit(self):
        self.assertEqual(catalog.selected_profiles(self.cfg, 'h3'), {'h3'})
    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError): catalog.selected_profiles(self.cfg, 'latest')
    def test_core_and_node_commits_pinned(self):
        self.assertRegex(catalog.read_json(self.cfg/'runtime.json')['comfy_ref'], '^[a-f0-9]{40}$')
        for node in catalog.read_json(self.cfg/'custom_nodes.json'): install_nodes.validate(node)
    def test_rolling_node_ref_rejected(self):
        row=catalog.read_json(self.cfg/'custom_nodes.json')[0];row['ref']='main'
        with self.assertRaises(ValueError):install_nodes.validate(row)
    def test_default_download_count(self):
        self.assertEqual(len(catalog.catalog(self.cfg, Path('/tmp/comfy'), {'seedvr2','krea2'})),6)
    def test_krea_download_includes_identity_lora(self):
        rows=catalog.catalog(self.cfg, Path('/tmp/comfy'), {'krea2'})
        self.assertEqual(len(rows),4)
        self.assertTrue(any('identity_edit_v1_2' in x['destination'] for x in rows))
    def test_h3_not_downloaded_by_default(self):
        rows=catalog.catalog(self.cfg,Path('/tmp/comfy'), {'seedvr2','krea2'})
        self.assertFalse(any('minimax' in x['destination'] for x in rows))
    def test_every_selected_hash_valid(self):
        rows=catalog.catalog(self.cfg,Path('/tmp/comfy'), {'seedvr2','krea2','h3'})
        for row in rows: self.assertRegex(row['sha256'],'^[a-f0-9]{64}$')
    def test_disabled_style_not_selected(self):
        rows=catalog.catalog(self.cfg,Path('/tmp/comfy'), {'krea2'})
        self.assertFalse(any('darkbrush' in x['destination'] for x in rows))
    def test_workflows_omit_h3_default(self):
        with tempfile.TemporaryDirectory() as t:
            paths=catalog.install_workflows(self.cfg,Path(t),{'seedvr2','krea2'})
            self.assertEqual(len([p for p in paths if 'Krea2' in p.name]),2)
            self.assertFalse(any('H3' in p.name for p in paths))
    def test_workflow_user_edit_preserved_on_process_restart(self):
        with tempfile.TemporaryDirectory() as t:
            paths=catalog.install_workflows(self.cfg,Path(t),{'krea2'})
            paths[0].write_text('saved user edit')
            catalog.install_workflows(self.cfg,Path(t),{'krea2'})
            self.assertEqual(paths[0].read_text(),'saved user edit')


class SafetyTests(unittest.TestCase):
    def test_valid_path(self):
        self.assertEqual(catalog.safe_destination(Path('/tmp/comfy'),'models/loras/example.safetensors'),Path('/tmp/comfy/models/loras/example.safetensors'))
    def test_path_rejections(self):
        for value in ['/etc/file.safetensors','models/../../etc/file.safetensors','models\\loras\\x.safetensors','scripts/evil.py','models/loras/evil.py']:
            with self.subTest(value=value),self.assertRaises(ValueError):catalog.safe_destination(Path('/tmp/comfy'),value)
    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as outside:
            root=Path(t);(root/'models').mkdir();(root/'models/loras').symlink_to(outside)
            with self.assertRaises(ValueError):catalog.safe_destination(root,'models/loras/x.safetensors')
    def test_root_models_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as outside:
            root=Path(t);(root/'models').symlink_to(outside)
            with self.assertRaises(ValueError):catalog.safe_destination(root,'models/loras/x.safetensors')
    def test_credential_url_rejected(self):
        for u in ['http://example.com/a.safetensors','https://user:password@example.com/a','https://example.com/a?token=secret']:
            with self.subTest(url=u),self.assertRaises(ValueError):catalog.asset_url({'source':'url','url':u})
    def test_hf_url_encoding(self):
        url=catalog.asset_url({'repo_id':'a/b','filename':'loras/a file.safetensors','revision':'tag/v1'})
        self.assertIn('tag%2Fv1',url);self.assertIn('a%20file.safetensors',url)
    def test_auth_not_sent_to_unrelated_host(self):
        with mock.patch.dict(os.environ,{'HF_TOKEN':'private','CIVITAI_TOKEN':'other'}):
            self.assertNotIn('Authorization',catalog.headers_for({},'https://example.com/a'))
            self.assertEqual(catalog.headers_for({},'https://huggingface.co/a')['Authorization'],'Bearer private')
    def test_valid_existing_asset_needs_no_network(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);dest=root/'models/loras/x.safetensors';dest.parent.mkdir(parents=True);dest.write_bytes(b'fixture')
            row={'name':'test','destination':'models/loras/x.safetensors','sha256':hashlib.sha256(b'fixture').hexdigest()}
            with mock.patch('requests.get',side_effect=AssertionError('Network not expected')):
                result=catalog.download_one(row,root)
            self.assertEqual(result['bytes'],7)


class GraphTests(unittest.TestCase):
    def test_all_graphs_links_and_identity_paths(self):
        for path in check_workflows.graph_paths(ROOT/'config'):
            with self.subTest(file=path.name):check_workflows.validate_graph(json.loads(path.read_text()))
    def test_reference_mismatch_detected(self):
        g=json.loads((ROOT/'config/workflows/10_Krea2_Same_Subject.json').read_text())
        n=next(n for n in g['nodes'] if n['id']==12)
        next(i for i in n['inputs'] if i['name']=='image')['link']=None
        with self.assertRaises(ValueError):check_workflows.validate_graph(g)
    def test_all_workflow_weights_exist_in_catalog(self):
        rows=catalog.catalog(ROOT/'config',Path('/tmp/comfy'),{'seedvr2','krea2','h3'})
        names={Path(row['destination']).name for row in rows}|{'None'}
        for path in catalog.workflow_sources(ROOT/'config'):
            for n in json.loads(path.read_text())['nodes']:
                for key,val in n.get('widgets_values_named',{}).items():
                    if key in {'unet_name','clip_name','vae_name','lora_name'}:self.assertIn(val,names)
    def test_optional_loras_really_off(self):
        for path in catalog.workflow_sources(ROOT/'config'):
            for n in json.loads(path.read_text())['nodes']:
                if n['type']=='EverydayOptionalLoRA':
                    self.assertFalse(n['widgets_values_named']['enabled']);self.assertEqual(n['widgets_values_named']['lora_name'],'None')
    def test_identity_lora_not_style_reference(self):
        g=json.loads((ROOT/'config/workflows/10_Krea2_Same_Subject.json').read_text())
        self.assertIn('krea2_identity_edit_v1_2.safetensors',json.dumps(g))
        self.assertNotIn('krea2_style_reference.safetensors',json.dumps(g))
    def test_both_branches_same_seed_defaults(self):
        for path in (ROOT/'config/workflows').glob('*Krea2*.json'):
            node=next(n for n in json.loads(path.read_text())['nodes'] if n['type']=='KSampler')
            self.assertEqual(node['widgets_values_named']['cfg'],3.0)
            self.assertEqual(node['widgets_values_named']['denoise'],1.0)
            self.assertEqual(node['widgets_values_named']['steps'],20)


class HelpersTests(unittest.TestCase):
    def test_square(self):self.assertEqual(helpers.canvas_size(1024,1024,'Square 1:1',1.0),(1024,1024))
    def test_portrait(self):
        w,h=helpers.canvas_size(3000,4000,'Portrait 4:5',1.0)
        self.assertLess(w,h);self.assertEqual(w%16,0);self.assertEqual(h%16,0)
    def test_budget_caps_all_aspects(self):
        for aspect in helpers.ASPECTS:
            for mp in (0.25,0.5,1.0,2.0):
                w,h=helpers.canvas_size(6000,4000,aspect,mp)
                self.assertLessEqual(w*h,mp*1024*1024);self.assertLessEqual(max(w,h),2048)
    def test_invalid_dimensions_rejected(self):
        for args in [(0,100,'Square 1:1',1),(100,100,'BAD',1),(100,100,'Square 1:1',4)]:
            with self.assertRaises(ValueError):helpers.canvas_size(*args)
    def test_optional_loader_true_passthrough(self):
        sentinel=object();node=helpers.EverydayOptionalLoRA()
        self.assertIs(node.apply(sentinel,False,'missing.safetensors',1)[0],sentinel)
        self.assertIs(node.apply(sentinel,True,'None',1)[0],sentinel)
        self.assertIs(node.apply(sentinel,True,'missing.safetensors',0)[0],sentinel)


class DownloadTests(unittest.TestCase):
    class Response:
        def __init__(self, body, status=200, offset=0):
            self.body=body; self.status_code=status
            self.headers={'Content-Length':str(len(body))}
            if status==206:self.headers['Content-Range']=f'bytes {offset}-{offset+len(body)-1}/{offset+len(body)}'
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def iter_content(self,n):yield self.body
    def row(self,payload):
        return {'name':'test','source':'url','url':'https://example.com/test.safetensors','destination':'models/loras/test.safetensors','sha256':hashlib.sha256(payload).hexdigest()}
    def test_atomic_download(self):
        payload=b'complete weights fixture'
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with mock.patch('requests.get',return_value=self.Response(payload)):
                result=catalog.download_one(self.row(payload),root)
            self.assertEqual((root/result['destination']).read_bytes(),payload)
            self.assertFalse((root/(result['destination']+'.part')).exists())
    def test_range_resume(self):
        payload=b'complete weights fixture'
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);part=root/'models/loras/test.safetensors.part';part.parent.mkdir(parents=True);part.write_bytes(payload[:8])
            with mock.patch('requests.get',return_value=self.Response(payload[8:],206,8)) as get:
                catalog.download_one(self.row(payload),root)
                self.assertEqual(get.call_args.kwargs['headers']['Range'],'bytes=8-')
            self.assertEqual(part.with_suffix('').read_bytes(),payload)
    def test_server_ignores_range(self):
        payload=b'complete weights fixture'
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);part=root/'models/loras/test.safetensors.part';part.parent.mkdir(parents=True);part.write_bytes(payload[:8])
            with mock.patch('requests.get',return_value=self.Response(payload,200)):
                catalog.download_one(self.row(payload),root)
            self.assertEqual(part.with_suffix('').read_bytes(),payload)
    def test_bad_hash_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);dest=root/'models/loras/test.safetensors';dest.parent.mkdir(parents=True);dest.write_bytes(b'old file')
            with mock.patch('requests.get',return_value=self.Response(b'corrupt')),mock.patch('catalog.time.sleep'),self.assertRaises(RuntimeError):
                catalog.download_one(self.row(b'expected'),root)
            self.assertEqual(dest.read_bytes(),b'old file')
            self.assertFalse(dest.with_name(dest.name+'.part').exists())

if __name__=='__main__':unittest.main(verbosity=2)
