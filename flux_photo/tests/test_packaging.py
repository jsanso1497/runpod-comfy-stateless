import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import assets
import check
import make_workflows
import runtime


class AssetTests(unittest.TestCase):
    def setUp(self): self.rows = json.loads((ROOT/'config/models.json').read_text())

    def test_official_immutable_three_model_manifest(self):
        self.assertEqual(len(assets.validate_manifest(self.rows)), 3)
        self.assertTrue(all(len(r['revision']) == 40 for r in self.rows))
        self.assertEqual([Path(r['destination']).name for r in self.rows],
                         ['flux-2-klein-9b.safetensors', 'qwen_3_8b.safetensors', 'flux2-vae.safetensors'])
        self.assertFalse(any('fp8' in r['filename'] for r in self.rows))

    def test_mutable_revision_rejected(self):
        self.rows[0]['revision'] = 'main'
        with self.assertRaises(ValueError): assets.validate_manifest(self.rows)

    def test_unofficial_repo_rejected(self):
        self.rows[0]['repo_id'] = 'someone/model'
        with self.assertRaises(ValueError): assets.validate_manifest(self.rows)

    def test_path_traversal_rejected(self):
        for path in ('../outside', '/etc/passwd', 'models/../escape.safetensors', 'models/x\\y', 'C:/model'):
            with self.subTest(path=path):
                rows = copy.deepcopy(self.rows); rows[0]['destination'] = path
                with self.assertRaises(ValueError): assets.validate_manifest(rows)

    def test_duplicate_destination_rejected(self):
        self.rows[1]['destination'] = self.rows[0]['destination']
        with self.assertRaises(ValueError): assets.validate_manifest(self.rows)

    def test_invalid_digest_rejected(self):
        self.rows[0]['sha256'] = 'not-a-digest'
        with self.assertRaises(ValueError): assets.validate_manifest(self.rows)

    def test_digest_matches_known_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'test'; p.write_bytes(b'abc')
            self.assertEqual(assets.digest(p), hashlib.sha256(b'abc').hexdigest())

    def fixture(self, td, bad_hash=False):
        payload = b'synthetic safetensors download for unit test'
        row = dict(self.rows[0], sha256=None, estimated_bytes=len(payload))
        manifest = Path(td)/'models.json'; manifest.write_text(json.dumps([row]))
        metadata = types.SimpleNamespace(commit_hash=row['revision'], etag=hashlib.sha256(payload).hexdigest(), size=len(payload))
        def download(**kw):
            path = Path(kw['local_dir'])/row['filename']; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'corrupt' if bad_hash else payload); return str(path)
        hub = types.SimpleNamespace(get_hf_file_metadata=Mock(return_value=metadata),
              hf_hub_url=lambda *a, **kw: 'https://huggingface.co/official', hf_hub_download=Mock(side_effect=download))
        return row, manifest, metadata, hub, payload

    def test_download_verified_and_second_start_skips_existing(self):
        with tempfile.TemporaryDirectory() as td:
            row, manifest, meta, hub, payload = self.fixture(td)
            home = Path(td)/'home'
            with patch.dict(sys.modules, {'huggingface_hub': hub}), patch('assets.shutil.disk_usage', return_value=types.SimpleNamespace(free=10**12)):
                assets.download_models(home, manifest); assets.download_models(home, manifest)
            self.assertEqual((home/row['destination']).read_bytes(), payload)
            self.assertEqual(hub.hf_hub_download.call_count, 1)

    def test_corrupt_download_not_installed(self):
        with tempfile.TemporaryDirectory() as td:
            row, manifest, meta, hub, _ = self.fixture(td, True)
            with patch.dict(sys.modules, {'huggingface_hub': hub}), patch('assets.shutil.disk_usage', return_value=types.SimpleNamespace(free=10**12)):
                with self.assertRaisesRegex(ValueError, 'integrity'): assets.download_models(Path(td)/'home', manifest)
            self.assertFalse((Path(td)/'home'/row['destination']).exists())

    def test_wrong_revision_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest, meta, hub, _ = self.fixture(td); meta.commit_hash = '0'*40
            with patch.dict(sys.modules, {'huggingface_hub': hub}):
                with self.assertRaisesRegex(ValueError, 'revision'): assets.download_models(Path(td)/'home', manifest)
            hub.hf_hub_download.assert_not_called()

    def test_auth_failure_no_mirror_substitution(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest, _, hub, _ = self.fixture(td); hub.get_hf_file_metadata.side_effect = RuntimeError('access denied')
            with patch.dict(sys.modules, {'huggingface_hub': hub}):
                with self.assertRaisesRegex(RuntimeError, 'No alternate model'): assets.download_models(Path(td)/'home', manifest)
            hub.hf_hub_download.assert_not_called()

    def test_insufficient_disk_fails_before_download(self):
        with tempfile.TemporaryDirectory() as td:
            _, manifest, _, hub, _ = self.fixture(td)
            with patch.dict(sys.modules, {'huggingface_hub': hub}), patch('assets.shutil.disk_usage', return_value=types.SimpleNamespace(free=1)):
                with self.assertRaisesRegex(RuntimeError, 'free disk'): assets.download_models(Path(td)/'home', manifest)
            hub.hf_hub_download.assert_not_called()

    def test_fixed_hash_must_match_authenticated_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            row, manifest, _, hub, _ = self.fixture(td); row['sha256'] = '0'*64; manifest.write_text(json.dumps([row]))
            with patch.dict(sys.modules, {'huggingface_hub': hub}):
                with self.assertRaisesRegex(ValueError, 'recorded digest'): assets.download_models(Path(td)/'home', manifest)


class WorkflowTests(unittest.TestCase):
    def test_static_check_both_presets(self): check.static_check()

    def test_three_none_loras_and_two_inputs(self):
        for native in (False, True):
            _, graph = make_workflows.build(native)
            loras = [n for n in graph['nodes'] if n['type'] == 'FluxPhotoOptionalLoRA']
            self.assertEqual(len(loras), 3)
            self.assertTrue(all(n['widgets_values_named']['lora_name'] == 'None' for n in loras))
            self.assertEqual(len([n for n in graph['nodes'] if n['type'] == 'LoadImage']), 2)

    def test_preset_modes_and_limit_differ(self):
        for native, mode, cap in [(False, 'fit', 2048), (True, 'native', 3072)]:
            _, graph = make_workflows.build(native)
            crop = next(n for n in graph['nodes'] if n['type'] == 'FluxPhotoCrop')
            self.assertEqual(crop['widgets_values_named']['processing'], mode)
            self.assertEqual(crop['widgets_values_named']['max_crop_side'], cap)

    def test_seed_control_widget_not_api_input(self):
        _, graph = make_workflows.build(False)
        n = next(n for n in graph['nodes'] if n['type'] == 'FluxPhotoSampler')
        self.assertEqual(n['widgets_values'], [12345, 'fixed', 4])
        self.assertNotIn('control_after_generate', make_workflows.api_graph(graph)[str(n['id'])]['inputs'])

    def test_broken_link_rejected(self):
        _, graph = make_workflows.build(False); graph['links'][0][5] = 'WRONG'
        with self.assertRaises(ValueError): check.validate_graph(graph)

    def test_missing_socket_rejected(self):
        _, graph = make_workflows.build(False)
        n = next(n for n in graph['nodes'] if n['inputs']); n['inputs'][0]['link'] = None
        with self.assertRaises(ValueError): check.validate_graph(graph)

    def test_no_api_inference_or_prompt_rewriter(self):
        _, graph = make_workflows.build(False)
        allowed = {'LoadImage', 'CLIPTextEncode', 'VAEDecodeTiled', 'Note'}
        self.assertTrue(all(n['type'].startswith('FluxPhoto') or n['type'] in allowed for n in graph['nodes']))

    def test_compare_and_save_are_wired_to_same_stitch_image(self):
        _, graph = make_workflows.build(False)
        api = make_workflows.api_graph(graph)
        compare = next(v for v in api.values() if v['class_type'] == 'FluxPhotoCompare')
        save = next(v for v in api.values() if v['class_type'] == 'FluxPhotoSave')
        self.assertEqual(compare['inputs']['edited'], save['inputs']['images'])
        self.assertEqual(compare['inputs']['report'], save['inputs']['report'])


class RuntimeTests(unittest.TestCase):
    def test_install_and_preserve_user_workflow_changes(self):
        with tempfile.TemporaryDirectory() as td:
            target = runtime.install_workflows(td)
            files = sorted(target.glob('*.json')); self.assertEqual(len(files), 2)
            files[0].write_text('my edit')
            runtime.install_workflows(td)
            self.assertEqual(files[0].read_text(), 'my edit')

    def test_flag_accepts_only_explicit_zero_one(self):
        for value, expected in [('0', False), ('1', True)]:
            with patch.dict(os.environ, {'UNIT_FLAG': value}): self.assertIs(runtime.flag('UNIT_FLAG'), expected)
        with patch.dict(os.environ, {'UNIT_FLAG': 'true'}):
            with self.assertRaises(ValueError): runtime.flag('UNIT_FLAG')

    def test_config_url_never_accepts_credentials_or_external_hosts(self):
        for value in ['https://token@github.com/a/b', 'https://evil.example/a/b', 'file:///tmp/repo',
                      'https://github.com/a/b?token=secret', 'https://github.com/a/b/tree/main']:
            with self.subTest(value=value), patch.dict(os.environ, {'CONFIG_REPO': value}):
                with self.assertRaises(ValueError): runtime.lora_links()

    def test_config_ref_cannot_be_an_option(self):
        with patch.dict(os.environ, {'CONFIG_REPO': 'https://github.com/a/b', 'CONFIG_REF': '--upload-pack=bad'}):
            with self.assertRaises(ValueError): runtime.lora_links()

    def test_default_links_path_is_bundled_config(self):
        with patch.dict(os.environ, {'CONFIG_REPO': ''}):
            self.assertEqual(runtime.lora_links(), ROOT/'config/lora_links.txt')


if __name__ == '__main__': unittest.main()
