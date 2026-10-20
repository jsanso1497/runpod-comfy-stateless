"""Focused offline tests. These are not ComfyUI execution or H3 GPU tests."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from graphlib import TopologicalSorter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from workbench.h3_first_last_contract import COMFY_COMMIT, h3_launch_options, inspect_schema


class TestH3Graphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.update = json.loads((ROOT / 'catalog/h3-first-last-update.json').read_text())
        cls.graphs = {v['file']: json.loads((ROOT / v['file']).read_text())
                      for task in cls.update['tasks'] for v in task['variants']}

    def test_links_are_reciprocal_and_acyclic(self):
        for path, graph in self.graphs.items():
            with self.subTest(graph=path):
                nodes = {n['id']: n for n in graph['nodes']}
                self.assertEqual(len(nodes), len(graph['nodes']))
                self.assertEqual(len({l[0] for l in graph['links']}), len(graph['links']))
                deps = {i: set() for i in nodes}
                for ident, src, out, dst, inp, kind in graph['links']:
                    self.assertIn(ident, nodes[src]['outputs'][out]['links'])
                    self.assertEqual(ident, nodes[dst]['inputs'][inp]['link'])
                    self.assertEqual(kind, nodes[src]['outputs'][out]['type'])
                    self.assertEqual(kind, nodes[dst]['inputs'][inp]['type'])
                    deps[dst].add(src)
                list(TopologicalSorter(deps).static_order())

    def test_same_photos_drive_both_refs_and_endpoint_anchors(self):
        for path, graph in self.graphs.items():
            if 'HFL4K' in path:
                continue
            nodes = {n['id']: n for n in graph['nodes']}
            refs = nodes[10]
            self.assertEqual(refs['type'], 'MiniMaxH3ReferenceToVideo')
            self.assertEqual(refs['widgets_values_named']['ref_image_size'], 'max')
            self.assertEqual(nodes[11]['widgets_values_named']['frame_idx'], 0)
            self.assertEqual(nodes[12]['widgets_values_named']['frame_idx'], -1)
            connections = {(src, dst, nodes[dst]['inputs'][inp]['name']) for _, src, _, dst, inp, _ in graph['links']}
            for edge in ((5, 10, 'ref_images.ref_image_0'), (6, 10, 'ref_images.ref_image_1'),
                         (5, 11, 'image'), (6, 12, 'image'), (3, 10, 'vae'),
                         (10, 11, 'positive'), (11, 12, 'positive'), (12, 13, 'conditioning')):
                self.assertIn(edge, connections)

    def test_quality_defaults_and_no_unwanted_generation_patches(self):
        for path, graph in self.graphs.items():
            nodes = {n['id']: n for n in graph['nodes']}
            if 'HFL4K' in path:
                continue
            settings = nodes[10]['widgets_values_named']
            self.assertEqual(settings['length'], 124)
            self.assertEqual(settings['length'] % 17, 5)
            self.assertEqual(settings['width'] * settings['height'], 768 * 1344)
            self.assertEqual(nodes[16]['widgets_values_named']['steps'], 30)
            self.assertEqual(nodes[15]['widgets_values_named']['sampler_name'], 'res_multistep')
            self.assertEqual(nodes[2]['widgets_values_named']['type'], 'minimax')
            self.assertTrue(nodes[1]['widgets_values'][0].endswith('_bf16.safetensors'))
            for node in nodes.values():
                self.assertEqual(node['mode'], 0)
                if node['type'] == 'MarkdownNote':
                    continue
                for value in node.get('widgets_values', []):
                    if isinstance(value, str) and value.endswith('.safetensors'):
                        self.assertFalse(any(s in value.lower() for s in ('int8', 'fp8', 'gguf', 'nvfp4', 'turbo')))
            self.assertFalse(any('Lora' in n['type'] or 'Sparse' in n['type'] for n in nodes.values()))

    def test_optional_face_ref_has_no_missing_required_load(self):
        for path, graph in self.graphs.items():
            if 'HFL4K' in path:
                continue
            load_nodes = [n for n in graph['nodes'] if n['type'] == 'LoadImage']
            self.assertEqual(len(load_nodes), 2)
            ref = next(n for n in graph['nodes'] if n['type'] == 'MiniMaxH3ReferenceToVideo')
            extra = next(p for p in ref['inputs'] if p['name'] == 'ref_images.ref_image_2')
            self.assertIsNone(extra['link'])
            self.assertNotIn('<Picture 3>', ref['widgets_values_named']['prompt'])

    def test_finish_preserves_source_audio_and_fps_and_keeps_native_separate(self):
        graph = next(g for p, g in self.graphs.items() if 'HFL4K' in p)
        nodes = {n['id']: n for n in graph['nodes']}
        self.assertEqual(nodes[1]['type'], 'WBH3LoadNativeMaster')
        self.assertEqual(nodes[2]['type'], 'WBH3ReleaseMemory')
        self.assertNotIn('SamplerCustomAdvanced', [n['type'] for n in nodes.values()])
        values = nodes[5]['widgets_values_named']
        self.assertEqual((values['resolution'], values['max_resolution']), (2160, 3840))
        self.assertEqual(values['batch_size'] % 4, 1)
        self.assertGreaterEqual(values['batch_size'], 5)
        self.assertTrue(values['uniform_batch_size'])
        self.assertEqual(values['temporal_overlap'], 4)
        self.assertEqual(values['input_noise_scale'], 0)
        self.assertEqual(values['latent_noise_scale'], 0)
        self.assertEqual(values['color_correction'], 'lab')
        for target in (6, 7):
            for slot, name in ((1, 'audio'), (2, 'fps')):
                self.assertTrue(any(src == 1 and out == slot and dst == target and nodes[dst]['inputs'][inp]['name'] == name
                                    for _, src, out, dst, inp, _ in graph['links']))

    def test_api_exports_match_all_ui_connections(self):
        for path, graph in self.graphs.items():
            name = Path(path).name.replace('.json', '.api.json')
            api_folder = 'HFL4K' if 'HFL4K' in path else 'HFL'
            api = json.loads((ROOT / 'automation' / api_folder / name).read_text())
            nodes = {n['id']: n for n in graph['nodes']}
            for _, src, out, dst, inp, _ in graph['links']:
                self.assertEqual(api[str(dst)]['inputs'][nodes[dst]['inputs'][inp]['name']], [str(src), out])
            self.assertFalse(any(n['class_type'] == 'MarkdownNote' for n in api.values()))

    def test_release_is_explicitly_unapproved_and_source_pinned(self):
        self.assertEqual(self.update['comfy_commit'], COMFY_COMMIT)
        self.assertFalse(self.update['approved'])
        self.assertFalse(self.update['gpu_tested'])
        self.assertFalse(self.update['docker_built'])
        self.assertTrue(all(t['approved'] is False for t in self.update['tasks']))


class TestH3LaunchAndSchema(unittest.TestCase):
    def test_h3_launch_uses_dense_pytorch_and_no_forced_low_precision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'comfy').mkdir()
            (root / 'comfy/cli_args.py').write_text('--use-pytorch-cross-attention\n--disable-comfy-compiler\n')
            self.assertEqual(h3_launch_options(root), ['--use-pytorch-cross-attention', '--disable-comfy-compiler'])
            self.assertEqual(h3_launch_options(root, '0'), ['--use-pytorch-cross-attention'])
            with self.assertRaises(ValueError):
                h3_launch_options(root, 'yes')

    def test_schema_check_detects_missing_nodes(self):
        report = inspect_schema({})
        self.assertFalse(report['ok'])
        self.assertIn('MiniMaxH3AddGuide', report['missing_nodes'])
        self.assertIn('WBH3LosslessMaster', report['missing_nodes'])


class TestH3MemoryRelease(unittest.TestCase):
    def test_release_is_not_cached_across_workflow_runs(self):
        import math
        import types
        from unittest.mock import patch
        fake = types.ModuleType('folder_paths')
        path = ROOT / 'src/nodes/h3_first_last_hq/__init__.py'
        spec = importlib.util.spec_from_file_location('h3_hq_nodes_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'folder_paths': fake}):
            spec.loader.exec_module(module)
        self.assertTrue(math.isnan(module.WBH3ReleaseMemory.IS_CHANGED()))
        self.assertEqual(set(module.NODE_CLASS_MAPPINGS),
                         {'WBH3LosslessMaster', 'WBH3LoadNativeMaster', 'WBH3ReleaseMemory'})


HAS_MEDIA = bool(shutil.which('ffmpeg') and shutil.which('ffprobe')
                 and importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'))


@unittest.skipUnless(HAS_MEDIA, 'Optional CPU media test needs torch, numpy and ffmpeg; no H3 weights are used.')
class TestH3LosslessMedia(unittest.TestCase):
    def test_rgb16_and_audio_roundtrip(self):
        import torch
        from workbench.h3_hq_media import save_master, read_native_master
        generator = torch.Generator().manual_seed(1497)
        images = torch.rand((5, 32, 48, 3), generator=generator)
        audio = {'waveform': torch.rand((1, 2, 4000), generator=generator) * 0.1, 'sample_rate': 24000}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'native.mkv'
            save_master(images, path, 24, audio)
            result, recovered_audio, fps = read_native_master(path)
            self.assertTrue(torch.equal(result, torch.round(images * 65535) / 65535))
            self.assertTrue(torch.equal(audio['waveform'], recovered_audio['waveform']))
            self.assertEqual(fps, 24)
            with self.assertRaises(FileExistsError):
                save_master(images, path, 24, audio)

    def test_half_and_bfloat16_image_inputs(self):
        import torch
        from workbench.h3_hq_media import save_master, read_native_master
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temp:
                images = torch.linspace(0, 1, 5 * 32 * 32 * 3).reshape(5, 32, 32, 3).to(dtype)
                path = Path(temp) / 'native.mkv'
                save_master(images, path, 24)
                result, _, _ = read_native_master(path)
                self.assertTrue(torch.equal(result, torch.round(images.float() * 65535) / 65535))

    def test_silent_master_roundtrip(self):
        import torch
        from workbench.h3_hq_media import save_master, read_native_master
        images = torch.zeros((5, 32, 32, 3))
        images[:, :, :, 0] = 1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'silent.mkv'
            save_master(images, path, 24)
            result, audio, fps = read_native_master(path)
            self.assertTrue(torch.equal(result, images))
            self.assertIsNone(audio)
            self.assertEqual(fps, 24)

    def test_invalid_pixels_and_interruption_leave_no_master(self):
        import torch
        from workbench.h3_hq_media import save_master
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'invalid.mkv'
            images = torch.full((1, 32, 32, 3), float('nan'))
            with self.assertRaises(ValueError):
                save_master(images, path)
            self.assertFalse(path.exists())
            def interrupt():
                raise RuntimeError('cancelled')
            with self.assertRaises(RuntimeError):
                save_master(torch.zeros((1, 32, 32, 3)), path, interrupt=interrupt)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
