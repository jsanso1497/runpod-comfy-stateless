"""Offline still graph, sizing, export and schema tests. No H3 inference."""
from __future__ import annotations
from graphlib import TopologicalSorter
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from workbench.h3_still_tools import still_dimensions, encode_rgb16_png, write_rgb16_png
from workbench.h3_still_contract import inspect_schema, verify_sources, SOURCE_BLOBS, COMFY_COMMIT

HAS_TORCH = bool(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'))


def load_local_nodes():
    spec = importlib.util.spec_from_file_location('h3_still_nodes_test', ROOT / 'src/nodes/h3_still_hq/__init__.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StillGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.update = json.loads((ROOT / 'catalog/h3-stills-update.json').read_text())
        cls.entries = [(task['id'], variant['file'], json.loads((ROOT / variant['file']).read_text()))
                       for task in cls.update['tasks'] for variant in task['variants']]

    def test_links_types_and_order(self):
        for _, path, graph in self.entries:
            with self.subTest(path=path):
                nodes = {n['id']: n for n in graph['nodes']}
                self.assertEqual(len(nodes), len(graph['nodes']))
                self.assertEqual(len(set(l[0] for l in graph['links'])), len(graph['links']))
                deps = {i: set() for i in nodes}
                for lid, src, out, dst, inp, typ in graph['links']:
                    self.assertIn(lid, nodes[src]['outputs'][out]['links'])
                    self.assertEqual(nodes[dst]['inputs'][inp]['link'], lid)
                    self.assertEqual(nodes[src]['outputs'][out]['type'], typ)
                    self.assertEqual(nodes[dst]['inputs'][inp]['type'], typ)
                    self.assertLess(nodes[src]['order'], nodes[dst]['order'])
                    deps[dst].add(src)
                list(TopologicalSorter(deps).static_order())

    def test_single_frame_latent_not_conditioner_latent_reaches_sampler(self):
        for _, _, graph in self.entries:
            nodes = {n['id']: n for n in graph['nodes']}
            self.assertEqual(nodes[7]['type'], 'FizgigH3StillLatent')
            self.assertEqual(nodes[7]['widgets_values_named']['batch_size'], 1)
            self.assertEqual(nodes[6]['outputs'][1]['links'], [])
            self.assertTrue(any(src == 7 and dst == 13 and nodes[dst]['inputs'][inp]['name'] == 'latent_image'
                                for _, src, _, dst, inp, _ in graph['links']))
            self.assertNotIn('FizgigH3StillDecode', [n['type'] for n in graph['nodes']])
            self.assertEqual(nodes[14]['type'], 'VAEDecode')

    def test_shared_dimensions_cannot_silently_diverge(self):
        for _, _, graph in self.entries:
            nodes = {n['id']: n for n in graph['nodes']}
            for target in (6, 7):
                for output, field in ((0, 'width'), (1, 'height')):
                    self.assertTrue(any(src == 5 and out == output and dst == target
                                        and nodes[dst]['inputs'][inp]['name'] == field
                                        for _, src, out, dst, inp, _ in graph['links']))

    def test_quality_recipe_and_full_precision(self):
        for task, _, graph in self.entries:
            nodes = {n['id']: n for n in graph['nodes']}
            self.assertEqual(nodes[11]['widgets_values_named']['sampler_name'], 'er_sde')
            self.assertEqual(nodes[12]['widgets_values_named'], {'scheduler':'simple','steps':50,'denoise':1.0})
            self.assertEqual(nodes[10]['widgets_values_named']['control_after_generate'], 'fixed')
            self.assertEqual(nodes[2]['widgets_values_named']['type'], 'minimax')
            self.assertEqual(nodes[3]['widgets_values_named']['vae_name'], 'minimax_h3_video_vae_fp16.safetensors')
            for n in graph['nodes']:
                self.assertEqual(n['mode'], 0)
                self.assertFalse(any(s in n['type'] for s in ('Lora', 'Sparse', 'Upscal', 'FaceRestore')))
                if n['type'] == 'MarkdownNote':
                    continue
                for value in n['widgets_values']:
                    if isinstance(value, str) and value.endswith('.safetensors'):
                        self.assertFalse(any(s in value for s in ('int8', 'fp8', 'nvfp4', 'turbo', 'gguf')))

    def test_reference_baseline_is_one_real_reference_not_a_video_anchor(self):
        for task, _, graph in self.entries:
            if task != 'HSTILL':
                continue
            nodes = {n['id']: n for n in graph['nodes']}
            self.assertEqual(sum(n['type'] == 'LoadImage' for n in nodes.values()), 1)
            self.assertEqual(nodes[6]['type'], 'MiniMaxH3ReferenceToVideo')
            self.assertEqual(nodes[6]['widgets_values_named']['ref_image_size'], 'max')
            self.assertNotIn('<Picture 2>', nodes[6]['widgets_values_named']['prompt'])
            self.assertNotIn('MiniMaxH3AddGuide', [n['type'] for n in nodes.values()])
            self.assertTrue(any(src == 3 and dst == 6 and nodes[dst]['inputs'][inp]['name'] == 'vae'
                                for _, src, _, dst, inp, _ in graph['links']))
            self.assertEqual(nodes[1]['widgets_values_named']['unet_name'], 'minimax_h3_ref2va_bf16.safetensors')

    def test_text_mode_requires_fl2va_and_has_no_reference_load(self):
        _, _, graph = next(row for row in self.entries if row[0] == 'HSTXT')
        nodes = {n['id']: n for n in graph['nodes']}
        self.assertEqual(nodes[6]['type'], 'MiniMaxH3ImageToVideo')
        self.assertEqual(nodes[1]['widgets_values_named']['unet_name'], 'minimax_h3_fl2va_bf16.safetensors')
        self.assertNotIn('LoadImage', [n['type'] for n in nodes.values()])
        self.assertEqual(nodes[5]['widgets_values_named']['aspect_ratio'], '16:9')

    def test_api_exports_match_ui_values_and_connections(self):
        for task, path, graph in self.entries:
            api = json.loads((ROOT / 'automation' / task / Path(path).name.replace('.json', '.api.json')).read_text())
            nodes = {n['id']: n for n in graph['nodes']}
            for _, src, out, dst, inp, _ in graph['links']:
                self.assertEqual(api[str(dst)]['inputs'][nodes[dst]['inputs'][inp]['name']], [str(src), out])
            for n in graph['nodes']:
                if n['type'] == 'MarkdownNote':
                    self.assertNotIn(str(n['id']), api)
                    continue
                linked = {p['name'] for p in n['inputs'] if p['link'] is not None}
                for k, v in n['widgets_values_named'].items():
                    if k not in linked | {'upload', 'control_after_generate'}:
                        self.assertEqual(api[str(n['id'])]['inputs'][k], v)

    def test_catalog_has_no_false_approval_or_default_extra_model_requirement(self):
        self.assertFalse(self.update['approved'])
        self.assertFalse(self.update['gpu_tested'])
        self.assertFalse(self.update['docker_built'])
        ref = next(t for t in self.update['tasks'] if t['id'] == 'HSTILL')
        self.assertEqual(ref['requires'], ['h3'])
        asset = self.update['assets'][0]
        self.assertEqual(asset['group'], 'h3_fl2va')
        self.assertEqual(len(asset['sha256']), 64)
        self.assertEqual(len(asset['revision']), 40)
        self.assertEqual(self.update['external_node']['commit'], '10d5171f6e45b0757a03df977f53a660e060c33d')


class StillSizing(unittest.TestCase):
    def test_documented_widescreen_8mp_size(self):
        self.assertEqual(still_dimensions(8.4, '16:9')[:2], (3872, 2176))
        self.assertEqual(still_dimensions(8.4, '9:16')[:2], (2176, 3872))

    def test_source_aspect_uses_geometry_only(self):
        w, h, report = still_dimensions(8.4, 'source', (2000, 3000))
        self.assertEqual((w, h), (2368, 3552))
        self.assertIn('not edited', report)
        self.assertLess(abs(w / h - 2/3), 0.01)

    def test_grid_and_cap_for_every_aspect(self):
        for aspect in ('16:9','9:16','3:2','2:3','1:1','4:5','5:4','3:4','4:3'):
            for mp in (0.25, 2.5, 3.0, 8.4, 16.777216):
                w, h, _ = still_dimensions(mp, aspect)
                self.assertEqual((w % 32, h % 32), (0, 0))
                self.assertGreaterEqual(min(w, h), 64)
                self.assertLessEqual(max(w, h), 4096)

    def test_bad_sizes_and_missing_source_are_rejected(self):
        for mp in (-1, 0, 17, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                still_dimensions(mp, '16:9')
        with self.assertRaises(ValueError):
            still_dimensions(8.4, 'source')
        with self.assertRaises(ValueError):
            still_dimensions(8.4, 'source', (100, 10000))

    @unittest.skipUnless(HAS_TORCH, 'CPU image test needs torch and numpy; no H3 weights.')
    def test_local_size_node_does_not_modify_source(self):
        import torch
        ref = torch.rand(1, 90, 60, 3)
        before = ref.clone()
        self.assertEqual(load_local_nodes().WBH3StillSize().size(8.4, 'source', ref)[:2], (2368, 3552))
        self.assertTrue(torch.equal(ref, before))


@unittest.skipUnless(HAS_TORCH, 'CPU export tests need torch and numpy; no H3 weights.')
class StillPNG(unittest.TestCase):
    def test_true_rgb16_png_pixels_and_metadata(self):
        import torch
        import numpy as np
        image = torch.linspace(0, 1, 17 * 23 * 3).reshape(17, 23, 3)
        data = encode_rgb16_png(image)
        self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(struct.unpack('>IIBBBBB', data[16:29]), (23, 17, 16, 2, 0, 0, 0))
        expected = (image * 65535).round().numpy().astype(np.uint16)
        # Validate every CRC, then independently decode with OpenCV when available.
        offset, compressed = 8, b''
        while offset < len(data):
            length = struct.unpack('>I', data[offset:offset+4])[0]
            kind = data[offset+4:offset+8]
            block = data[offset+8:offset+8+length]
            crc = struct.unpack('>I', data[offset+8+length:offset+12+length])[0]
            self.assertEqual(crc, zlib.crc32(kind + block) & 0xffffffff)
            if kind == b'IDAT':
                compressed += block
            offset += length + 12
        raw = zlib.decompress(compressed)
        rows = []
        for y in range(17):
            start = y * (1 + 23 * 6)
            self.assertEqual(raw[start], 0)
            rows.append(np.frombuffer(raw[start+1:start+1+23*6], dtype='>u2').reshape(23,3))
        self.assertTrue(np.array_equal(expected, np.stack(rows)))
        if importlib.util.find_spec('cv2'):
            import cv2
            actual = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)[..., ::-1]
            self.assertEqual(actual.dtype, np.uint16)
            self.assertTrue(np.array_equal(expected, actual))
        self.assertNotIn(b'prompt', data)

    def test_half_and_bfloat_export_inputs(self):
        import torch
        for dtype in (torch.float16, torch.bfloat16):
            self.assertTrue(encode_rgb16_png(torch.zeros(8, 8, 3, dtype=dtype)).startswith(b'\x89PNG'))

    def test_bad_pixels_are_not_silently_saved(self):
        import torch
        with self.assertRaises(ValueError):
            encode_rgb16_png(torch.full((8, 8, 3), float('nan')))
        with self.assertRaises(ValueError):
            encode_rgb16_png(torch.zeros(8, 8, 4))

    def test_existing_master_is_never_overwritten(self):
        import torch
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'test.png'
            write_rgb16_png(torch.zeros(8,8,3), p)
            before = p.read_bytes()
            with self.assertRaises(FileExistsError):
                write_rgb16_png(torch.ones(8,8,3), p)
            self.assertEqual(p.read_bytes(), before)

    def test_export_node_preview_and_path_checks(self):
        import torch
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            fake = types.ModuleType('folder_paths')
            fake.get_output_directory = lambda: t
            fake.get_save_image_path = lambda prefix, out, w, h: (t, 'test', 1, '', prefix)
            with patch.dict(sys.modules, {'folder_paths': fake}):
                node = load_local_nodes().WBH3SaveStillMaster()
                result = node.save(torch.zeros(1,8,8,3), 'h3_still/test')
                self.assertTrue((root / result['ui']['images'][0]['filename']).is_file())
                self.assertEqual(result['result'][0].shape, (1,8,8,3))
                for prefix in ('../escape', '/tmp/escape', 'C:\\escape', 'a//b'):
                    with self.assertRaises(ValueError):
                        node.save(torch.zeros(1,8,8,3), prefix)


class StillContract(unittest.TestCase):
    def test_missing_nodes_and_model_are_reported(self):
        report = inspect_schema({})
        self.assertFalse(report['ok'])
        self.assertIn('VAEDecode', report['missing_nodes'])
        schema = {k: {'input': {'required': {}}} for k in ('KSamplerSelect','UNETLoader')}
        self.assertTrue(any('er_sde' in e for e in inspect_schema(schema)['problems']))

    def test_optional_text_mode_checks_the_correct_model(self):
        schema = {'UNETLoader': {'input': {'required': {'unet_name': [['minimax_h3_ref2va_bf16.safetensors']]}}}}
        ref = inspect_schema(schema)
        text = inspect_schema(schema, text_only=True)
        self.assertFalse(any('fl2va' in p for p in ref['problems']))
        self.assertTrue(any('fl2va' in p for p in text['problems']))

    def test_source_checks_fail_on_changed_core_or_node(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                verify_sources(t, '0' * 40)
            for path in SOURCE_BLOBS:
                p = Path(t) / path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('modified source\n')
            with self.assertRaises(ValueError):
                verify_sources(t)


if __name__ == '__main__':
    unittest.main()
