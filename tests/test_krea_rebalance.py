"""Correction regressions: Rebalance belongs to Krea; H3 and restoration stay unchanged."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import catalog
import check_workflows

CFG = ROOT/'config'
KREA = ['10_Krea2_Same_Subject.json', '11_Krea2_Subject_into_Scene.json']
RETIRED = '30_Ideogram4_Rebalance_Reference_Quality.json'
WEIGHTS = '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0'

def graph(name):
    return json.loads((CFG/'workflows'/name).read_text())

def by_type(g, name):
    return [n for n in g['nodes'] if n['type'] == name]

def source_node(g, node, socket):
    item = next(i for i in node['inputs'] if i['name'] == socket)
    edge = next(l for l in g['links'] if l[0] == item['link'])
    return next(n for n in g['nodes'] if n['id'] == edge[1])

class KreaRebalanceTests(unittest.TestCase):
    def test_both_sampler_branches_use_rebalance(self):
        for name in KREA:
            g = graph(name)
            sampler = by_type(g, 'KSampler')[0]
            for branch in ['positive', 'negative']:
                scaler = source_node(g, sampler, branch)
                self.assertEqual(scaler['type'], 'ConditioningKrea2Rebalance')
                self.assertEqual(scaler['mode'], 0)
                self.assertEqual(source_node(g, scaler, 'conditioning')['type'], 'Krea2EditGroundedEncode')

    def test_rebalance_only_in_the_two_krea_workflows(self):
        affected = []
        for path in catalog.workflow_sources(CFG):
            g = json.loads(path.read_text())
            if any('Rebalance' in n['type'] for n in g['nodes']):
                affected.append(path.name)
        self.assertEqual(affected, KREA)

    def test_same_weights_both_branches_and_no_global_four_times_boost(self):
        for name in KREA:
            scalers = by_type(graph(name), 'ConditioningKrea2Rebalance')
            self.assertEqual(len(scalers), 2)
            for node in scalers:
                self.assertEqual(node['widgets_values_named'], {'multiplier': 1.0, 'per_layer_weights': WEIGHTS})

    def test_original_trained_grounding_is_retained(self):
        for name in KREA:
            g = graph(name)
            encoders = by_type(g, 'Krea2EditGroundedEncode')
            self.assertEqual(len(encoders), 2)
            sampler = by_type(g, 'KSampler')[0]
            negative = source_node(g, source_node(g, sampler, 'negative'), 'conditioning')
            self.assertEqual(negative['widgets_values_named']['prompt'], '')
            self.assertEqual(negative['widgets_values_named']['grounding_px'], 1024)
            self.assertEqual(len(by_type(g, 'Krea2EditModelPatch')), 1)
            check_workflows.validate_graph(g)

    def test_no_eight_step_rebalance_encoder_schedule(self):
        for name in KREA:
            types = {n['type'] for n in graph(name)['nodes']}
            self.assertFalse(types.intersection({'Krea2EditRebalance', 'RebalanceCFG', 'StepRebalance'}))

    def test_labels_match_actual_raw_and_guidance_settings(self):
        for name in KREA:
            g = graph(name)
            self.assertIn('Raw', by_type(g, 'UNETLoader')[0]['title'])
            self.assertEqual(by_type(g, 'UNETLoader')[0]['widgets_values_named']['unet_name'], 'krea2_raw_bf16.safetensors')
            self.assertIn('20 steps / CFG 3', by_type(g, 'KSampler')[0]['title'])
            self.assertNotIn('CFG 1', json.dumps(g))

    def test_disabled_rebalance_rejected_in_recipe(self):
        g = graph(KREA[0])
        by_type(g, 'ConditioningKrea2Rebalance')[0]['mode'] = 4
        with self.assertRaisesRegex(ValueError, 'active'):
            check_workflows.validate_graph(g)

    def test_mismatched_branch_scaling_rejected(self):
        g = graph(KREA[0])
        by_type(g, 'ConditioningKrea2Rebalance')[1]['widgets_values_named']['multiplier'] = 2.0
        with self.assertRaisesRegex(ValueError, 'matching layer scaling'):
            check_workflows.validate_graph(g)

    def test_dropped_layer_rejected(self):
        g = graph(KREA[0])
        by_type(g, 'ConditioningKrea2Rebalance')[0]['widgets_values_named']['per_layer_weights'] = '0,' + ','.join(['1'] * 11)
        with self.assertRaisesRegex(ValueError, 'twelve'):
            check_workflows.validate_graph(g)

    def test_wrong_profile_rejected(self):
        g = graph(KREA[0])
        g['extra']['everyday']['profile'] = 'h3'
        with self.assertRaisesRegex(ValueError, 'only in the Krea'):
            check_workflows.validate_graph(g)

    def test_runtime_has_only_requested_profiles(self):
        d = catalog.read_json(CFG/'runtime.json')
        self.assertEqual(set(d['profiles']), {'seedvr2', 'krea2', 'h3'})
        self.assertEqual(d['package_version'], '3.2.0')
        self.assertEqual(d['retired_workflows'], [RETIRED])

    def test_no_ideogram_weights_or_active_workflows(self):
        rows = catalog.catalog(CFG, Path('/tmp/comfy'), {'seedvr2','krea2','h3','ideogram4'})
        self.assertFalse(any('ideogram' in json.dumps(r).lower() for r in rows))
        self.assertFalse(any('ideogram' in p.name.lower() for p in catalog.workflow_sources(CFG)))

    def test_obsolete_ideogram_profile_gets_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'seedvr2, krea2, h3'):
            catalog.selected_profiles(CFG, 'seedvr2,krea2,h3,ideogram4')

    def test_leftover_ideogram_recipe_is_not_installed(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            shutil.copytree(CFG, p/'config')
            (p/'config/workflows'/RETIRED).write_text('{}')
            root = p/'comfy'
            installed = catalog.install_workflows(p/'config', root, {'seedvr2','krea2','h3'})
            self.assertFalse((root/'user/default/workflows'/RETIRED).exists())
            self.assertTrue({Path(n).stem+'_safe_v1_5_3.json' for n in KREA}.issubset({x.name for x in installed}))
            self.assertNotIn(RETIRED, {x.name for x in check_workflows.graph_paths(p/'config')})

    def test_local_rebalance_schema_contract(self):
        # Contract read from the pinned upstream node. Live schemas are checked at Docker build.
        schema = {
            'input': {'required': {
                'conditioning': ['CONDITIONING', {}],
                'multiplier': ['FLOAT', {'default': 4.0}],
                'per_layer_weights': ['STRING', {'default': WEIGHTS}],
            }},
            'output': ['CONDITIONING'],
        }
        for name in KREA:
            for node in by_type(graph(name), 'ConditioningKrea2Rebalance'):
                check_workflows.validate_schema({'nodes': [node]}, {'ConditioningKrea2Rebalance': schema}, False)

    def test_real_upstream_tensor_probe_included_in_build(self):
        docker = (ROOT/'Dockerfile').read_text()
        prepare = (ROOT/'scripts/prepare_image.sh').read_text()
        self.assertIn('scripts/check_rebalance_runtime.py', docker)
        self.assertIn('python "$SCRIPTS/check_rebalance_runtime.py" --comfy-home "$CORE"', prepare)

    def test_only_actual_generation_models_remain(self):
        self.assertEqual(len(catalog.read_json(CFG/'models.json')), 9)
        text = (ROOT/'Dockerfile').read_text()
        self.assertNotIn('Ideogram', text)
        self.assertIn('comfy-quality-3.2.0', text)

if __name__ == '__main__':
    unittest.main(verbosity=2)
