import importlib.util
import inspect
import json
from pathlib import Path
import unittest
import torch
HERE=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
N=load('_krea_text_scene_test',HERE/'node/__init__.py')
C=load('_krea_text_scene_check',HERE/'check.py')

def args():
    vals={}
    for name,d in N.KreaIdentityTextSceneTwoRefs.INPUT_TYPES()['required'].items():
        vals[name]=torch.rand(1,48,64,3) if d[0]=='IMAGE' else d[1]['default']
    return vals

class TextScene(unittest.TestCase):
    def test_original_references_are_not_collaged_or_generated(self):
        a=args();out=N.KreaIdentityTextSceneTwoRefs().prepare(**a)
        self.assertEqual(out[0].data_ptr(),a['reference_a'].data_ptr());self.assertEqual(out[1].data_ptr(),a['reference_b'].data_ptr())
    def test_default_is_one_person(self):self.assertIn('ONE and the same person',N.KreaIdentityTextSceneTwoRefs().prepare(**args())[2])
    def test_two_people_keeps_identities_separate(self):
        a=args();a['reference_relationship']=N.TEXT_SCENE_MODES[1];p=N.KreaIdentityTextSceneTwoRefs().prepare(**a)[2]
        self.assertIn('TWO distinct',p);self.assertNotIn('SAME person',p);self.assertNotIn('ONE and the same',p)
    def test_text_scene_and_purposes_are_literal(self):
        a=args();a.update(reference_a_use='Jaw and eyes.',reference_b_use='Shoulder proportions.',scene_description='At a cafe in Paris.',extra_instruction='No glasses.')
        p=N.KreaIdentityTextSceneTwoRefs().prepare(**a)[2]
        for text in ('Jaw and eyes.','Shoulder proportions.','At a cafe in Paris.','No glasses.'):self.assertIn(text,p)
    def test_all_aspects_exact_grid_and_bounded(self):
        for aspect,(a,b) in N.TEXT_SCENE_ASPECTS.items():
            for mp in (.25,.5,1,1.5,2):
                with self.subTest(aspect=aspect,mp=mp):
                    w,h=N.described_canvas(aspect,mp);self.assertEqual(w*b,h*a);self.assertEqual(w%16,0);self.assertEqual(h%16,0);self.assertLessEqual(w*h,mp*1e6)
    def test_invalid_megapixels(self):
        for val in (True,None,-1,0,2.1,float('nan'),float('inf')):
            with self.subTest(val=val),self.assertRaises(ValueError):N.described_canvas('1:1 square',val)
    def test_invalid_aspect(self):
        with self.assertRaises(ValueError):N.described_canvas('free',1.5)
    def test_empty_fields_rejected(self):
        for key in ('reference_a_use','reference_b_use','scene_description'):
            a=args();a[key]=' '
            with self.subTest(key=key),self.assertRaises(ValueError):N.KreaIdentityTextSceneTwoRefs().prepare(**a)
    def test_invalid_relationship(self):
        a=args();a['reference_relationship']='guess'
        with self.assertRaises(ValueError):N.KreaIdentityTextSceneTwoRefs().prepare(**a)
    def test_node_schema_matches_signature(self):
        schema=N.KreaIdentityTextSceneTwoRefs.INPUT_TYPES()['required'];sig=inspect.signature(N.KreaIdentityTextSceneTwoRefs.prepare)
        self.assertEqual(set(schema),set(sig.parameters)-{'self'})
    def test_new_graph_static(self):C.validate_graph(json.loads(next((HERE/'workflows').glob('*08*')).read_text()))
    def test_exactly_two_uploads_no_scene_node(self):
        g=json.loads(next((HERE/'workflows').glob('*08*')).read_text());types=[n['type'] for n in g['nodes']]
        self.assertEqual(types.count('LoadImage'),2);self.assertIn('KreaIdentityTextSceneTwoRefs',types)
        self.assertNotIn('KreaIdentityPrepare',types);self.assertNotIn('KreaIdentityPair',types)
    def test_balanced_reference_strengths_and_original_rebalancer(self):
        g=json.loads(next((HERE/'workflows').glob('*08*')).read_text());nodes={n['type']:n for n in g['nodes']}
        patch=nodes['Krea2EditModelPatch']['widgets_values_named'];self.assertEqual((patch['ref_boost_a'],patch['ref_boost']),(4,4))
        self.assertIn('ConditioningKrea2Rebalance',nodes)
    def test_new_workflow_generation_reproducible(self):
        file=next((HERE/'workflows').glob('*08*'));before=file.read_bytes();builder=load('_krea_text_scene_build',HERE/'make_text_scene_workflow.py');builder.make();self.assertEqual(file.read_bytes(),before)
