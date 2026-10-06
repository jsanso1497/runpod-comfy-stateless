import copy
import importlib.util
import json
from pathlib import Path
import unittest

REPO=Path(__file__).resolve().parents[2]
# Repair CLI is a repo tool, not required inside the minimal Docker node bundle.
REPAIR=REPO/'tools/repair_workflow_models.py'


@unittest.skipUnless(REPAIR.is_file(),'Repository repair CLI is not copied to this minimal Docker stage.')
class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sp=importlib.util.spec_from_file_location('repair_test',REPAIR)
        cls.mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(cls.mod)

    def krea(self):
        return {'nodes':[{'id':66,'type':'UNETLoader','widgets_values':['krea2_turbo_bf16.safetensors','default']},
                         {'id':62,'type':'CLIPLoader','widgets_values':['qwen3vl_32b_minimax_h3_bf16.safetensors','lumina2','default']},
                         {'id':63,'type':'VAELoader','widgets_values':['minimax_h3_audio_vae_fp32.safetensors']},
                         {'id':70,'type':'KSampler','widgets_values':[42,'fixed',8,1,'res_multistep','simple',1]}], 'links':[], 'revision':0}

    def test_repairs_exact_reported_krea_model_mixup(self):
        g=self.krea();got,changes=self.mod.repair(g)
        ns={n['id']:n for n in got['nodes']}
        self.assertEqual(ns[62]['widgets_values'],['qwen3vl_4b_bf16.safetensors','krea2','default'])
        self.assertEqual(ns[63]['widgets_values'],['qwen_image_vae.safetensors'])
        self.assertEqual(ns[70]['widgets_values'],g['nodes'][3]['widgets_values'])
        self.assertEqual(len(changes),3)

    def test_original_is_never_mutated(self):
        g=self.krea();saved=copy.deepcopy(g);self.mod.repair(g);self.assertEqual(g,saved)

    def test_mixed_family_is_not_guessed(self):
        g=self.krea();g['nodes'].append({'id':80,'type':'UNETLoader','widgets_values':['minimax_h3_ref2va_bf16.safetensors','default']})
        with self.assertRaisesRegex(ValueError,'Mixed'):self.mod.repair(g)

    def test_h3_audio_role_comes_from_links_not_wrong_filename(self):
        path=REPO/'h3_portrait/workflows/H3_Reference_Video_Swap_Local.json'
        graph=json.loads(path.read_text())
        vae=next(n for n in graph['nodes'] if n['id']==4)
        vae['widgets_values']=['qwen_image_vae.safetensors']
        encoder=next(n for n in graph['nodes'] if n['id']==2)
        encoder['widgets_values'][1]='lumina2'
        got,_=self.mod.repair(graph)
        self.assertEqual(next(n for n in got['nodes'] if n['id']==4)['widgets_values'],['minimax_h3_audio_vae_fp32.safetensors'])
        self.assertEqual(next(n for n in got['nodes'] if n['id']==2)['widgets_values'][1],'minimax')

    def test_protected_output_requires_correct_nodes(self):
        got,_=self.mod.repair(self.krea(),protected=True)
        self.assertEqual(got['nodes'][1]['type'],'SafeKreaCLIPLoader')


if __name__=='__main__':unittest.main()
