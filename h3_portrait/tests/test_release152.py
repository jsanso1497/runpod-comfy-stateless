import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference_video = load_module(ROOT / 'node' / 'reference_video.py', 'h3_reference_video_test')
installer = load_module(ROOT / 'install_workflows.py', 'h3_install_workflows_test')


class Release152Tests(unittest.TestCase):
    def test_source_reference_video_graph_is_local_only(self):
        graph = json.loads((ROOT / 'workflows' / 'H3_Reference_Video_Swap_Local.json').read_text())
        by_type = {}
        for node in graph['nodes']:
            by_type.setdefault(node['type'], []).append(node)

        self.assertIn('MiniMaxH3ReferencePack', by_type)
        ref = by_type['MiniMaxH3ReferencePack'][0]
        values = ref['widgets_values']
        self.assertEqual(values[3], 'local')
        self.assertEqual(values[4], '')
        self.assertEqual(values[5], '(hosted models disabled)')
        self.assertEqual(values[6], 'none')
        self.assertEqual(values[7], 'http://127.0.0.1:11434/v1')
        self.assertEqual(values[9], 'replacement')
        self.assertEqual(values[13], 2048)

        settings = by_type['H3ReferenceVideoSettings'][0]
        self.assertEqual(settings['widgets_values'], ['9:16', 'High fidelity', 5])
        native = by_type['MiniMaxH3ReferenceToVideo'][0]
        self.assertEqual(native['widgets_values'][1:5], [768, 1344, 124, 'max'])
        scheduler = by_type['BasicScheduler'][0]
        self.assertEqual(scheduler['widgets_values'][1], 25)

    def test_full_landscape_high_fidelity_geometry(self):
        got = reference_video.geometry('16:9', 'High fidelity', 5, profile='full')
        self.assertEqual((got['width'], got['height']), (1344, 768))
        self.assertEqual((got['output_width'], got['output_height']), (1344, 756))
        self.assertEqual(got['frames'], 124)
        self.assertEqual(got['steps'], 25)

    def test_lite_portrait_standard_geometry(self):
        got = reference_video.geometry('9:16', 'Standard', 5, profile='lite')
        self.assertEqual((got['width'], got['height']), (576, 1024))
        self.assertEqual((got['output_width'], got['output_height']), (576, 1024))
        self.assertEqual(got['frames'], 124)
        self.assertEqual(got['steps'], 16)

    def _install_for(self, settings_path):
        settings = json.loads(settings_path.read_text())
        with tempfile.TemporaryDirectory() as td:
            installer.install(td, settings, source=ROOT)
            files = sorted((Path(td) / 'user/default/workflows').glob('*.json'))
            return [(p.name, json.loads(p.read_text())) for p in files]

    def test_full_installer_uses_existing_local_32b_instruct_model(self):
        installed = self._install_for(ROOT / 'settings.json')
        self.assertEqual(len(installed), 3)
        name, graph = next(row for row in installed if 'Reference_Video' in row[0])
        self.assertEqual(name, 'H3_Reference_Video_Swap_Local_Full_v1_5_2.json')
        ref = next(n for n in graph['nodes'] if n['type'] == 'MiniMaxH3ReferencePack')
        self.assertEqual(ref['widgets_values'][3], 'local')
        self.assertEqual(ref['widgets_values'][7], 'http://127.0.0.1:11434/v1')
        self.assertEqual(ref['widgets_values'][8], 'huihui_ai/qwen3-vl-abliterated:32b-instruct-q4_K_M')
        self.assertEqual(ref['widgets_values'][9], 'replacement')

    def test_lite_installer_uses_existing_local_8b_instruct_model(self):
        installed = self._install_for(ROOT / 'profiles' / 'lite' / 'settings.json')
        self.assertEqual(len(installed), 4)
        name, graph = next(row for row in installed if 'Reference_Video' in row[0])
        self.assertEqual(name, 'H3_Reference_Video_Swap_Local_Lite_v1_5_2.json')
        ref = next(n for n in graph['nodes'] if n['type'] == 'MiniMaxH3ReferencePack')
        self.assertEqual(ref['widgets_values'][3], 'local')
        self.assertEqual(ref['widgets_values'][8], 'huihui_ai/qwen3-vl-abliterated:8b-instruct-q4_K_M')
        native = next(n for n in graph['nodes'] if n['type'] == 'MiniMaxH3ReferenceToVideo')
        self.assertEqual(native['widgets_values'][1:5], [576, 1024, 124, 'match'])
        sched = next(n for n in graph['nodes'] if n['type'] == 'BasicScheduler')
        self.assertEqual(sched['widgets_values'][1], 16)

    def test_docker_pins_and_patches_reference_pack(self):
        docker = (ROOT / 'Dockerfile').read_text()
        self.assertIn('H3_REFPACK_COMMIT=7012734eabf6f98063d6eaf8ce1f9264ee803664', docker)
        self.assertIn('Hearmeman24/ComfyUI-MiniMaxRefPack.git', docker)
        self.assertIn('patch_refpack_local.py', docker)
        self.assertIn('0.3.5-local-only', docker)

    def test_patcher_disables_hosted_prompt_provider(self):
        text = (ROOT / 'patch_refpack_local.py').read_text()
        self.assertIn('PROVIDERS = ("local", "none")', text)
        self.assertIn('DEFAULT_PROVIDER = "local"', text)
        self.assertIn('Hosted prompt providers are disabled', text)
        self.assertIn('(hosted models disabled)', text)
        self.assertIn('_DEFAULT_ENDPOINT = _endpoint.resolve(\"local\", \"http://127.0.0.1:11434/v1\")', text)


if __name__ == '__main__':
    unittest.main()
