"""CPU-only schema registration regression tests; no GPU inference is simulated.

Fixtures reproduce the upstream loader's device-list/default contract. They are
not a substitute for the real ComfyUI registry check that still runs in Docker.
"""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('krea_seed_schema_install', HERE/'install.py')
I = importlib.util.module_from_spec(spec)
spec.loader.exec_module(I)
NAMES = ('dit_model_loader.py', 'vae_model_loader.py')
ORIGINAL = b'        devices = get_device_list()\n'
CORRECTED = b'        devices = get_device_list() or ["cpu"]\n'
FIXTURE = '''class Loader:
    @classmethod
    def define_schema(cls):
        devices = get_device_list()
        return {"options": devices, "default": devices[0]}

    @classmethod
    def execute(cls, model, device):
        return {"model": model, "device": device}
'''


class SeedVR2CpuSchemaTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.seed = self.root/'seed'
        self.folder = self.seed/'src/interfaces'
        self.folder.mkdir(parents=True)
        for name in NAMES:
            (self.folder/name).write_text(FIXTURE)

    def apply(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return I.patch_seedvr2_cpu_schemas(self.seed)

    def schema(self, name, devices):
        env = {'get_device_list': lambda: devices}
        exec(compile((self.folder/name).read_bytes(), name, 'exec'), env)
        return env['Loader'].define_schema()

    def test_original_cpu_failure_is_reproduced_for_both_loaders(self):
        for name in NAMES:
            with self.subTest(name=name), self.assertRaises(IndexError):
                self.schema(name, [])

    def test_cpu_fallback_registers_both_loader_schemas(self):
        self.assertEqual(self.apply(), 2)
        for name in NAMES:
            self.assertEqual(self.schema(name, []), {'options': ['cpu'], 'default': 'cpu'})

    def test_cuda_choices_and_first_device_remain_identical(self):
        self.apply()
        devices = ['cuda:0', 'cuda:1']
        for name in NAMES:
            schema = self.schema(name, devices)
            self.assertIs(schema['options'], devices)
            self.assertEqual(schema['default'], 'cuda:0')

    def test_mps_choice_is_not_replaced_by_cpu(self):
        self.apply()
        devices = ['mps']
        for name in NAMES:
            self.assertIs(self.schema(name, devices)['options'], devices)
            self.assertEqual(self.schema(name, devices)['default'], 'mps')

    def test_mixed_gpu_choices_are_not_reordered(self):
        self.apply()
        devices = ['cuda:1', 'mps', 'cuda:0']
        for name in NAMES:
            self.assertIs(self.schema(name, devices)['options'], devices)
            self.assertEqual(self.schema(name, devices)['default'], 'cuda:1')

    def test_reapplying_does_not_rewrite_files(self):
        self.apply()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.folder.iterdir()}
        self.assertEqual(self.apply(), 0)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.folder.iterdir()}
        self.assertEqual(before, after)

    def test_only_one_assignment_changes_in_each_file(self):
        self.apply()
        for name in NAMES:
            actual = (self.folder/name).read_bytes()
            self.assertEqual(actual, FIXTURE.encode().replace(ORIGINAL, CORRECTED))
            before = ast.parse(FIXTURE).body[0].body[1]
            after = ast.parse(actual).body[0].body[1]
            self.assertEqual(ast.dump(before), ast.dump(after))

    def test_source_drift_in_second_file_does_not_partially_patch_first(self):
        p = self.folder/NAMES[1]
        p.write_text(FIXTURE.replace('get_device_list()', 'new_device_api()'))
        with self.assertRaisesRegex(ValueError, 'Unrecognized SeedVR2'):
            self.apply()
        self.assertEqual((self.folder/NAMES[0]).read_text(), FIXTURE)

    def test_missing_second_file_does_not_partially_patch_first(self):
        (self.folder/NAMES[1]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.apply()
        self.assertEqual((self.folder/NAMES[0]).read_text(), FIXTURE)

    def test_invalid_python_does_not_partially_patch_first(self):
        with (self.folder/NAMES[1]).open('a') as out:
            out.write('\n! invalid python\n')
        with self.assertRaises(SyntaxError):
            self.apply()
        self.assertEqual((self.folder/NAMES[0]).read_text(), FIXTURE)

    def test_duplicate_target_is_rejected(self):
        p = self.folder/NAMES[0]
        p.write_text(FIXTURE.replace(ORIGINAL.decode(), ORIGINAL.decode()*2))
        with self.assertRaisesRegex(ValueError, 'Unrecognized SeedVR2'):
            self.apply()

    def test_partially_prepatched_source_is_supported(self):
        p = self.folder/NAMES[0]
        p.write_bytes(FIXTURE.encode().replace(ORIGINAL, CORRECTED))
        self.assertEqual(self.apply(), 1)
        self.assertEqual(self.schema(NAMES[1], [])['default'], 'cpu')

    def test_symlink_is_rejected(self):
        p = self.folder/NAMES[0]
        p.unlink()
        p.symlink_to(self.folder/NAMES[1])
        with self.assertRaises(FileNotFoundError):
            self.apply()

    def test_build_patches_bundled_copy_without_changing_inherited_source(self):
        home = self.root/'comfy'
        with patch.object(I.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            I.build(home, self.root/'scripts', self.seed, self.root/'constraints.txt')
        self.assertGreaterEqual(run.call_count, 2)
        for name in NAMES:
            self.assertEqual((self.folder/name).read_text(), FIXTURE)
            out = home/'custom_nodes/ComfyUI-SeedVR2_VideoUpscaler/src/interfaces'/name
            self.assertIn(CORRECTED, out.read_bytes())

    def test_all_four_graphs_still_use_real_registry_validation(self):
        text = (HERE/'check.py').read_text()
        self.assertIn('for graph in graphs:validate_schema(graph,registry)', text)
        self.assertIn('if len(graphs)!=4', text)
        self.assertNotIn('skip_seed', text)


if __name__ == '__main__':
    unittest.main()
