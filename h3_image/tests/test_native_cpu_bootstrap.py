"""Regression tests for CPU-only CI startup; no weights or CUDA are required.

The import-contract fixture reproduces the pinned ComfyUI options/cli_args
behavior. The full real ComfyUI CPU smoke remains a required Docker CI step.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
SMOKE = PACKAGE / 'native_smoke.py'

OPTIONS = '''args_parsing = False

def enable_args_parsing(enable=True):
    global args_parsing
    args_parsing = enable
'''
CLI_ARGS = '''import argparse
import comfy.options
parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--cache-none', action='store_true')
parser.add_argument('--disable-pinned-memory', action='store_true')
if comfy.options.args_parsing:
    args = parser.parse_args()
else:
    args = parser.parse_args([])
'''


class NativeCPUBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.comfy = Path(self.temp.name)
        pkg = self.comfy / 'comfy'
        pkg.mkdir()
        (pkg / '__init__.py').write_text('')
        (pkg / 'options.py').write_text(OPTIONS)
        (pkg / 'cli_args.py').write_text(CLI_ARGS)
        self.prefix = textwrap.dedent('''
            import importlib.util, json, sys
            from pathlib import Path
            spec = importlib.util.spec_from_file_location('native_smoke_under_test', %r)
            smoke = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(smoke)
            smoke.COMFY = Path(%r)
        ''' % (str(SMOKE), str(self.comfy)))

    def run_python(self, body, expected_returncode=0):
        proc = subprocess.run([sys.executable, '-I', '-c', self.prefix + textwrap.dedent(body)],
                              text=True, capture_output=True, timeout=20)
        self.assertEqual(proc.returncode, expected_returncode, proc.stdout + proc.stderr)
        return proc

    def test_importing_test_module_does_not_import_comfy_or_torch(self):
        self.run_python('''
            assert 'comfy.options' not in sys.modules
            assert 'comfy.model_management' not in sys.modules
            assert 'torch' not in sys.modules
        ''')

    def test_reproduces_old_argv_only_failure(self):
        self.run_python('''
            sys.path.insert(0, str(smoke.COMFY))
            sys.argv = ['native-smoke', '--cpu']
            from comfy.cli_args import args
            assert args.cpu is False
        ''')

    def test_new_bootstrap_selects_cpu(self):
        self.run_python('''
            args = smoke.configure_cpu_imports()
            import comfy.options
            assert comfy.options.args_parsing is True
            assert args.cpu is True
        ''')

    def test_avoids_pinned_memory_and_cache(self):
        self.run_python('''
            args = smoke.configure_cpu_imports()
            assert args.disable_pinned_memory is True
            assert args.cache_none is True
        ''')

    def test_foreign_cli_arguments_do_not_leak(self):
        self.run_python('''
            sys.argv = ['test-runner', '--unrelated-test-option', 'value']
            args = smoke.configure_cpu_imports()
            assert args.cpu is True
            assert '--unrelated-test-option' not in sys.argv
        ''')

    def test_comfy_path_precedes_other_search_paths(self):
        self.run_python('''
            smoke.configure_cpu_imports()
            assert sys.path[0] == str(smoke.COMFY)
        ''')

    def test_previously_cached_gpu_defaults_are_rejected(self):
        self.run_python('''
            sys.path.insert(0, str(smoke.COMFY))
            import comfy.cli_args
            assert not comfy.cli_args.args.cpu
            try:
                smoke.configure_cpu_imports()
            except RuntimeError as error:
                assert 'fresh Python process' in str(error)
            else:
                raise AssertionError('Unsafe cached arguments were accepted')
        ''')

    def test_late_bootstrap_is_rejected(self):
        self.run_python('''
            import types
            sys.modules['comfy.model_management'] = types.ModuleType('comfy.model_management')
            try:
                smoke.configure_cpu_imports()
            except RuntimeError as error:
                assert 'before comfy.model_management' in str(error)
            else:
                raise AssertionError('Late bootstrap was accepted')
        ''')

    def test_bootstrap_precedes_native_import_and_device_check(self):
        tree = ast.parse(SMOKE.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        call = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'configure_cpu_imports')
        imports = [n for n in ast.walk(main) if isinstance(n, ast.ImportFrom)
                   and n.module in ('comfy', 'comfy_extras.nodes_minimax_h3')]
        self.assertTrue(imports)
        self.assertTrue(all(call.lineno < n.lineno for n in imports))
        self.assertIn("if device.type != 'cpu':", SMOKE.read_text())

    def test_cpu_server_and_gpu_workspace_remain_separate(self):
        text = SMOKE.read_text()
        self.assertIn("str(COMFY/'main.py'), '--cpu'", text)
        self.assertIn("'gpu_generation_tested': False", text)
        backend = (PACKAGE / 'runtime/backend.py').read_text()
        self.assertNotIn("'--cpu'", backend)
        self.assertNotIn('"--cpu"', backend)


if __name__ == '__main__':
    unittest.main()
