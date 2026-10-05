"""Regression checks for missing/old/misplaced files and editable LoRA links."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import verify_snapshot as vs


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.write('h3_portrait/node/ollama_client.py', 'print("split models")\n')
        self.write('.github/workflows/build-h3-portrait.yml', 'name: Portrait\n')
        self.write('config/lora_links.txt', '# Add links here\n')
        self.manifest = {
            'snapshot': vs.SNAPSHOT,
            'editable_files': sorted(vs.EDITABLE),
            'files': {n: None if n in vs.EDITABLE else vs.normalized_sha256(self.root / n)
                      for n in vs.inventory(self.root)},
        }
        self.save_manifest()

    def write(self, name, content):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode('utf-8'))

    def save_manifest(self):
        self.write(vs.MANIFEST, json.dumps(self.manifest))

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return vs.verify(self.root)

    def test_complete_snapshot(self): self.assertEqual(self.check(), 3)
    def test_lora_links_editable(self):
        self.write('config/lora_links.txt', 'https://civitai.com/models/123\n')
        self.assertEqual(self.check(), 3)
    def test_empty_lora_list_valid(self):
        self.write('config/lora_links.txt', '')
        self.assertEqual(self.check(), 3)
    def test_missing_source_rejected(self):
        (self.root / 'h3_portrait/node/ollama_client.py').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'): self.check()
    def test_old_source_rejected(self):
        self.write('h3_portrait/node/ollama_client.py', 'print("old single pass")\n')
        with self.assertRaisesRegex(ValueError, 'Changed/truncated'): self.check()
    def test_truncated_source_rejected(self):
        self.write('h3_portrait/node/ollama_client.py', 's =')
        with self.assertRaisesRegex(ValueError, 'Changed/truncated'): self.check()
    def test_nested_upload_rejected(self):
        self.write('h3_portrait/h3_portrait/node/ollama_client.py', 'pass\n')
        with self.assertRaisesRegex(ValueError, 'Unexpected files'): self.check()
    def test_root_duplicate_rejected(self):
        self.write('test_links.py', 'pass\n')
        with self.assertRaisesRegex(ValueError, 'Unexpected files'): self.check()
    def test_old_workflow_rejected(self):
        self.write('.github/workflows/old-portrait.yml', 'name: Old\n')
        with self.assertRaisesRegex(ValueError, 'Unexpected files'): self.check()
    def test_missing_hidden_action_rejected(self):
        (self.root / '.github/workflows/build-h3-portrait.yml').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'): self.check()
    def test_windows_line_endings_supported(self):
        self.write('h3_portrait/node/ollama_client.py', 'print("split models")\r\n')
        self.assertEqual(self.check(), 3)
    def test_utf8_bom_supported(self):
        self.write('h3_portrait/node/ollama_client.py', '\ufeffprint("split models")\n')
        self.assertEqual(self.check(), 3)
    def test_git_directory_ignored(self):
        self.write('.git/HEAD', 'ref: refs/heads/main\n')
        self.assertEqual(self.check(), 3)
    def test_python_cache_ignored(self):
        self.write('h3_portrait/node/__pycache__/client.pyc', 'cache')
        self.assertEqual(self.check(), 3)
    def test_missing_manifest_rejected(self):
        (self.root / vs.MANIFEST).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing SOURCE'): self.check()
    def test_mixed_manifest_version_rejected(self):
        self.manifest['snapshot'] = 'older'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Mixed snapshot'): self.check()
    def test_path_traversal_rejected(self):
        self.manifest['files']['../outside.py'] = 'a' * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Unsafe'): self.check()
    def test_source_symlink_rejected(self):
        p = self.root / 'h3_portrait/node/ollama_client.py'
        p.unlink()
        p.symlink_to(self.root / 'config/lora_links.txt')
        with self.assertRaisesRegex(ValueError, 'symlink'): self.check()
    def test_editable_file_still_required(self):
        (self.root / 'config/lora_links.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing'): self.check()
    def test_cannot_exempt_arbitrary_code(self):
        self.manifest['editable_files'].append('h3_portrait/node/ollama_client.py')
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'editable-file'): self.check()
    def test_cannot_hash_user_lora_links(self):
        self.manifest['files']['config/lora_links.txt'] = 'a' * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'remain editable'): self.check()


if __name__ == '__main__':
    unittest.main()
