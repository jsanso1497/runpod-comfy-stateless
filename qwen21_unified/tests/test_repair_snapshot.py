"""No-Torch regression checks for conservative unified snapshot registration."""
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

REPAIR = Path(__file__).resolve().parents[1] / "scripts/repair_source_snapshot.py"
spec = importlib.util.spec_from_file_location("snapshot_repair", REPAIR)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

DOCKER = '''FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
COPY qwen21_photo_edit/ /opt/qwen21_photo_edit/
RUN python -m pip install --constraint /tmp/pytorch-constraints.txt -r /opt/ComfyUI/requirements.txt
WORKDIR /opt/ComfyUI
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/qwen21_photo_edit/start.sh"]
'''
MOCK_VERIFIER = '''import hashlib,json,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1]
m=json.loads((r/'SOURCE_SNAPSHOT.json').read_text())['files']
for path in (r/'PACKAGE_MANIFEST.json',r/'qwen21_photo_edit/Dockerfile.unified'):
 if path.relative_to(r).as_posix() not in m:sys.exit(2)
for path in sorted((r/'qwen21_unified').rglob('*.py')):
 if path.relative_to(r).as_posix() not in m:sys.exit(3)
for path in sorted((r/'qwen21_native').rglob('*.json')):
 if path.relative_to(r).as_posix() not in m:sys.exit(4)
for name,digest in m.items():
 path=r/name
 if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:sys.exit(5)
'''


class SnapshotRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        src = REPAIR.parent
        target = self.root / "qwen21_unified/scripts"
        target.mkdir(parents=True)
        shutil.copy2(src / "repair_source_snapshot.py", target / "repair_source_snapshot.py")
        shutil.copy2(Path(__file__).resolve().parents[1] / "scripts/generate_dockerfile.py", target / "generate_dockerfile.py")
        tests = self.root / "qwen21_unified/tests"
        tests.mkdir(parents=True)
        shutil.copy2(Path(__file__).resolve(), tests / "test_repair_snapshot.py")
        act = self.root / ".github/workflows/repair-qwen21-unified-snapshot.yml"
        act.parent.mkdir(parents=True)
        shutil.copy2(Path(__file__).resolve().parents[2] / ".github/workflows/repair-qwen21-unified-snapshot.yml", act)
        photo = self.root / "qwen21_photo_edit"
        photo.mkdir()
        (photo / "Dockerfile").write_text(DOCKER)
        generator = target / "generate_dockerfile.py"
        spec = importlib.util.spec_from_file_location("make_unified_dockerfile", generator)
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        (photo / "Dockerfile.unified").write_text(gen.generate(DOCKER))
        manifest_files = {}
        for i in range(111):
            rel = f"qwen21_native/workflows/test_{i:03}.json"
            file = self.root / rel
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('{"id":' + str(i) + '}')
            manifest_files[rel] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}
        generator_rel = "qwen21_unified/scripts/generate_dockerfile.py"
        generator_file = self.root / generator_rel
        manifest_files[generator_rel] = {"sha256": hashlib.sha256(generator_file.read_bytes()).hexdigest(), "bytes": generator_file.stat().st_size}
        package = self.root / "PACKAGE_MANIFEST.json"
        package.write_text(json.dumps({"release": "test", "files": manifest_files}))
        self.old_digests = {}
        for i in range(12):
            rel = f"existing_{i:02}.txt"
            file = self.root / rel
            file.write_text(str(i))
            self.old_digests[rel] = hashlib.sha256(file.read_bytes()).hexdigest()
        self.snapshot = self.root / "SOURCE_SNAPSHOT.json"
        self.snapshot.write_text(json.dumps({"files": self.old_digests.copy(), "count": 12}))
        verifier = self.root / "tools/verify_snapshot.py"
        verifier.parent.mkdir()
        verifier.write_text(MOCK_VERIFIER)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_is_read_only(self):
        before = self.snapshot.read_bytes()
        result = repair.reconcile(self.root, write=False)
        self.assertGreater(len(result["added"]), 100)
        self.assertEqual(before, self.snapshot.read_bytes())

    def test_write_registers_only_expected_paths(self):
        result = repair.reconcile(self.root, write=True)
        data = json.loads(self.snapshot.read_text())
        self.assertEqual(result["previous_entries"], 12)
        self.assertEqual(data["count"], len(data["files"]))
        for name, digest in self.old_digests.items():
            self.assertEqual(data["files"][name], digest)
        self.assertIn("qwen21_unified/scripts/repair_source_snapshot.py", data["files"])
        self.assertIn("qwen21_photo_edit/Dockerfile.unified", data["files"])
        self.assertNotIn("qwen21_photo_edit/Dockerfile", data["files"])
        before = self.snapshot.read_bytes()
        result2 = repair.reconcile(self.root, write=True)
        self.assertEqual(result2["added"], [])
        self.assertEqual(result2["refreshed_release_files"], [])
        self.assertEqual(before, self.snapshot.read_bytes())

    def test_preserves_list_snapshot_schema(self):
        data = {"files": [{"path": k, "sha256": v} for k, v in self.old_digests.items()], "file_count": 12}
        self.snapshot.write_text(json.dumps(data))
        repair.reconcile(self.root, write=True, run_verifier=False)
        result = json.loads(self.snapshot.read_text())
        self.assertEqual(result["file_count"], len(result["files"]))
        self.assertEqual({r["path"]: r["sha256"] for r in result["files"] if r["path"] in self.old_digests}, self.old_digests)

    def test_tampered_release_is_not_registered(self):
        (self.root / "qwen21_native/workflows/test_000.json").write_text('{"tampered":true}')
        original = self.snapshot.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "differs from reviewed package"):
            repair.reconcile(self.root, write=True)
        self.assertEqual(original, self.snapshot.read_bytes())

    def test_unrelated_extra_source_is_not_registered(self):
        (self.root / "qwen21_native/workflows/arbitrary.json").write_text('{}')
        original = self.snapshot.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "verifier rejected"):
            repair.reconcile(self.root, write=True)
        self.assertEqual(original, self.snapshot.read_bytes())

    def test_verifier_failure_restores_original_exact_bytes(self):
        (self.root / "tools/verify_snapshot.py").write_text('raise RuntimeError("forced")\n')
        original = self.snapshot.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "verifier rejected"):
            repair.reconcile(self.root, write=True)
        self.assertEqual(original, self.snapshot.read_bytes())


if __name__ == "__main__":
    unittest.main()
