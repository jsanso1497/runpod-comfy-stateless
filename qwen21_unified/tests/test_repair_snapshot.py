"""No-Torch regression checks for conservative unified snapshot registration."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
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

    def test_action_bootstraps_missing_generated_dockerfile(self):
        # Reproduces the uploaded October 8 GitHub repair failure exactly.
        generated = self.root / "qwen21_photo_edit/Dockerfile.unified"
        generated.unlink()
        self.assertFalse(generated.exists())
        manifest_original = self.snapshot.read_bytes()
        generate = self.root / "qwen21_unified/scripts/generate_dockerfile.py"
        subprocess.run([sys.executable, str(generate), "--repo", str(self.root)], check=True)
        self.assertTrue(generated.is_file())
        # Dry-run should not modify the snapshot or any unrelated sources.
        repair.reconcile(self.root, write=False)
        self.assertEqual(self.snapshot.read_bytes(), manifest_original)
        repair.reconcile(self.root, write=True)
        updated = json.loads(self.snapshot.read_text())["files"]
        self.assertEqual(updated["qwen21_photo_edit/Dockerfile.unified"], repair.digest(generated))
        self.assertTrue(generated.read_text().startswith("FROM pytorch/"))

    def test_repair_workflow_generates_before_audit_and_stages_dockerfile(self):
        action = (Path(__file__).resolve().parents[2]
                  / ".github/workflows/repair-qwen21-unified-snapshot.yml").read_text()
        self.assertLess(action.index("Generate missing Photo-derived unified Dockerfile safely"),
                        action.index("Audit unified package"))
        self.assertIn("generate_dockerfile.py --repo . --check", action)
        self.assertIn("generate_dockerfile.py --repo .\n", action)
        self.assertIn("git add -- SOURCE_SNAPSHOT.json qwen21_photo_edit/Dockerfile.unified", action)
        self.assertIn("SOURCE_SNAPSHOT.json|qwen21_photo_edit/Dockerfile.unified", action)
        self.assertIn("python qwen21_native/addons/torso_lock/scripts/validate_workflows.py", action)

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

    def test_unrecognized_top_level_key_and_nested_digest_map(self):
        old = json.loads(self.snapshot.read_text())["files"]
        self.snapshot.write_text(json.dumps({"metadata": {"version": 2},
              "source_data": {"tracked_files": {k: {"digest": v,
                         "size": (self.root / k).stat().st_size} for k, v in old.items()},
                         "count": len(old)}}))
        self._set_verifier_for("nested_dict")
        result = repair.reconcile(self.root, write=True)
        self.assertEqual(result["snapshot_mapping"], "source_data.tracked_files")
        data = json.loads(self.snapshot.read_text())
        records = data["source_data"]["tracked_files"]
        self.assertEqual(data["source_data"]["count"], len(records))
        self.assertEqual(records["existing_00.txt"]["digest"], old["existing_00.txt"])
        self.assertIn("qwen21_photo_edit/Dockerfile.unified", records)

    def test_nested_list_uses_nonstandard_path_and_hash_keys(self):
        old = json.loads(self.snapshot.read_text())["files"]
        self.snapshot.write_text(json.dumps({"schema_version": 4,
                   "snapshot": {"paths": [{"file": k, "hash": v, "bytes": (self.root/k).stat().st_size}
                                           for k, v in old.items()], "file_count": len(old)}}))
        self._set_verifier_for("nested_list")
        result = repair.reconcile(self.root, write=True)
        self.assertEqual(result["snapshot_mapping"], "snapshot.paths")
        data = json.loads(self.snapshot.read_text())
        records = data["snapshot"]["paths"]
        self.assertEqual(data["snapshot"]["file_count"], len(records))
        self.assertIn("qwen21_photo_edit/Dockerfile.unified", [r["file"] for r in records])

    def test_unknown_root_mapping_and_prefixed_sha(self):
        old = json.loads(self.snapshot.read_text())["files"]
        self.snapshot.write_text(json.dumps({k: "sha256:" + v for k, v in old.items()}))
        self._set_verifier_for("root_dict")
        result = repair.reconcile(self.root, write=True)
        self.assertEqual(result["snapshot_mapping"], "(root)")
        data = json.loads(self.snapshot.read_text())
        self.assertTrue(data["qwen21_photo_edit/Dockerfile.unified"].startswith("sha256:"))

    def test_unrecognizable_schema_fails_without_modifications(self):
        self.snapshot.write_text(json.dumps({"files": ["foo", "bar"], "version": 1}))
        original = self.snapshot.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Could not identify a verified path/SHA256 inventory"):
            repair.reconcile(self.root, write=True)
        self.assertEqual(self.snapshot.read_bytes(), original)

    def test_preexisting_unrelated_hash_mismatch_is_not_overwritten(self):
        records = json.loads(self.snapshot.read_text())
        records["files"]["existing_00.txt"] = "0"*64
        self.snapshot.write_text(json.dumps(records))
        before = self.snapshot.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "verifier rejected"):
            repair.reconcile(self.root, write=True)
        self.assertEqual(self.snapshot.read_bytes(), before)

    def _set_verifier_for(self, mode):
        script = """import hashlib,json,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1]
j=json.loads((r/'SOURCE_SNAPSHOT.json').read_text())
mode=MODE
if mode=='nested_dict':
 m={k:v['digest'] for k,v in j['source_data']['tracked_files'].items()}
elif mode=='nested_list':
 m={row['file']:row['hash'] for row in j['snapshot']['paths']}
else:
 m={k:v.removeprefix('sha256:') for k,v in j.items()}
for rel,sha in m.items():
 p=r/rel
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:sys.exit(3)
for rel in ('PACKAGE_MANIFEST.json','qwen21_photo_edit/Dockerfile.unified',
 'qwen21_unified/scripts/repair_source_snapshot.py',
 'qwen21_unified/tests/test_repair_snapshot.py'):
 if rel not in m:sys.exit(4)
"""
        (self.root / "tools/verify_snapshot.py").write_text(script.replace("MODE", repr(mode)))


if __name__ == "__main__":
    unittest.main()
