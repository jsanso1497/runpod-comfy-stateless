import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('unified_apply',ROOT/'APPLY_QWEN_UNIFIED.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

PHOTO_DOCKERFILE='''FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
COPY qwen21_photo_edit/ /opt/qwen21_photo_edit/
RUN python /opt/qwen21_photo_edit/pin_torch.py /tmp/pytorch-constraints.txt
RUN python -m pip install --constraint /tmp/pytorch-constraints.txt -r /opt/ComfyUI/requirements.txt
WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/qwen21_photo_edit/start.sh"]
'''

class TestApply(unittest.TestCase):
    def test_reconcile_manifest_preserves_earlier_content_digest(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)
            old=repo/'README.md';old.write_text('original source')
            incoming=repo/'qwen21_native/new_workflow.json';incoming.parent.mkdir()
            incoming.write_text('{}')
            snapshot=repo/'SOURCE_SNAPSHOT.json'
            original_hash=hashlib.sha256(old.read_bytes()).hexdigest()
            snapshot.write_text(json.dumps({'version':'test','files':{'README.md':original_hash}}))
            # Snapshot fixture needs 10+ records to meet guarded real-repository schema.
            manifest=json.loads(snapshot.read_text())
            for i in range(12):
                name=f'old_{i}.txt';(repo/name).write_text('x')
                manifest['files'][name]=hashlib.sha256(b'x').hexdigest()
            snapshot.write_text(json.dumps(manifest))
            mod.update_snapshot(repo,{'qwen21_native/new_workflow.json'})
            after=json.loads(snapshot.read_text())['files']
            self.assertEqual(after['README.md'],original_hash)
            self.assertEqual(after['qwen21_native/new_workflow.json'],hashlib.sha256(b'{}').hexdigest())
            self.assertEqual(len(after),14)
    def test_conflict_refuses_to_overwrite_photo(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);p=repo/'qwen21_photo_edit';p.mkdir()
            (p/'Dockerfile').write_text(PHOTO_DOCKERFILE)
            (p/'start.sh').write_text('#!/usr/bin/env bash\n')
            (p/'ci_validate.py').write_text('print("ok")')
            # A different native file is not silently replaced.
            existing=repo/'qwen21_native/README.md';existing.parent.mkdir();existing.write_text('custom version')
            with self.assertRaises(RuntimeError):mod.apply(repo,skip_snapshot=True)
            self.assertEqual(existing.read_text(),'custom version')
    def test_explicit_force_rehash_only_allows_expected_native_paths(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)
            manifest=repo/'SOURCE_SNAPSHOT.json'
            items={}
            for i in range(12):
                rel=f'old_{i}.txt';(repo/rel).write_text('a');items[rel]=hashlib.sha256(b'a').hexdigest()
            native=repo/'qwen21_native/item.json';native.parent.mkdir();native.write_text('{}')
            items['qwen21_native/item.json']=hashlib.sha256(b'old').hexdigest()
            manifest.write_text(json.dumps({'files':items}))
            with self.assertRaises(RuntimeError):mod.update_snapshot(repo,{'qwen21_native/item.json'})
            mod.update_snapshot(repo,{'qwen21_native/item.json'},allowed_rehash={'qwen21_native/item.json'})
            changed=json.loads(manifest.read_text())['files']
            self.assertEqual(changed['qwen21_native/item.json'],hashlib.sha256(b'{}').hexdigest())
            self.assertEqual(changed['old_0.txt'],hashlib.sha256(b'a').hexdigest())

    def test_snapshot_error_rolls_back_all_copied_sources(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);p=repo/'qwen21_photo_edit';p.mkdir()
            (p/'Dockerfile').write_text(PHOTO_DOCKERFILE)
            (p/'start.sh').write_text('#!/usr/bin/env bash\n')
            (p/'ci_validate.py').write_text('print("ok")')
            # Wrongly structured snapshot forces an error after copy starts.
            (repo/'SOURCE_SNAPSHOT.json').write_text('{"bad":"schema"}')
            with self.assertRaises(RuntimeError):mod.apply(repo)
            self.assertFalse((repo/'qwen21_native/workflows').exists())
            self.assertFalse((p/'Dockerfile.unified').exists())
            self.assertEqual((repo/'SOURCE_SNAPSHOT.json').read_text(),'{"bad":"schema"}')

    def test_generate_and_merge_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);p=repo/'qwen21_photo_edit';p.mkdir()
            (p/'Dockerfile').write_text(PHOTO_DOCKERFILE)
            (p/'start.sh').write_text('#!/usr/bin/env bash\n')
            (p/'ci_validate.py').write_text('print("ok")')
            mod.apply(repo,skip_snapshot=True)
            self.assertTrue((repo/'.github/workflows/build-qwen21-unified.yml').is_file())
            self.assertTrue((repo/'qwen21_native/addons/torso_lock/INSTALL.py').is_file())
            self.assertTrue((repo/'qwen21_photo_edit/Dockerfile.unified').is_file())
            first=(p/'Dockerfile.unified').read_bytes()
            mod.apply(repo,skip_snapshot=True)
            self.assertEqual((p/'Dockerfile.unified').read_bytes(),first)
            self.assertEqual((p/'Dockerfile').read_text(),PHOTO_DOCKERFILE)

if __name__=='__main__':unittest.main()

class TestPhotoWorkflowInstall(unittest.TestCase):
    def test_both_photo_variants_copied_and_user_edits_preserved(self):
        import sys
        sys.path.insert(0,str(ROOT/'qwen21_unified/scripts'))
        import runtime_prepare
        old_photo=runtime_prepare.PHOTO
        try:
            with tempfile.TemporaryDirectory() as t:
                photo=Path(t)/'Photo'
                wf=photo/'workflows';wf.mkdir(parents=True)
                for name in runtime_prepare.PHOTO_WORKFLOWS:
                    (wf/name).write_text('{"nodes":[]}')
                runtime_prepare.PHOTO=photo
                user=Path(t)/'user'
                self.assertEqual(runtime_prepare.install_photo_workflows(user),2)
                base=user/'default/workflows/Qwen21 Photo 2K SAM3'
                self.assertEqual(len(list(base.glob('*.json'))),2)
                first=base/runtime_prepare.PHOTO_WORKFLOWS[0]
                first.write_text('{"modified":true}')
                self.assertEqual(runtime_prepare.install_photo_workflows(user),2)
                self.assertEqual(first.read_text(),'{"modified":true}')
        finally:runtime_prepare.PHOTO=old_photo

class TestBundleAudit(unittest.TestCase):
    def test_photo_source_required_even_with_native_and_torso_present(self):
        import sys
        sys.path.insert(0,str(ROOT/'qwen21_unified/scripts'))
        from verify_bundle import check
        with tempfile.TemporaryDirectory() as t:
            repo=Path(t)
            import shutil
            # Symlinks avoid expensive copies; verifier reads actual reviewed workflow source.
            (repo/'qwen21_native').symlink_to(ROOT/'qwen21_native',target_is_directory=True)
            (repo/'qwen21_unified').symlink_to(ROOT/'qwen21_unified',target_is_directory=True)
            ph=repo/'qwen21_photo_edit';ph.mkdir()
            (ph/'Dockerfile.unified').write_text('FROM something')
            (ph/'workflows').mkdir()
            with self.assertRaises(RuntimeError):check(repo)
            for name in ('Qwen21_Photo_2K_SAM3_BF16.json','Qwen21_Photo_2K_SAM3_INT8.json'):
                (ph/'workflows'/name).write_text('{}')
            result=check(repo)
            self.assertEqual(result['default_accessible_workflows'],40)
