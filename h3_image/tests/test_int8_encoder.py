"""Regression checks for the INT8 default and non-destructive workflow installation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(PACKAGE/'runtime'))
import assets
import build_workflows as builder

NAME='qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
OLD='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
FOLDER='00_H3_Image_INT8_Encoder_v3'

class INT8DefaultTests(unittest.TestCase):
    def source(self):
        return json.loads((PACKAGE/'sources.json').read_text())

    def test_exactly_one_required_encoder(self):
        enc=[r for r in self.source()['models'] if r['destination'].startswith('text_encoders/')]
        self.assertEqual(len(enc),1)
        self.assertEqual(enc[0]['id'],'h3_encoder_int8')
        self.assertEqual(enc[0]['file'],'text_encoders/'+NAME)
        self.assertEqual(enc[0]['destination'],'text_encoders/'+NAME)
        self.assertTrue(enc[0]['reuse'].endswith('/'+NAME))
        self.assertNotIn(OLD,json.dumps(self.source()))

    def test_published_fingerprint_present(self):
        r=self.source()['models'][1]
        self.assertEqual(r['expected_size_bytes'],27141342152)
        self.assertEqual(r['expected_sha256'],'bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923')

    def metadata(self):
        r=self.source()['models'][1]
        return r,{'sha':'a'*40,'siblings':[{'rfilename':r['file'], 'lfs':{
            'size':r['expected_size_bytes'],'sha256':r['expected_sha256']}}]}

    def test_matching_source_can_lock(self):
        r,m=self.metadata();locked=assets.lock_row(r,m)
        self.assertEqual(locked['revision'],'a'*40)
        self.assertEqual(locked['sha256'],r['expected_sha256'])

    def test_changed_encoder_hash_rejected(self):
        r,m=self.metadata();m['siblings'][0]['lfs']['sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'checksum changed'):assets.lock_row(r,m)

    def test_changed_encoder_size_rejected(self):
        r,m=self.metadata();m['siblings'][0]['lfs']['size']+=1
        with self.assertRaisesRegex(ValueError,'size changed'):assets.lock_row(r,m)

    def test_other_models_and_core_not_changed(self):
        s=self.source()
        self.assertEqual(s['comfy_commit'],'ede4e5fe8d15445c6fcd6828d70b202c22f9502c')
        self.assertEqual(len(s['models']),3)
        self.assertEqual(s['models'][0]['file'],'diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors')
        self.assertEqual(s['models'][2]['file'],'vae/minimax_h3_image_vae_fp16.safetensors')

    def test_every_shipped_gui_uses_int8(self):
        count=0
        for p in (PACKAGE/'workflows').glob('*.json'):
            g=json.loads(p.read_text())
            for n in g['nodes']:
                if n['type']=='CLIPLoader':
                    self.assertEqual(n['widgets_values'][0],NAME,p.name)
                    self.assertEqual(n['widgets_values_named']['clip_name'],NAME,p.name)
                    self.assertEqual(n['widgets_values_named']['type'],'minimax')
                    count+=1
        self.assertEqual(count,15)

    def test_every_api_template_uses_int8(self):
        count=0
        for p in (PACKAGE/'api_templates').glob('*.json'):
            for n in json.loads(p.read_text()).values():
                if n['class_type']=='CLIPLoader':
                    self.assertEqual(n['inputs']['clip_name'],NAME,p.name);count+=1
        self.assertEqual(count,15)

    def test_builder_default(self):self.assertEqual(builder.ENCODER,NAME)

    def test_all_prompts_remain_blank(self):
        for p in (PACKAGE/'workflows').glob('*.json'):
            for n in json.loads(p.read_text())['nodes']:
                if n['type']=='H3ImageManualConditioning':
                    self.assertEqual(n['widgets_values_named']['prompt'],'')

    def test_existing_control_protections_unchanged(self):
        s=(PACKAGE/'runtime/backend.py').read_text()
        for t in ['--cache-none','--disable-pinned-memory','--enable-manager','fcntl.LOCK_NB']:
            self.assertIn(t,s)

    def test_native_workflow_folder_is_new(self):
        self.assertIn(FOLDER,(PACKAGE/'runtime/backend.py').read_text())

    def test_prepare_preserves_old_and_saved_workflows(self):
        spec=importlib.util.spec_from_file_location('int8_backend_test',PACKAGE/'runtime/backend.py')
        backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
        # prepare uses only safe_dump from YAML. Use a stand-in so this also runs in
        # the Docker test stage BEFORE third-party packages are installed.
        fake=types.ModuleType('yaml');fake.safe_dump=lambda value:json.dumps(value)
        with tempfile.TemporaryDirectory() as t, patch.dict(sys.modules,{'yaml':fake}):
            root=Path(t)
            old=root/'user/default/workflows/00_H3_Image_Manual_v2/personal.json'
            old.parent.mkdir(parents=True);old.write_text('personal settings')
            manager=root/'user/__manager/config.ini';manager.parent.mkdir();manager.write_text('custom manager settings')
            backend.prepare(root)
            new=root/'user/default/workflows'/FOLDER
            self.assertEqual(len(list(new.glob('*.json'))),15)
            self.assertEqual(old.read_text(),'personal settings')
            self.assertEqual(manager.read_text(),'custom manager settings')
            target=new/'00_Generate_No_References.json';target.write_text('user-edited v3 settings')
            backend.prepare(root)
            self.assertEqual(target.read_text(),'user-edited v3 settings')

    def test_build_summary_points_to_correct_folder(self):
        p=PACKAGE.parent/'.github/workflows/build-h3-image.yml'
        if not p.exists():self.skipTest('GitHub action file is outside isolated Docker build context')
        self.assertIn(FOLDER,p.read_text())
        self.assertIn('INT8/ConvRot',p.read_text())

if __name__=='__main__':unittest.main()
