import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / 'runtime'))
import assets
import build_workflows as builder
from origin_policy import normalize_origin, public_origins
from helpers import validate_password, terminal_environment
spec = importlib.util.spec_from_file_location('h3_image_helpers', PACKAGE / 'nodes/H3ImageWorkspace/__init__.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class WorkflowTests(unittest.TestCase):
    def test_all_graphs(self):
        for p in builder.PROFILES:
            with self.subTest(profile=p[0]):
                graph, api = builder.build(p)
                self.assertTrue(builder.validate(graph))
                self.assertEqual(api['40']['inputs']['prompt'], '')

    def test_generated_reproducible(self):
        for p in builder.PROFILES:
            saved = json.loads((PACKAGE/'workflows'/(p[0]+'.json')).read_text())
            self.assertEqual(saved, builder.build(p)[0])

    def test_four_role_order(self):
        g, api = builder.build(builder.PROFILES[4])
        self.assertEqual([api['40']['inputs']['reference_%s'%i] for i in range(1,5)], [[str(i),0] for i in range(1,5)])

    def test_generation_has_no_ref(self):
        g, api = builder.build(builder.PROFILES[0])
        self.assertFalse(any(k.startswith('reference_') for k in api['40']['inputs']))

    def test_no_enhancer_or_legacy_decoder(self):
        for p in builder.PROFILES:
            graph, api = builder.build(p)
            self.assertFalse(any('Fizgig' in n['type'] or 'Turbo' in n['type'] for n in graph['nodes']))
            self.assertEqual(api['31']['inputs']['clip_name'], builder.ENCODER)

    def test_sigmas_match_uploaded(self):
        pdd = json.loads((PACKAGE/'pdd_config.json').read_text())
        self.assertEqual(pdd['num_inference_steps'], 6)
        self.assertEqual(pdd['sigmas'], helpers.SIGMAS)
        self.assertEqual(len(pdd['sigmas']), 7)

    def test_missing_reference(self):
        with self.assertRaises(ValueError):
            helpers.select_references({}, 1)

    def test_slots_compact_order(self):
        rows = helpers.select_references({'reference_3': {'image': 'b'}, 'reference_1': {'image': 'a'}}, 2)
        self.assertEqual([x[0] for x in rows], [1,3])

    def test_bad_prompt_reference(self):
        with self.assertRaises(ValueError):
            helpers.validate_prompt('Use <Picture 3>', 2)

    def test_empty_prompt(self):
        with self.assertRaises(ValueError):
            helpers.validate_prompt(' ', 0)

    def test_prompt_not_mutated(self):
        text='  My own {prompt}. <Picture 1>  '
        helpers.validate_prompt(text, 1)
        self.assertEqual(text, '  My own {prompt}. <Picture 1>  ')

    def test_four_mp_square(self):
        self.assertEqual(helpers.dimensions('1:1', 4), (2048,2048))

    def test_source_aspect(self):
        w,h=helpers.dimensions('1:1',4,True,(1920,1080))
        self.assertLess(abs(w/h-1920/1080), .03)
        self.assertEqual(w%32+h%32,0)

    def test_low_resolution_rejected(self):
        with self.assertRaises(ValueError):helpers.dimensions('1:1',1)

    def test_extreme_aspect_rejected(self):
        with self.assertRaises(ValueError):helpers.dimensions('1:1',4,True,(9000,100))


class AssetTests(unittest.TestCase):
    def fake_meta(self, row):
        return {'sha':'a'*40,'siblings':[{'rfilename':row['file'],'lfs':{'sha256':'b'*64,'size':123}}]}

    def test_lock_includes_sha_and_size(self):
        row={'id':'x','repo':'owner/repo','file':'foo.safetensors','destination':'vae/foo.safetensors'}
        result=assets.lock_row(row,self.fake_meta(row))
        self.assertEqual((result['revision'],result['sha256'],result['size_bytes']),('a'*40,'b'*64,123))

    def test_missing_model_fails_closed(self):
        with self.assertRaises(ValueError):assets.lock_row({'repo':'a/b','file':'x'}, {'sha':'a'*40,'siblings':[]})

    def test_missing_hash_rejected(self):
        with self.assertRaises(ValueError):assets.lock_row({'repo':'a/b','file':'x'}, {'sha':'a'*40,'siblings':[{'rfilename':'x','size':12}]})

    def test_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):assets.safe_destination(Path(temp),'../escape')

    def test_verification_does_not_change_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'a';p.write_bytes(b'unchanged')
            row={'id':'unit','size_bytes':9,'sha256':assets.digest(p)}
            self.assertTrue(assets.verify(p,row,root))
            self.assertEqual(p.read_bytes(),b'unchanged')
            self.assertTrue(assets.verify(p,row,root))

    def test_bad_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'a';p.write_bytes(b'x')
            self.assertFalse(assets.verify(p,{'id':'x','size_bytes':1,'sha256':'a'*64},root))

    def test_source_models_no_bf16_base(self):
        sources=assets.read_json(PACKAGE/'sources.json')
        self.assertEqual(len(sources['models']),3)
        self.assertIn('int8',sources['models'][0]['file'])
        self.assertNotIn('prompt_upsampler',json.dumps(sources))


class AccessTests(unittest.TestCase):
    def test_password_rules(self):
        validate_password('a-good-password-123')
        with self.assertRaises(ValueError):validate_password('short')

    def test_proxy_origins(self):
        out=public_origins({'RUNPOD_POD_ID':'abc123'})
        self.assertEqual(out['comfy'],'https://abc123-8188.proxy.runpod.net')

    def test_invalid_public_origin(self):
        with self.assertRaises(ValueError):public_origins({'H3_FILES_PUBLIC_ORIGIN':'http://bad.example'})

    def test_origin_normalization(self):
        self.assertEqual(normalize_origin('https://example.com:443'),('https','example.com',443))

    def test_normal_manager_security(self):
        s=(PACKAGE/'runtime/backend.py').read_text()
        self.assertIn('security_level = normal',s)
        self.assertNotIn('security_level = weak',s)

    def test_no_old_models_download(self):
        s=(PACKAGE/'runtime/backend.py').read_text()
        self.assertNotIn('from bootstrap import',s)
        self.assertIn('from assets import data_root, download',s)


if __name__=='__main__':unittest.main()
