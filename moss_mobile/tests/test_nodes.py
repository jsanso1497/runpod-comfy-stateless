import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
import json
import hashlib
import warnings
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runtime'))
import nodes
import bootstrap
from launch import validate_password, NGINX
from validate_package import validate
from validate_live import validate as validate_installed

class GeometryTests(unittest.TestCase):
    def test_exact_ratios(self):
        for shape,(a,b) in nodes.ASPECTS.items():
            for detail,mp in nodes.DETAILS.items():
                w,h=nodes.MossMobileCanvas().make(shape,detail)
                self.assertEqual(w*b,h*a)
                self.assertEqual(w%16,0);self.assertEqual(h%16,0)
                self.assertLessEqual(w*h,mp*1024**2)

    def test_pad_not_crop(self):
        image=torch.rand(1,113,171,3)
        padded,w,h,geom=nodes.MossMobilePrepare().prepare(image,upscale_small=False)
        self.assertEqual((w,h),(176,128))
        self.assertTrue(torch.equal(image,padded[:,:113,:171,:]))
        restored,=nodes.MossMobileRestore().restore(padded,geom)
        self.assertTrue(torch.equal(image,restored))

    def test_large_source_restores_dimensions(self):
        image=torch.rand(1,1101,1703,3)
        fitted,w,h,geom=nodes.MossMobilePrepare().prepare(image,1.0)
        self.assertEqual(w%16,0);self.assertEqual(h%16,0)
        self.assertLess(w,1703)
        restored,=nodes.MossMobileRestore().restore(fitted,geom)
        self.assertEqual(restored.shape,image.shape)

    def test_small_reference_uses_training_scale(self):
        image=torch.rand(1,512,512,3)
        fitted,w,h,geom=nodes.MossMobilePrepare().prepare(image)
        self.assertEqual((w,h),(1024,1024))
        restored,=nodes.MossMobileRestore().restore(fitted,geom)
        self.assertEqual(restored.shape,image.shape)

    def test_reject_changed_generated_size(self):
        image=torch.rand(1,113,171,3)
        _,_,_,geom=nodes.MossMobilePrepare().prepare(image)
        with self.assertRaises(ValueError):nodes.MossMobileRestore().restore(image,geom)

    def test_reject_batches(self):
        with self.assertRaises(ValueError):nodes.MossMobilePrepare().prepare(torch.zeros(2,32,32,3))

class MaskTests(unittest.TestCase):
    def test_protected_pixels_exact(self):
        original=torch.rand(1,63,95,3)
        edited=torch.rand_like(original)
        mask=torch.zeros_like(original);mask[:,20:40,30:60]=1
        output,=nodes.MossMobileComposite().composite(original,edited,mask)
        self.assertTrue(torch.equal(output[mask==0],original[mask==0]))
        self.assertTrue(torch.equal(output[mask==1],edited[mask==1]))

    def test_grey_blends(self):
        original=torch.zeros(1,32,32,3);edited=torch.ones_like(original);mask=edited*0.25
        output,=nodes.MossMobileComposite().composite(original,edited,mask)
        self.assertTrue(torch.equal(output,mask))

    def test_mask_checked_before_sampling(self):
        image=torch.ones(1,32,32,3)
        result,=nodes.MossMobileMaskGuard().check(image,image)
        self.assertTrue(torch.equal(result,image))
        with self.assertRaises(ValueError):nodes.MossMobileMaskGuard().check(image,torch.zeros_like(image))

    def test_black_fails(self):
        image=torch.zeros(1,32,32,3)
        with self.assertRaises(ValueError):nodes.MossMobileComposite().composite(image,image,image)

    def test_wrong_size_fails(self):
        image=torch.ones(1,32,32,3)
        with self.assertRaises(ValueError):nodes.MossMobileComposite().composite(image,image,torch.ones(1,33,32,3))

class PromptTests(unittest.TestCase):
    def test_off_skips_expansion(self):
        choice=nodes.MossMobilePromptChoice()
        self.assertEqual(choice.check_lazy_status(False,'exact original',None),[])
        self.assertEqual(choice.choose(False,'exact original',None),('exact original',))

    def test_on_requests_expansion(self):
        self.assertEqual(nodes.MossMobilePromptChoice().check_lazy_status(True,'prompt',None),['expanded'])

    def test_empty_falls_back(self):
        with warnings.catch_warnings(record=True) as caught:
            self.assertEqual(nodes.MossMobilePromptChoice().choose(True,'original',''),('original',))
            self.assertTrue(caught)

    def test_quoted_text_guard(self):
        with warnings.catch_warnings(record=True):
            self.assertEqual(nodes.MossMobilePromptChoice().choose(True,'A sign reading "MOSS"','A sign reading moss.'),('A sign reading "MOSS"',))

    def test_empty_prompt_fails(self):
        with self.assertRaises(ValueError):nodes.MossMobilePromptChoice().choose(False,' ')

class PackagingTests(unittest.TestCase):
    def test_graphs(self):self.assertEqual(validate(),4)

    def test_password(self):
        validate_password('a-unique-long-password')
        for value in ('','short','a'*73,'a'*20+'\n','{{ RUNPOD_SECRET_comfy_password }}'):
            with self.assertRaises(ValueError):validate_password(value)

    def test_auth_gateway(self):
        self.assertIn('auth_basic_user_file',NGINX)
        self.assertIn('127.0.0.1:8189',NGINX)
        self.assertIn('user www-data;',NGINX)
        self.assertNotIn('auth_basic off',NGINX)

    def test_hash_marker_and_changed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model';path.write_bytes(b'test')
            row={'id':'test', 'sha256':hashlib.sha256(b'test').hexdigest(),'revision':'a'*40}
            marker=Path(folder)/'ok.json'
            self.assertTrue(bootstrap.verified(path,row,marker))
            self.assertTrue(bootstrap.verified(path,row,marker))
            path.write_bytes(b'bad')
            self.assertFalse(bootstrap.verified(path,row,marker))

    def test_cached_model_symlink_stays_inside_model_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"models"
            source=root/".sources"/"example"/"model.safetensors"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"model")
            dest=root/"diffusion_models"/"model.safetensors"
            dest.parent.mkdir()
            dest.symlink_to(source)
            self.assertEqual(bootstrap.safe_child(root,"diffusion_models/model.safetensors"),dest)

    def test_live_checker_rejects_missing_nodes(self):
        self.assertTrue(validate_installed({}))

    def test_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for bad in ('/outside','../outside'):
                with self.assertRaises(ValueError):bootstrap.safe_child(root,bad)
            self.assertEqual(bootstrap.safe_child(root,'vae/example'),root/'vae/example')

if __name__=='__main__':unittest.main()
