"""CPU-only routing/media/security regression tests. No model weights are loaded."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('h3media_tests',P/'nodes/studio/__init__.py',submodule_search_locations=[str(P/'nodes/studio')])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
r=sys.modules['h3media_tests.media_rules'];v=sys.modules['h3media_tests.video_input'];n=sys.modules['h3media_tests.media_nodes']
from build_media_workflows import build
from media_validate import validate_media_graph


def plan(mode='Auto',photos=0,video=0,preset=None,target=False,protect=False):
    return r.media_plan(mode,photos,video,'2:3',2.5,2.5,preset or r.MEDIA_PRESETS[0],(640,960),target,protect)

class MediaRules(unittest.TestCase):
    def test_auto_no_media_generate(self):self.assertEqual(plan()['mode'],'Generate')
    def test_auto_single_photo_edit(self):self.assertEqual(plan(photos=1)['mode'],'Edit')
    def test_auto_video_always_ref(self):
        for count in (0,1,5,9):self.assertEqual(plan(photos=count,video=1)['mode'],'References')
    def test_no_new_base_precision(self):
        self.assertIn('int8',plan(video=1)['model']);self.assertIn('int8',plan()['model'])
    def test_generate_cannot_ignore_video(self):
        with self.assertRaises(ValueError):plan('Generate',video=1)
    def test_edit_cannot_ignore_video(self):
        with self.assertRaises(ValueError):plan('Edit',photos=1,video=1)
    def test_replace_requires_slot_one(self):
        with self.assertRaises(ValueError):plan('Replace person',photos=2)
    def test_replace_needs_identity(self):
        with self.assertRaises(ValueError):plan('Replace person',photos=1,target=True)
    def test_replace_with_video_identity(self):self.assertEqual(plan('Replace person',photos=1,video=1,target=True)['mode'],'Replace person')
    def test_ref_turbo_sampling(self):
        p=plan(video=1,preset=r.MEDIA_PRESETS[1]);self.assertEqual((p['steps'],p['sampler'],p['adapter_strength']),(8,'euler',1.))
    def test_legacy_fl_turbo_rejected_for_refs(self):
        with self.assertRaises(ValueError):plan(photos=2,preset=r.MEDIA_PRESETS[2])
    def test_ref_turbo_rejected_for_fl(self):
        with self.assertRaises(ValueError):plan(preset=r.MEDIA_PRESETS[1])
    def test_empty_legacy_turbo_strength_prevents_double_apply(self):self.assertEqual(plan(video=1,preset=r.MEDIA_PRESETS[1])['turbo_strength'],0.)
    def test_frame_grid(self):
        for x in range(5,361):
            y=r.aligned_frames(x);self.assertLessEqual(y,x);self.assertEqual(y%17,5)
    def test_too_short_rejected(self):
        with self.assertRaises(ValueError):r.aligned_frames(4)
    def test_video_resize_bounded(self):
        for w,h in [(3840,2160),(2160,3840),(8000,400),(160,96)]:
            a,b=r.video_dimensions(w,h,.5);self.assertEqual(a%32,0);self.assertEqual(b%32,0);self.assertLessEqual(max(a,b),2048)
    def test_video_prompt_tags(self):
        r.validate_media_prompt('Use <Video 1> and <Picture 1>',1,1)
        for text,photos,videos in [('Use <Video 2>',1,1),('Use <Video 1>',1,0),('Use <Picture 2>',1,1),('Use <Audio 1>',1,1)]:
            with self.assertRaises(ValueError):r.validate_media_prompt(text,photos,videos)

class MediaGraphs(unittest.TestCase):
    def test_three_graphs_valid(self):
        for i in range(3):self.assertTrue(validate_media_graph(build(i),i))
    def test_deterministic(self):
        for i in range(3):self.assertEqual(build(i),build(i))
    def test_person_mask_enabled(self):self.assertTrue(next(x for x in build(2)['nodes'] if x['id']==10)['widgets_values_named']['protect_background'])
    def test_video_workflow_defaults_turbo(self):self.assertEqual(next(x for x in build(1)['nodes'] if x['id']==10)['widgets_values_named']['preset'],r.MEDIA_PRESETS[1])
    def test_no_ailab_or_enhancer(self):
        for i in range(3):
            kinds=[x['type'] for x in build(i)['nodes']]
            self.assertFalse(any(x.startswith('AILab') for x in kinds));self.assertNotIn('TextGenerate',kinds)

class MediaNodes(unittest.TestCase):
    def test_disabled_video_no_io(self):self.assertIsNone(v.H3StudioOptionalVideo().load('bad.mp4',False)[0]['frames'])
    def test_empty_video_no_io(self):self.assertIsNone(v.H3StudioOptionalVideo().load('(none)')[0]['frames'])
    def test_video_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{'folder_paths':types.SimpleNamespace(get_input_directory=lambda:folder)}):
            for filename in ('../x.mp4','/tmp/x.mp4','abc.txt'):
                with self.assertRaises(ValueError):v.checked_video_path(filename)
    def test_native_ref_video_receives_full_length(self):
        calls=[]
        def execute(**kwargs):calls.append(kwargs);return types.SimpleNamespace(result=('conditioning','unused video latent'))
        stub=types.SimpleNamespace(MiniMaxH3ImageToVideo=types.SimpleNamespace(execute=execute),MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=execute))
        p=plan(video=1);p.update(images=['photo'],videos=['73 frame tensor'],count=1,conditioning_length=73)
        with patch.dict(sys.modules,{'comfy_extras':types.ModuleType('comfy_extras'),'comfy_extras.nodes_minimax_h3':stub}):
            result=n.H3StudioMediaConditioning().encode(p,'clip','vae','<Video 1> and <Picture 1>')
        self.assertEqual(result,('conditioning',));self.assertEqual(calls[0]['length'],73)
        self.assertEqual(calls[0]['ref_videos'],{'ref_video_0':'73 frame tensor'})
        self.assertNotIn('audio_vae',calls[0])
    def test_no_mask_silently_used(self):
        # Tensor tests run separately with the actual torch install in media_cpu_smoke.
        self.assertEqual(plan('Replace person',photos=2,target=True,protect=True)['protect_background'],True)

class AccessRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # aiohttp is installed after Docker source-only tests. Skip imports here when absent.
        try:
            import aiohttp
        except ImportError:raise unittest.SkipTest('aiohttp not installed in source-only build stage')
        sys.path.insert(0,str(P/'runtime'))
        import access_gateway
        cls.a=access_gateway
    def test_cookie_signature(self):
        value=self.a.issue_cookie('secret');self.assertTrue(self.a.valid_cookie('secret',value))
        self.assertFalse(self.a.valid_cookie('other',value))
    def test_forged_cookie_denied(self):self.assertFalse(self.a.valid_cookie('secret','session.9999999999.fake.forged'))
    def test_login_nonce_not_session(self):self.assertFalse(self.a.valid_cookie('secret',self.a.issue_cookie('secret','login')))
    def test_expired_cookie(self):self.assertFalse(self.a.valid_cookie('secret',self.a.issue_cookie('secret',ttl=-10)))
    def test_password_hash_not_plain(self):
        p='test-password-16-chars';h=self.a.password_hash(p);self.assertNotIn(p,h)
        self.assertTrue(self.a.password_ok(p,h));self.assertFalse(self.a.password_ok('incorrect',h))
    def test_no_cache_headers(self):self.assertIn('no-store',self.a.NO_CACHE['Cache-Control'])
    def test_origin(self):
        good=types.SimpleNamespace(headers={'Origin':'https://example.test'},host='example.test')
        bad=types.SimpleNamespace(headers={'Origin':'https://evil.test'},host='example.test')
        self.assertTrue(self.a.same_origin(good));self.assertFalse(self.a.same_origin(bad))

if __name__=='__main__':unittest.main()
