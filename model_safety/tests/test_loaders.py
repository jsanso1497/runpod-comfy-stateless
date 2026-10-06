import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location('safe_loader_tests', HERE/'node/__init__.py', submodule_search_locations=[str(HERE/'node')])
mod = importlib.util.module_from_spec(sp); sys.modules[sp.name] = mod; sp.loader.exec_module(mod)
policy = sys.modules[sp.name+'.policy']


class LoaderTests(unittest.TestCase):
    def test_schema_matches_native_widget_order(self):
        for name, cls in mod.NODE_CLASS_MAPPINGS.items():
            schema = cls.INPUT_TYPES()
            fields = list(schema['required']) + list(schema.get('optional', {}))
            self.assertEqual(fields, list(policy.NATIVE_FIELDS[policy.SPECS[name][0]]))

    def test_accepts_every_shipped_model(self):
        for name,(native,field,category,allowed,kind) in policy.SPECS.items():
            for file in allowed:
                args={field:file}
                if kind: args['type']=kind
                self.assertEqual(policy.check_selection(name,args),(category,file))

    def test_wrong_h3_encoder_is_rejected_for_krea(self):
        with self.assertRaisesRegex(ValueError,'not interchangeable'):
            policy.check_selection('SafeKreaCLIPLoader',{'clip_name':policy.H3_ENCODERS[0],'type':'lumina2'})

    def test_krea_requires_12_layer_loader_type(self):
        with self.assertRaisesRegex(ValueError,'type=krea2'):
            policy.check_selection('SafeKreaCLIPLoader',{'clip_name':policy.KREA_ENCODER,'type':'lumina2'})

    def test_h3_audio_rejects_qwen_image_vae(self):
        with self.assertRaises(ValueError):
            policy.check_selection('SafeH3AudioVAELoader',{'vae_name':policy.KREA_VAE})

    def test_h3_video_rejects_audio_vae(self):
        with self.assertRaises(ValueError):
            policy.check_selection('SafeH3VideoVAELoader',{'vae_name':policy.H3_AUDIO_VAE})

    def test_h3_encoder_requires_minimax(self):
        with self.assertRaisesRegex(ValueError,'type=minimax'):
            policy.check_selection('SafeH3CLIPLoader',{'clip_name':policy.H3_ENCODERS[0],'type':'lumina2'})

    def test_no_precision_downgrade(self):
        with self.assertRaises(ValueError):
            policy.check_selection('SafeKreaUNETLoader',{'unet_name':policy.KREA_DIFFUSION[0],'weight_dtype':'fp8_e4m3fn'})

    def test_missing_file_never_loads_native_weights(self):
        folders=types.SimpleNamespace(get_full_path=lambda *a:None)
        native=MagicMock()
        with patch.dict(sys.modules,{'folder_paths':folders,'nodes':native}):
            with self.assertRaisesRegex(ValueError,'missing models'):
                mod.SafeH3AudioVAELoader().load(vae_name=policy.H3_AUDIO_VAE)
        native.VAELoader.assert_not_called()

    def test_success_delegates_unchanged_native_precision(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'fixture'; f.touch()
            folders=types.SimpleNamespace(get_full_path=lambda *a:str(f))
            native=MagicMock();native.CLIPLoader.return_value.load_clip.return_value=('same-object',)
            with patch.dict(sys.modules,{'folder_paths':folders,'nodes':native}):
                out=mod.SafeKreaCLIPLoader().load(clip_name=policy.KREA_ENCODER,type='krea2',device='default')
            self.assertEqual(out,('same-object',))
            native.CLIPLoader.return_value.load_clip.assert_called_once_with(policy.KREA_ENCODER,'krea2','default')

    def graph(self):
        return {'nodes':[{'id':1,'type':'CLIPLoader','widgets_values':[policy.KREA_ENCODER,'krea2','default'],'properties':{'cnr_id':'comfy-core','models':[{'name':'wrong'}]},'inputs':[],'outputs':[]}],'links':[]}

    def test_protect_does_not_mutate_source(self):
        g=self.graph();saved=copy.deepcopy(g)
        got=policy.protect_graph(g)
        self.assertEqual(g,saved);self.assertEqual(got['nodes'][0]['type'],'SafeKreaCLIPLoader')
        self.assertNotIn('models',got['nodes'][0]['properties'])

    def test_protection_is_idempotent(self):
        a=policy.protect_graph(self.graph())
        self.assertEqual(a,policy.protect_graph(a))

    def test_named_and_positional_drift_is_rejected(self):
        g=self.graph();g['nodes'][0]['widgets_values_named']={'clip_name':policy.H3_ENCODERS[0]}
        with self.assertRaisesRegex(ValueError,'disagree'):policy.protect_graph(g)

    def test_source_errors_are_not_silently_repaired(self):
        g=self.graph();g['nodes'][0]['widgets_values'][1]='lumina2'
        with self.assertRaises(ValueError):policy.protect_graph(g)

    def test_native_view_preserves_connections(self):
        g=self.graph();protected=policy.protect_graph(g)
        native=policy.native_view(protected)
        self.assertEqual(native['nodes'][0]['type'],'CLIPLoader')
        self.assertEqual(native['links'],g['links'])


if __name__=='__main__':unittest.main()
