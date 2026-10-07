import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import node as photo
from node.photo_ops import crop_photo, stitch_photo, to_uint8

torch.set_num_threads(2)


class NodeTests(unittest.TestCase):
    def test_loader_uses_exact_matching_models_and_flux2_type(self):
        paths = types.SimpleNamespace(get_full_path=lambda *a: '/exists')
        core = types.SimpleNamespace(UNETLoader=Mock(), CLIPLoader=Mock(), VAELoader=Mock())
        core.UNETLoader.return_value.load_unet.return_value = ('model',)
        core.CLIPLoader.return_value.load_clip.return_value = ('clip',)
        core.VAELoader.return_value.load_vae.return_value = ('vae',)
        with patch.dict(sys.modules, {'folder_paths': paths, 'nodes': core}):
            self.assertEqual(photo.FluxPhotoModels().load(), ('model', 'clip', 'vae'))
        core.UNETLoader.return_value.load_unet.assert_called_once_with('flux-2-klein-9b.safetensors', 'default')
        core.CLIPLoader.return_value.load_clip.assert_called_once_with('qwen_3_8b.safetensors', 'flux2', device='cpu')
        core.VAELoader.return_value.load_vae.assert_called_once_with('flux2-vae.safetensors')

    def test_missing_model_fails_before_loading(self):
        paths = types.SimpleNamespace(get_full_path=lambda *a: None)
        with patch.dict(sys.modules, {'folder_paths': paths, 'nodes': types.SimpleNamespace()}):
            with self.assertRaisesRegex(FileNotFoundError, 'HF_TOKEN'): photo.FluxPhotoModels().load()

    def test_none_lora_passes_model_through(self):
        model = object()
        self.assertIs(photo.FluxPhotoOptionalLoRA().apply(model)[0], model)

    def test_zero_strength_does_not_load(self):
        model = object()
        self.assertIs(photo.FluxPhotoOptionalLoRA().apply(model, 'missing.safetensors', 0)[0], model)

    def test_lora_none_is_valid_with_empty_directory(self):
        with patch.dict(sys.modules, {'folder_paths': types.SimpleNamespace(get_filename_list=lambda _: [])}):
            self.assertEqual(photo.FluxPhotoOptionalLoRA.INPUT_TYPES()['required']['lora_name'][0], ['None'])

    def test_unsafe_lora_format_rejected(self):
        with self.assertRaisesRegex(ValueError, 'safetensors'):
            photo.FluxPhotoOptionalLoRA().apply(object(), 'adapter.pt', 0.8)

    def test_lora_core_model_only_api(self):
        loader = Mock(); loader.load_lora_model_only.return_value = ('patched',)
        with patch.dict(sys.modules, {'nodes': types.SimpleNamespace(LoraLoaderModelOnly=lambda: loader)}):
            self.assertEqual(photo.FluxPhotoOptionalLoRA().apply('model', 'adapter.safetensors', 0.6), ('patched',))
        loader.load_lora_model_only.assert_called_once_with('model', 'adapter.safetensors', 0.6)

    def fake_encoder(self, channels=128):
        calls = []
        class Encoder:
            def encode(inner, vae, image, size, overlap):
                calls.append((image, size, overlap))
                return ({'samples': torch.full((1, channels, image.shape[1]//16, image.shape[2]//16), float(image.mean()))},)
        return types.SimpleNamespace(VAEEncodeTiled=Encoder), calls

    def test_reference_order_mask_and_input_conditioning_not_mutated(self):
        core, calls = self.fake_encoder()
        original_meta = {'pooled_output': 'untouched'}
        cond = [[torch.zeros(1, 1, 1), original_meta]]
        crop = torch.full((1, 512, 768, 3), 0.2); mask = torch.ones(1, 512, 768)
        ref = torch.full((1, 128, 256, 3), 0.8)
        with patch.dict(sys.modules, {'nodes': core}):
            out, latent = photo.FluxPhotoReferenceConditioning().prepare(cond, 'vae', crop, mask, ref, 256, 256)
        self.assertNotIn('reference_latents', original_meta)
        refs = out[0][1]['reference_latents']
        self.assertEqual(len(refs), 2)
        self.assertAlmostEqual(float(refs[0].mean()), 0.2, places=5)
        self.assertAlmostEqual(float(refs[1].mean()), 0.8, places=5)
        self.assertIs(latent['noise_mask'], mask)
        self.assertEqual(latent['flux_photo_dimensions'], [768, 512])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c[1:] == (1024, 128) for c in calls))

    def test_same_size_scene_reuses_encoded_latent(self):
        core, calls = self.fake_encoder()
        with patch.dict(sys.modules, {'nodes': core}):
            out, latent = photo.FluxPhotoReferenceConditioning().prepare([[torch.zeros(1), {}]], 'vae',
                torch.ones(1, 256, 256, 3), torch.ones(1, 256, 256), torch.ones(1, 256, 256, 3))
        self.assertEqual(len(calls), 2)
        self.assertIs(out[0][1]['reference_latents'][0], latent['samples'])

    def test_wrong_vae_channels_rejected(self):
        core, _ = self.fake_encoder(16)
        with patch.dict(sys.modules, {'nodes': core}):
            with self.assertRaisesRegex(ValueError, '128-channel'):
                photo.FluxPhotoReferenceConditioning().prepare([[torch.zeros(1), {}]], 'vae',
                    torch.ones(1, 32, 32, 3), torch.ones(1, 32, 32), torch.ones(1, 32, 32, 3))

    def test_existing_refs_and_wrong_sampling_mask_rejected(self):
        core, _ = self.fake_encoder()
        with patch.dict(sys.modules, {'nodes': core}):
            for cond, mask, error in [([[torch.zeros(1), {'reference_latents': [1]}]], torch.ones(1, 32, 32), 'already contains'),
                                      ([[torch.zeros(1), {}]], torch.ones(1, 30, 32), 'match the crop')]:
                with self.subTest(error=error):
                    with self.assertRaisesRegex(ValueError, error):
                        photo.FluxPhotoReferenceConditioning().prepare(cond, 'vae', torch.ones(1, 32, 32, 3),
                            mask, torch.ones(1, 32, 32, 3))

    def fake_sampler_modules(self):
        comfy = types.ModuleType('comfy'); sample = types.ModuleType('comfy.sample')
        samplers = types.ModuleType('comfy.samplers'); utils = types.ModuleType('comfy.utils')
        sample.prepare_noise = Mock(return_value='noise'); sample.sample_custom = Mock(return_value=torch.ones(1))
        samplers.sampler_object = Mock(return_value='euler_sampler'); utils.PROGRESS_BAR_ENABLED = False
        comfy.sample = sample; comfy.samplers = samplers; comfy.utils = utils
        preview = types.ModuleType('latent_preview'); preview.prepare_callback = Mock(return_value='callback')
        extras = types.ModuleType('comfy_extras'); flux = types.ModuleType('comfy_extras.nodes_flux')
        flux.get_schedule = Mock(return_value='sigmas'); extras.nodes_flux = flux
        return {'comfy': comfy, 'comfy.sample': sample, 'comfy.samplers': samplers, 'comfy.utils': utils,
                'latent_preview': preview, 'comfy_extras': extras, 'comfy_extras.nodes_flux': flux}

    def test_sampler_uses_native_schedule_cfg_one_mask_and_four_steps(self):
        modules = self.fake_sampler_modules()
        latent = {'samples': torch.zeros(1), 'noise_mask': 'mask', 'flux_photo_dimensions': [2768, 2768]}
        with patch.dict(sys.modules, modules):
            out = photo.FluxPhotoSampler().sample('model', 'conditioning', latent, 123, 4)
        modules['comfy_extras.nodes_flux'].get_schedule.assert_called_once_with(4, 29929)
        args, kw = modules['comfy.sample'].sample_custom.call_args
        self.assertEqual(args[:7], ('model', 'noise', 1.0, 'euler_sampler', 'sigmas', 'conditioning', 'conditioning'))
        self.assertEqual(kw['noise_mask'], 'mask'); self.assertEqual(kw['seed'], 123)
        self.assertIsNot(out[0], latent)
        self.assertEqual(float(latent['samples'][0]), 0.0)

    def test_sampler_oom_never_silently_reduces_quality(self):
        modules = self.fake_sampler_modules(); modules['comfy.sample'].sample_custom.side_effect = torch.cuda.OutOfMemoryError()
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, 'No automatic precision or resolution'):
                photo.FluxPhotoSampler().sample('model', [], {'samples': torch.zeros(1), 'noise_mask': 'm', 'flux_photo_dimensions': [32, 32]})

    def test_saved_png_exact_outside_mask_and_includes_icc_and_report(self):
        rng = np.random.default_rng(123)
        src8 = rng.integers(0, 256, size=(73, 101, 3), dtype=np.uint8)
        src = torch.from_numpy(src8.astype(np.float32)/255).unsqueeze(0)
        mask = torch.zeros(1, 73, 101); mask[:, 20:55, 30:70] = 1
        data, work, _ = crop_photo(src, mask, 8, 'native', 256, 0)
        edited, report = stitch_photo(data, torch.full_like(work, 0.8), 4)
        with tempfile.TemporaryDirectory() as td:
            paths = types.SimpleNamespace(get_save_image_path=lambda *a: (td, 'photo', 1, '', ''), get_output_directory=lambda: td)
            with patch.dict(sys.modules, {'folder_paths': paths}):
                ui = photo.FluxPhotoSave().save(edited, report, prompt={'p': 'x'}, extra_pnginfo={'workflow': {'nodes': []}})
            path = Path(td) / ui['ui']['images'][0]['filename']
            with Image.open(path) as im:
                self.assertEqual(im.size, (101, 73)); self.assertIn('icc_profile', im.info)
                self.assertIn('workflow', im.info); self.assertIn('prompt', im.info)
                actual = np.array(im)
                np.testing.assert_array_equal(actual[mask[0].numpy() == 0], src8[mask[0].numpy() == 0])
            self.assertEqual(json.loads(path.with_suffix('.verification.json').read_text())['outside_mask_changed_channels'], 0)

    def test_save_rejects_failed_or_mismatched_report(self):
        with tempfile.TemporaryDirectory() as td:
            paths = types.SimpleNamespace(get_save_image_path=lambda *a: (td, 'p', 1, '', ''), get_output_directory=lambda: td)
            with patch.dict(sys.modules, {'folder_paths': paths}):
                for report in ({'outside_mask_changed_channels': 1}, {'outside_mask_changed_channels': 0, 'output_size': [1, 1]}):
                    with self.assertRaises(ValueError): photo.FluxPhotoSave().save(torch.ones(1, 32, 32, 3), json.dumps(report))
            self.assertFalse(list(Path(td).glob('*.png')))

    def test_compare_only_resizes_preview_and_preserves_inputs(self):
        original = torch.ones(1, 512, 768, 3); edited = original * 0.5
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(sys.modules, {'folder_paths': types.SimpleNamespace(get_temp_directory=lambda: td)}):
                ui = photo.FluxPhotoCompare().compare(original, edited, '{}', 256)
            images = ui['ui']['photo_compare']; self.assertEqual(len(images), 2)
            self.assertEqual(ui['ui']['photo_report'], ['{}'])
            with Image.open(Path(td) / images[0]['filename']) as im: self.assertEqual(im.size, (256, 171))
        self.assertTrue(torch.all(original == 1)); self.assertEqual(tuple(edited.shape), (1, 512, 768, 3))

    def test_compare_rejects_mismatched_canvas(self):
        with patch.dict(sys.modules, {'folder_paths': types.SimpleNamespace()}):
            with self.assertRaises(ValueError): photo.FluxPhotoCompare().compare(torch.zeros(1, 4, 4, 3), torch.zeros(1, 5, 4, 3), '{}')


if __name__ == '__main__': unittest.main()
