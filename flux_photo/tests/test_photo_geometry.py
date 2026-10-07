"""CPU tests only. Synthetic patches do not establish model/identity quality."""
import gc
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from node.photo_ops import (crop_photo, fit_dimensions, inward_alpha, prepare_reference,
                            resize_rgb, stitch_photo, to_uint8)

torch.set_num_threads(2)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.linspace(0, 1, 137 * 193 * 3).reshape(1, 137, 193, 3)
        self.mask = torch.zeros((1, 137, 193))
        self.mask[:, 30:100, 40:145] = 1

    def assert_protected(self, source, output, mask):
        self.assertEqual(tuple(source.shape), tuple(output.shape))
        self.assertTrue(torch.equal(source[mask == 0], output[mask == 0]))

    def test_native_padding_never_stretches(self):
        data, work, noise = crop_photo(self.source, self.mask, 7, 'native', 256, 0)
        p = data['plan']
        self.assertEqual((p.crop_width, p.crop_height), (119, 84))
        self.assertEqual((p.work_width, p.work_height), (128, 96))
        self.assertFalse(p.resampled)
        self.assertTrue(torch.equal(work[:, :84, :119], self.source[:, 23:107, 33:152]))
        self.assertEqual(tuple(noise.shape), (1, 96, 128))
        self.assertEqual(float(noise[:, 84:].sum()), 0)

    def test_fit_changes_only_crop_not_canvas(self):
        data, work, _ = crop_photo(self.source, self.mask, 10, 'fit', 64, 0)
        self.assertTrue(data['plan'].resampled)
        self.assertLessEqual(max(work.shape[1:3]), 64)
        out, report = stitch_photo(data, torch.ones_like(work), 0)
        self.assert_protected(self.source, out, self.mask)
        self.assertTrue(json.loads(report)['crop_resampled'])

    def test_native_limit_is_a_hard_error(self):
        with self.assertRaisesRegex(ValueError, 'never silently downsizes'):
            crop_photo(self.source, self.mask, 20, 'native', 64, 0)

    def test_no_upscale_small_crop(self):
        data, work, _ = crop_photo(self.source, self.mask, 0, 'fit', 2048, 0)
        self.assertFalse(data['plan'].resampled)
        self.assertEqual(work.shape[2], 112)

    def test_empty_mask_error(self):
        with self.assertRaisesRegex(ValueError, 'Empty mask'):
            crop_photo(self.source, torch.zeros_like(self.mask))

    def test_mask_size_is_not_silently_resized(self):
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            crop_photo(self.source, self.mask[:, :, :-1])

    def test_two_dimensional_mask_is_accepted(self):
        data, _, _ = crop_photo(self.source, self.mask[0])
        self.assertEqual(data['paste_mask'].ndim, 3)

    def test_nonfinite_mask_rejected(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value):
                bad = self.mask.clone(); bad[0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, 'NaN or infinity'):
                    crop_photo(self.source, bad)

    def test_out_of_range_mask_rejected(self):
        for value in (-0.1, 1.1):
            with self.subTest(value=value):
                bad = self.mask.clone(); bad[0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, 'values must'):
                    crop_photo(self.source, bad)

    def test_batch_and_rgba_rejected(self):
        for bad in (self.source.expand(2, -1, -1, -1), torch.zeros(1, 137, 193, 4)):
            with self.assertRaisesRegex(ValueError, 'one RGB'):
                crop_photo(bad, self.mask)

    def test_sampling_grow_never_expands_paste(self):
        data, work, noise = crop_photo(self.source, self.mask, 20, 'native', 256, 16)
        self.assertGreater(float(noise.sum()), float(data['paste_mask'].sum()))
        out, report = stitch_photo(data, torch.ones_like(work), 0)
        self.assert_protected(self.source, out, self.mask)
        self.assertEqual(json.loads(report)['outside_mask_changed_channels'], 0)

    def test_feather_stays_inside_mask_and_holes(self):
        self.mask[:, 45:60, 50:60] = 0
        self.mask[:, 60:80, 75:100] = 0.3
        data, work, _ = crop_photo(self.source, self.mask, 20, 'native', 256, 16)
        out, _ = stitch_photo(data, torch.ones_like(work), 16)
        self.assert_protected(self.source, out, self.mask)
        alpha = inward_alpha(data['paste_mask'], 16, data['plan'])
        self.assertTrue(torch.all(alpha <= data['paste_mask']))
        self.assertTrue(torch.all(alpha[data['paste_mask'] == 0] == 0))

    def test_source_and_mask_not_mutated(self):
        source_copy = self.source.clone(); mask_copy = self.mask.clone()
        data, work, _ = crop_photo(self.source, self.mask, 16, 'fit', 64, 16)
        stitch_photo(data, torch.ones_like(work), 16, 1.0)
        self.assertTrue(torch.equal(self.source, source_copy))
        self.assertTrue(torch.equal(self.mask, mask_copy))

    def test_color_matching_disabled_by_default(self):
        data, work, _ = crop_photo(self.source, self.mask)
        out, report = stitch_photo(data, torch.full_like(work, 0.7), 0)
        self.assertEqual(json.loads(report)['boundary_rgb_shift'], [0, 0, 0])
        self.assertAlmostEqual(out[0, 70, 90, 0].item(), 0.7, places=6)

    def test_optional_color_matching_protects_unmasked(self):
        data, work, _ = crop_photo(self.source, self.mask)
        out, report = stitch_photo(data, torch.full_like(work, 0.7), 16, 0.5)
        self.assert_protected(self.source, out, self.mask)
        self.assertTrue(all(abs(v) <= 0.02001 for v in json.loads(report)['boundary_rgb_shift']))

    def test_wrong_generated_size_rejected(self):
        data, work, _ = crop_photo(self.source, self.mask)
        with self.assertRaisesRegex(ValueError, 'misaligned'):
            stitch_photo(data, work[:, :, :-1])

    def test_nonfinite_generated_rejected(self):
        data, work, _ = crop_photo(self.source, self.mask)
        work = work.clone(); work[0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'NaN/infinity'):
            stitch_photo(data, work)

    def test_full_canvas_mask_does_not_fade_at_frame(self):
        self.mask.fill_(1)
        data, work, _ = crop_photo(self.source, self.mask, 16, 'native', 256, 16)
        out, _ = stitch_photo(data, torch.ones_like(work), 16)
        self.assertTrue(torch.allclose(out, torch.ones_like(out)))

    def test_photo_edge_mask_is_protected_elsewhere(self):
        self.mask.zero_(); self.mask[:, :100, :145] = 1
        data, work, _ = crop_photo(self.source, self.mask, 16, 'native', 256, 16)
        out, _ = stitch_photo(data, torch.ones_like(work), 16)
        self.assert_protected(self.source, out, self.mask)
        self.assertEqual(out[0, 0, 0, 0].item(), 1.0)

    def test_disconnected_mask_is_single_bbox_without_extra_paste(self):
        self.mask.zero_(); self.mask[:, 10:20, 20:30] = 1; self.mask[:, 100:115, 160:180] = 1
        data, work, _ = crop_photo(self.source, self.mask, 7, 'native', 256, 0)
        out, _ = stitch_photo(data, torch.ones_like(work), 0)
        self.assert_protected(self.source, out, self.mask)

    def test_thin_mask_survives_fit_for_sampling(self):
        self.mask.zero_(); self.mask[:, 10:120, 30] = 1
        _, _, noise = crop_photo(self.source, self.mask, 32, 'fit', 32, 0)
        self.assertGreater(float(noise.sum()), 0)

    def test_lanczos_is_float_not_eight_bit_working_data(self):
        out = resize_rgb(self.source, 60, 45)
        self.assertEqual(out.dtype, torch.float32)
        self.assertGreater(float(torch.max(torch.abs(out * 255 - torch.round(out * 255)))), 0.01)

    def test_all_eight_bit_values_roundtrip(self):
        values = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (2, 1, 3))
        working = torch.from_numpy(values.astype(np.float32) / 255).unsqueeze(0)
        np.testing.assert_array_equal(to_uint8(working), values)

    def test_reference_aspect_and_padding(self):
        result = prepare_reference(torch.ones(1, 700, 333, 3), 512)
        self.assertEqual(tuple(result.shape), (1, 512, 256, 3))
        self.assertEqual(fit_dimensions(333, 700, 512), (244, 512))

    def test_invalid_settings_fail(self):
        for changes in ({'max_crop_side': 63}, {'processing': 'automatic'}, {'context_pixels': -1}, {'sampling_mask_grow': -1}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError): crop_photo(self.source, self.mask, **changes)


class LargeCanvasTests(unittest.TestCase):
    def exercise(self, width, height):
        # Deliberately odd 7K canvas and a genuine 2500x2500 mask. No model is used.
        source = torch.full((1, height, width, 3), 91 / 255, dtype=torch.float32)
        mask = torch.zeros((1, height, width), dtype=torch.float32)
        x = (width - 2500) // 2; y = (height - 2500) // 2
        mask[:, y:y+2500, x:x+2500] = 1
        data, work, _ = crop_photo(source, mask, 128, 'native', 3072, 16)
        self.assertEqual(tuple(work.shape), (1, 2768, 2768, 3))
        self.assertFalse(data['plan'].resampled)
        out, report = stitch_photo(data, torch.full_like(work, 0.7), 16)
        self.assertEqual(tuple(out.shape), (1, height, width, 3))
        # Row-wise verification keeps the test's peak CPU memory bounded.
        for start in range(0, height, 256):
            protected = mask[:, start:start+256] == 0
            self.assertTrue(torch.equal(out[:, start:start+256][protected], source[:, start:start+256][protected]))
        self.assertAlmostEqual(out[0, y+1250, x+1250, 0].item(), 0.7, places=6)
        self.assertEqual(json.loads(report)['outside_mask_changed_channels'], 0)
        del source, mask, data, work, out
        gc.collect()

    def test_7001_by_4003_landscape(self): self.exercise(7001, 4003)
    def test_4003_by_7001_portrait(self): self.exercise(4003, 7001)


if __name__ == '__main__': unittest.main()
