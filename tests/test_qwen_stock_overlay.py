# Legacy crop-generator compatibility; active A/B graph is tested in test_qwen_selective_overlay.py.
"""CPU contracts. No assertion here certifies live SAM or Qwen inference."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from workbench.qwen_overlay import compose_overlay, WBOverlayExport, WBOverlayProtection
from workbench.qwen_stock_workflows import make_stock, make_legacy_overlay as make_overlay, verify_links, build, UPSTREAM_REV
from workbench.qwen_stock_catalog import PE_FILE, apply, STOCK_PATH, OVERLAY_PATH


def package():
    rng = np.random.default_rng(41)
    source = rng.integers(0, 256, (1, 119, 137, 3), dtype=np.uint8)
    original = torch.from_numpy(source.astype(np.float32) / 255)
    p = SimpleNamespace(source_width=137, source_height=119, x=11, y=13,
                        crop_width=79, crop_height=65, scaled_width=79, scaled_height=65,
                        work_width=96, work_height=96)
    mask = torch.zeros((1, 65, 79)); mask[:, 4:60, 5:74] = 1
    protection = torch.zeros((1, 119, 137)); protection[:, 35:51, 36:60] = 1
    data = {'source': original, 'plan': p, 'paste_mask': mask, 'overlay_protection': protection}
    generated = torch.ones((1, 96, 96, 3))
    return data, generated, source[0]


def test_protection_and_original_dimensions():
    data, generated, original = package()
    final, overlay, mask, report = compose_overlay(data, generated, 12)
    assert final.shape == original.shape
    assert overlay.shape == (*original.shape[:2], 4)
    assert np.array_equal(final[mask == 0], original[mask == 0])
    protect = data['overlay_protection'][0].numpy() > 0
    assert np.array_equal(final[protect], original[protect])
    assert not np.any(mask[protect])
    assert report['protected_changed_channels'] == report['outside_overlay_changed_channels'] == 0


def test_straight_alpha_recomposes_exactly():
    data, generated, original = package()
    final, rgba, _, _ = compose_overlay(data, generated, 12)
    recomposed = np.rint(original.astype(np.float64) +
        (rgba[..., :3].astype(np.float64) - original) * (rgba[..., 3:4] / 255.0)).astype(np.uint8)
    assert np.array_equal(final, recomposed)
    assert np.any((rgba[..., 3] > 0) & (rgba[..., 3] < 255))


def test_padding_is_removed_before_pasting():
    data, generated, _ = package()
    baseline = compose_overlay(data, generated, 0)[0]
    generated[:, 65:, :, :] = 0
    generated[:, :, 79:, :] = 0
    assert np.array_equal(baseline, compose_overlay(data, generated, 0)[0])


def test_identical_pixels_have_no_overlay():
    data, generated, original = package()
    generated[:, :65, :79] = data['source'][:, 13:78, 11:90]
    final, rgba, _, report = compose_overlay(data, generated, 12)
    assert np.array_equal(final, original)
    assert not np.any(rgba)
    assert report['changed_pixels'] == 0


def test_generated_transparency_does_not_destroy_original():
    data, generated, original = package()
    transparent = torch.cat([generated, torch.zeros((1, 96, 96, 1))], dim=-1)
    final, rgba, _, _ = compose_overlay(data, transparent, 0)
    assert np.array_equal(final, original)
    assert not np.any(rgba)


@pytest.mark.parametrize('failure', ['size', 'mask', 'nan', 'feather'])
def test_invalid_geometry_fails_closed(failure):
    data, generated, _ = package()
    feather = 12
    if failure == 'size': generated = generated[:, :-1]
    if failure == 'mask': data['overlay_protection'] = torch.zeros((1, 64, 64))
    if failure == 'nan': generated[0, 0, 0, 0] = float('nan')
    if failure == 'feather': feather = -1
    with pytest.raises(ValueError): compose_overlay(data, generated, feather)


def test_disabled_protection_requires_no_sam():
    out = WBOverlayProtection().detect(torch.zeros((1, 91, 117, 3)), enabled=False)[0]
    assert out.shape == (1, 91, 117)
    assert not torch.any(out)


def test_failed_sam_is_not_silently_ignored(monkeypatch):
    import sys
    class Failed:
        def make_mask(self, **kwargs): raise ValueError('SAM found no face')
    monkeypatch.setitem(sys.modules, 'nodes', SimpleNamespace(NODE_CLASS_MAPPINGS={'Q21PhotoMask': Failed}))
    with pytest.raises(ValueError, match='no face'):
        WBOverlayProtection().detect(torch.zeros((1, 91, 117, 3)))


def test_review_is_lazy():
    node = WBOverlayExport()
    assert node.check_lazy_status(run_edit=False) == []
    assert node.check_lazy_status(run_edit=True) == ['generated_crop']
    assert node.check_lazy_status(run_edit=True, generated_crop=torch.zeros(1)) == []


def fixture_graph():
    # CI uses the REAL pinned template fetched by build_site.py. This small
    # fixture also exercises the adapter when network access is unavailable.
    archive = Path(__file__).resolve().parents[1] / 'vendor/qwen-stock' / (UPSTREAM_REV + '.json')
    if archive.exists(): return json.loads(archive.read_text())
    values = {'resolution': 0, 'prompt': 'Edit the photograph.',
              'cfg': 1, 'steps': 25, 'scheduler': 'euler', 'scheduler_1': 'simple',
              'switch': False, 'switch_1': False, 'thinking': False,
              'device': 'auto', 'dtype': 'default', 'clip_name_1': PE_FILE,
              'unet_name': 'qwen_image_2.1_int8_convrot.safetensors',
              'clip_name': 'qwen3vl_8b_int8_convrot.safetensors'}
    return {'id': 'fixture', 'nodes': [
        {'id': 470, 'type': 'LoadImage', 'inputs': [], 'outputs': [
            {'name': 'IMAGE', 'type': 'IMAGE', 'links': [704]},
            {'name': 'MASK', 'type': 'MASK', 'links': []}]},
        {'id': 459, 'type': 'native-edit', 'inputs': [{'name': 'images.image_1', 'label': 'image_1',
            'type': 'IMAGE', 'link': 704}], 'outputs': [{'name': 'IMAGE', 'type': 'IMAGE', 'links': []}],
         'widgets_values_named': values, 'widgets_values': list(values.values())}],
        'links': [[704, 470, 0, 459, 0, 'IMAGE']], 'definitions': {'subgraphs': [{
            'id': 'native-edit', 'nodes': [{'type': 'TextGenerate', 'widgets_values': [1, 20, .95, 0, 1, 0],
            'system_prompt': 'unchanged upstream system prompt'}]}]}, 'extra': {}}


def test_stock_preserves_defaults_and_prompt_enhancer():
    upstream = fixture_graph()
    pristine = copy.deepcopy(upstream)
    stock = make_stock(upstream)
    assert upstream == pristine
    assert stock['links'] == upstream['links']
    before = next(n for n in upstream['nodes'] if 'unet_name' in n.get('widgets_values_named', {}))
    after = next(n for n in stock['nodes'] if 'unet_name' in n.get('widgets_values_named', {}))
    for key, value in before['widgets_values_named'].items():
        if key not in ('unet_name', 'clip_name'):
            assert after['widgets_values_named'][key] == value
    assert after['widgets_values_named']['clip_name_1'] == PE_FILE
    assert after['widgets_values_named']['unet_name'] == 'qwen_image_2.1_bf16.safetensors'
    assert after['widgets_values_named']['clip_name'] == 'qwen3vl_8b_bf16.safetensors'


def test_legacy_overlay_uses_identical_native_subgraph():
    stock = make_stock(fixture_graph())
    overlay = make_overlay(stock)
    assert overlay['definitions'] == stock['definitions']
    verify_links(overlay)
    assert not any(n['type'] in ('WBOverlayProtection', 'WBOverlayPrepare', 'WBOverlayExport')
                   for n in overlay['nodes'])
    assert sum(n['type'] == 'Q21PhotoCrop' for n in overlay['nodes']) == 1
    assert sum(n['type'] == 'Q21PhotoStitch' for n in overlay['nodes']) == 1
    assert sum(n['type'] == 'Q21PhotoMask' for n in overlay['nodes']) == 1
    assert not any(n['type'] in ('SaveImage', 'SaveImageAdvanced', 'ImageCompare') for n in overlay['nodes'])
    save = next(n for n in overlay['nodes'] if n['type'] == 'Q21PhotoReviewSave')
    assert save['widgets_values'][0] is True
    mask = next(n for n in overlay['nodes'] if n['type'] == 'Q21PhotoMask')
    assert mask['widgets_values'][:2] == [True, 'watermark']


def test_build_is_reproducible_offline(tmp_path):
    archive = tmp_path / 'vendor/qwen-stock' / (UPSTREAM_REV + '.json')
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps(fixture_graph()))
    build(tmp_path, offline=True)
    first = [(tmp_path / p).read_bytes() for p in (STOCK_PATH, OVERLAY_PATH)]
    build(tmp_path, offline=True)
    assert first == [(tmp_path / p).read_bytes() for p in (STOCK_PATH, OVERLAY_PATH)]



def test_catalog_additive_and_separate_from_custom_enhancer():
    value = {'tasks': [{'id': 'old', 'requires': ['qwen']}]}
    apply('tasks.json', value)
    assert value['tasks'][0] == {'id': 'old', 'requires': ['qwen']}
    tasks = {x['id']: x for x in value['tasks']}
    assert set(tasks) == {'old', 'QNATIVE', 'QOVERLAY'}
    assert tasks['QNATIVE']['requires'] == ['qwen', 'qwen_stock_pe']
    assert tasks['QOVERLAY']['requires'] == ['sam']
    for ident in ('QNATIVE', 'QOVERLAY'):
        assert 'ollama' not in tasks[ident]['requires']
        assert 'qwen_pe_i2i' not in tasks[ident]['requires']
