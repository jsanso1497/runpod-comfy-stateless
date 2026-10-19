"""Current QOVERLAY contracts and real Workbench CPU pixel operations.

Registry resolution is stubbed; compositor and feather code are not mocked.
This suite does not claim a live ComfyUI or SAM/GPU execution.
"""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import pytest
import torch

from workbench.qwen_selective_overlay_workflow import (
    make_selective_overlay, verify_graph, update_overlay_task, REQUIRED_NODES,
    WORKFLOW_PATH, VERSION,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tools(monkeypatch):
    registry = types.ModuleType('workbench.registry')
    registry.choices = lambda *a, **k: []
    registry.check_selection = lambda *a, **k: a[-1]
    registry.lora_choices = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, 'workbench.registry', registry)
    path = ROOT / 'src/nodes/workbench_tools/__init__.py'
    spec = importlib.util.spec_from_file_location('_selective_overlay_tools_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def images(h=83, w=109, batch=1):
    rng = np.random.default_rng(1981)
    a8 = rng.integers(0, 256, (batch, h, w, 3), dtype=np.uint8)
    b8 = rng.integers(0, 256, a8.shape, dtype=np.uint8)
    a = torch.from_numpy(a8.astype(np.float32) / 255)
    b = torch.from_numpy(b8.astype(np.float32) / 255)
    m = torch.zeros((batch, h, w))
    m[:, 20:60, 30:80] = 1
    return a, b, m, a8, b8


def one(graph, kind):
    hits = [n for n in graph['nodes'] if n['type'] == kind]
    assert len(hits) == 1, kind
    return hits[0]


def parent(graph, node, name):
    p = next(p for p in node['inputs'] if p['name'] == name)
    row = next(r for r in graph['links'] if r[0] == p['link'])
    return next(n for n in graph['nodes'] if n['id'] == row[1]), row[2]


def test_graph_has_no_generation_crop_or_stitch():
    g = make_selective_overlay()
    verify_graph(g)
    assert not g.get('definitions', {}).get('subgraphs')
    assert {n['type'] for n in g['nodes']} <= set(REQUIRED_NODES) | {'Note'}
    forbidden = ('KSampler', 'VAE', 'UNET', 'CLIP', 'Q21PhotoCrop', 'Q21PhotoStitch',
                 'Q21PhotoReviewSave', 'WBOverlayExport', 'TextGenerate')
    assert not any(any(t in n['type'] for t in forbidden) for n in g['nodes'])


def test_selection_comes_only_from_original_a():
    g = make_selective_overlay()
    selector, comp = one(g, 'Q21PhotoMask'), one(g, 'WBComposite')
    a, slot = parent(g, comp, 'original_a')
    b, _ = parent(g, comp, 'candidate_b')
    assert a['type'] == b['type'] == 'LoadImage' and a['id'] != b['id']
    assert slot == 0
    assert parent(g, selector, 'source') == (a, 0)
    assert parent(g, selector, 'manual_mask') == (a, 1)
    assert not b['outputs'][1]['links']


def test_final_output_is_the_composite_not_a_preview_or_b():
    g = make_selective_overlay()
    assert parent(g, one(g, 'SaveImage'), 'images') == (one(g, 'WBComposite'), 0)
    assert len([n for n in g['nodes'] if n['type'] == 'SaveImage']) == 1
    assert one(g, 'SaveImage')['widgets_values'][0].endswith('FINAL_A_plus_selected_B')


def test_feather_drives_all_previews_and_composite():
    g = make_selective_overlay()
    feather = one(g, 'WBMaskExpandFeather')
    assert parent(g, feather, 'mask') == (one(g, 'Q21PhotoMask'), 0)
    for kind in ('WBComposite', 'WBMaskPreview', 'MaskToImage'):
        assert parent(g, one(g, kind), 'mask') == (feather, 0)
    original = parent(g, one(g, 'WBComposite'), 'original_a')
    assert parent(g, one(g, 'WBMaskPreview'), 'original_a') == original


def test_existing_sam_default_and_manual_connection_are_preserved():
    g = make_selective_overlay()
    assert one(g, 'Q21PhotoMask')['widgets_values'] == [True, 'watermark', .5, 2, -1, 'full photo', 'ignore', 0]
    assert one(g, 'WBMaskExpandFeather')['widgets_values'] == [0, 12, False]
    assert one(g, 'WBComposite')['inputs'][3] == {'name': 'protect_mask', 'type': 'MASK', 'link': None, 'shape': 7}


def test_builder_is_deterministic_and_returns_independent_data():
    a, b = make_selective_overlay(), make_selective_overlay()
    assert a == b
    a['nodes'][0]['title'] = 'changed'
    assert a != b and make_selective_overlay() == b
    assert b['extra']['workbench_source']['overlay_version'] == VERSION


@pytest.mark.parametrize('failure', ['input', 'output', 'type', 'duplicate', 'order'])
def test_graph_verifier_rejects_bad_wiring(failure):
    g = make_selective_overlay()
    if failure == 'input': g['nodes'][2]['inputs'][0]['link'] = 999
    if failure == 'output': g['nodes'][0]['outputs'][0]['links'] = []
    if failure == 'type': g['links'][0][-1] = 'MASK'
    if failure == 'duplicate': g['links'].append(copy.deepcopy(g['links'][0]))
    if failure == 'order': g['nodes'][2]['order'] = -1
    with pytest.raises(ValueError): verify_graph(g)


def test_catalog_changes_only_overlay():
    other = {'id': 'QNATIVE', 'requires': ['qwen', 'qwen_stock_pe']}
    before = copy.deepcopy(other)
    assert update_overlay_task(other) == before
    row = {'id': 'QOVERLAY', 'family': 'qwen', 'workspaces': ['qwen'],
           'requires': ['qwen', 'qwen_stock_pe', 'sam']}
    update_overlay_task(row)
    assert row['requires'] == ['sam']
    assert row['variants'][0]['file'] == WORKFLOW_PATH
    assert row['required_nodes'] == REQUIRED_NODES
    assert row['family'] == 'qwen' and row['workspaces'] == ['qwen']
    baseline = copy.deepcopy(row)
    assert update_overlay_task(row) == baseline


def test_white_is_b_and_black_is_exact_a(tools):
    a, b, mask, _, _ = images()
    original = a.clone(); candidate = b.clone(); saved_mask = mask.clone()
    result, report = tools.WBComposite().composite(a, b, mask)
    assert torch.equal(result[mask == 0], a[mask == 0])
    assert torch.equal(result[mask == 1], b[mask == 1])
    assert result.shape == a.shape
    assert torch.equal(a, original) and torch.equal(b, candidate) and torch.equal(mask, saved_mask)
    assert json.loads(report)['outside_selection_max_error'] == 0


@pytest.mark.parametrize('value', [0., 1., .25, .5, .75])
def test_constant_masks(tools, value):
    a, b, mask, _, _ = images()
    mask.fill_(value)
    result, _ = tools.WBComposite().composite(a, b, mask)
    assert torch.equal(result, a * (1-value) + b * value)


def test_inversion_selects_the_opposite_area(tools):
    a, b, mask, _, _ = images()
    inverse = tools.WBMaskExpandFeather().process(mask, 0, 0, True)[0]
    assert torch.equal(inverse, 1-mask)
    result, _ = tools.WBComposite().composite(a, b, inverse)
    assert torch.equal(result[mask == 1], a[mask == 1])
    assert torch.equal(result[mask == 0], b[mask == 0])


@pytest.mark.parametrize('grow,feather', [(0, 0), (0, 12), (4, 0), (4, 12), (12, 24)])
def test_finite_feather_preserves_pixels_outside_its_band(tools, grow, feather):
    a, b, mask, _, _ = images(129, 151)
    mask.zero_(); mask[:, 45:75, 60:90] = 1
    alpha = tools.WBMaskExpandFeather().process(mask, grow, feather, False)[0]
    result, report = tools.WBComposite().composite(a, b, alpha)
    assert torch.equal(result[alpha == 0], a[alpha == 0])
    assert torch.equal(result[alpha == 1], b[alpha == 1])
    assert torch.all(alpha[mask == 1] == 1)
    assert alpha[0, 0, 0].item() == 0
    assert json.loads(report)['outside_selection_max_error'] == 0
    if feather > 1: assert torch.any((alpha > 0) & (alpha < 1))


def test_hard_protection_is_applied_after_feather(tools):
    a, b, mask, _, _ = images()
    alpha = tools.WBMaskExpandFeather().process(mask, 6, 12, False)[0]
    protect = torch.zeros_like(alpha); protect[:, 25:40, 35:50] = 1
    result, _ = tools.WBComposite().composite(a, b, alpha, protect)
    assert torch.equal(result[protect > 0], a[protect > 0])
    assert torch.equal(result[alpha == 0], a[alpha == 0])


@pytest.mark.parametrize('failure', ['size', 'batch', 'channels', 'mask_size', 'mask_nan',
                                     'image_nan', 'negative_mask', 'large_mask', 'protect_size'])
def test_invalid_inputs_fail_instead_of_resizing_or_revealing_b(tools, failure):
    a, b, mask, _, _ = images()
    protect = None
    if failure == 'size': b = b[:, :-1]
    if failure == 'batch': b = b.repeat(2, 1, 1, 1)
    if failure == 'channels': b = b[..., :2]
    if failure == 'mask_size': mask = mask[:, :-1]
    if failure == 'mask_nan': mask[0, 0, 0] = float('nan')
    if failure == 'image_nan': b[0, 0, 0, 0] = float('nan')
    if failure == 'negative_mask': mask[0, 0, 0] = -.1
    if failure == 'large_mask': mask[0, 0, 0] = 1.1
    if failure == 'protect_size': protect = torch.zeros((1, 64, 64))
    with pytest.raises(ValueError): tools.WBComposite().composite(a, b, mask, protect)


def test_rgb_png_remains_full_size_and_preserves_unselected_pixels(tools, tmp_path):
    a, b, mask, original8, _ = images(257, 389)
    alpha = tools.WBMaskExpandFeather().process(mask, 0, 12, False)[0]
    final, _ = tools.WBComposite().composite(a, b, alpha)
    # Matches ordinary float-IMAGE -> 8-bit PNG conversion, not a ComfyUI runtime claim.
    encoded = np.clip(final[0].numpy() * 255, 0, 255).astype(np.uint8)
    path = tmp_path / 'full_photo.png'
    Image.fromarray(encoded).save(path)
    with Image.open(path) as image:
        assert image.mode == 'RGB' and image.size == (389, 257)
        pixels = np.asarray(image)
    outside = alpha[0].numpy() == 0
    assert np.array_equal(pixels[outside], original8[0][outside])


def test_graph_saves_pixel_composite_using_real_nodes(tools):
    # Execute the IMAGE/MASK route. Substitute only image loading and SAM selection.
    graph = make_selective_overlay()
    a, b, mask, _, _ = images()
    result_by_node = {1: (a, mask), 2: (b, torch.zeros_like(mask)), 3: (mask, '{}')}
    for node in graph['nodes']:
        kind = node['type']
        if kind in ('WBMaskExpandFeather', 'WBComposite', 'WBMaskPreview'):
            cls = getattr(tools, kind)
            kwargs = {}
            for p in node['inputs']:
                if p['link'] is not None:
                    source, slot = parent(graph, node, p['name'])
                    kwargs[p['name']] = result_by_node[source['id']][slot]
            connected = set(kwargs)
            names = [name for name, spec in cls.INPUT_TYPES()['required'].items()
                     if name not in connected and spec[0] in ('INT', 'BOOLEAN', 'FLOAT', 'STRING')]
            assert len(names) == len(node['widgets_values'])
            kwargs.update(zip(names, node['widgets_values']))
            result_by_node[node['id']] = getattr(cls(), cls.FUNCTION)(**kwargs)
        elif kind == 'MaskToImage':
            source, slot = parent(graph, node, 'mask')
            result_by_node[node['id']] = (result_by_node[source['id']][slot][..., None].repeat(1, 1, 1, 3),)
        elif kind == 'SaveImage':
            source, slot = parent(graph, node, 'images')
            final = result_by_node[source['id']][slot]
    alpha = result_by_node[4][0]
    assert final.shape == a.shape
    assert torch.equal(final[alpha == 0], a[alpha == 0])
    assert torch.equal(final[alpha == 1], b[alpha == 1])


def test_active_factory_adapter_and_catalog_when_available():
    path = ROOT / 'src/workbench/qwen_stock_workflows.py'
    if not path.exists():
        pytest.skip('Standalone payload test; full adapter checked in installer tests.')
    from workbench.qwen_stock_workflows import make_overlay
    from workbench.qwen_stock_catalog import apply
    assert make_overlay(None) == make_selective_overlay()
    rows = apply('tasks.json', {'tasks': []})['tasks']
    assert next(t for t in rows if t['id'] == 'QOVERLAY')['requires'] == ['sam']
    assert next(t for t in rows if t['id'] == 'QNATIVE')['requires'] == ['qwen', 'qwen_stock_pe']
