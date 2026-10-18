"""QOVERLAY regressions: actual CPU mask/crop/stitch/save, not GPU inference.

Only ComfyUI model loading/SAM inference and folder_paths are mocked. These
checks use the repository's real photo nodes and real PNG encoder.
"""
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
import torch.nn.functional as F

from workbench.qwen_stock_workflows import make_stock, make_overlay, native_node, verify_links, WATERMARK_PROMPT
from workbench.qwen_stock_catalog import apply, PE_FILE

ROOT = Path(__file__).resolve().parents[1]


def source_fixture():
    spec = importlib.util.spec_from_file_location('_watermark_stock_fixture',
                                                  ROOT / 'tests/test_qwen_stock_overlay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture_graph()


@pytest.fixture
def graph():
    return make_overlay(make_stock(source_fixture()))


def one(graph, kind):
    values = [n for n in graph['nodes'] if n['type'] == kind]
    assert len(values) == 1, kind
    return values[0]


def upstream(graph, node, socket):
    port = next(p for p in node['inputs'] if p['name'] == socket)
    row = next(l for l in graph['links'] if l[0] == port['link'])
    return next(n for n in graph['nodes'] if n['id'] == row[1]), row[2]


def test_watermark_defaults_and_native_settings():
    stock = make_stock(source_fixture())
    baseline = copy.deepcopy(stock)
    graph = make_overlay(stock)
    assert stock == baseline
    assert graph['definitions'] == stock['definitions']
    before, after = native_node(stock), native_node(graph)
    assert after['widgets_values_named']['prompt'] == WATERMARK_PROMPT
    assert after['widgets_values'][1] == WATERMARK_PROMPT
    assert WATERMARK_PROMPT.startswith('Remove the watermark')
    for k, value in before['widgets_values_named'].items():
        if k != 'prompt':
            assert after['widgets_values_named'][k] == value
    assert after['widgets_values'][:1] + after['widgets_values'][2:] == before['widgets_values'][:1] + before['widgets_values'][2:]
    assert after['widgets_values_named']['clip_name_1'] == PE_FILE
    assert one(graph, 'Q21PhotoMask')['widgets_values'][:2] == [True, 'watermark']
    assert one(graph, 'Q21PhotoReviewSave')['widgets_values'][0] is True


def test_single_crop_and_only_full_photo_output(graph):
    verify_links(graph)
    crop, stitch, save = (one(graph, k) for k in ('Q21PhotoCrop', 'Q21PhotoStitch', 'Q21PhotoReviewSave'))
    assert upstream(graph, native_node(graph), 'images.image_1') == (crop, 1)
    assert upstream(graph, stitch, 'crop_data') == (crop, 0)
    assert upstream(graph, save, 'crop_data') == (crop, 0)
    assert upstream(graph, save, 'edited_result') == (stitch, 0)
    assert upstream(graph, stitch, 'generated_crop') == (native_node(graph), 0)
    forbidden = {'WBOverlayExport', 'WBOverlayPrepare', 'WBOverlayProtection', 'SaveImage',
                 'SaveImageAdvanced', 'PreviewImage', 'ImageCompare'}
    assert not (forbidden & {n['type'] for n in graph['nodes']})
    for definition in graph.get('definitions', {}).get('subgraphs', []):
        assert not (forbidden & {n['type'] for n in definition['nodes']})


def test_protection_is_connected_to_final_paste_mask(graph):
    mask, grow, protect, gate, crop = (one(graph, k) for k in
        ('Q21PhotoMask', 'GrowMask', 'InvertMask', 'MaskComposite', 'Q21PhotoCrop'))
    assert grow['widgets_values'] == [16, False]
    assert upstream(graph, grow, 'mask') == (mask, 0)
    assert upstream(graph, protect, 'mask') == (grow, 0)
    assert upstream(graph, gate, 'destination') == (mask, 0)
    assert upstream(graph, gate, 'source') == (protect, 0)
    assert gate['widgets_values'] == [0, 0, 'subtract']
    assert upstream(graph, crop, 'mask') == (gate, 0)
    assert crop['widgets_values'] == [2048, 3072, 128, False]


def test_catalog_keeps_stock_enhancer_and_legacy_nodes():
    tasks = apply('tasks.json', {'tasks': []})['tasks']
    overlay = next(t for t in tasks if t['id'] == 'QOVERLAY')
    assert set(overlay['requires']) == {'qwen', 'qwen_stock_pe', 'sam'}
    assert {'Q21PhotoCrop', 'Q21PhotoStitch', 'Q21PhotoReviewSave'} <= set(overlay['required_nodes'])
    assert 'WBOverlayExport' not in overlay['required_nodes']
    models = apply('models.json', {'models': [{'id': 'qwen'}]})
    assert 'qwen_overlay' in models['models'][0]['local_nodes']  # Old saved graphs still load.
    row = apply('assets.json', {'assets': []})['assets'][0]
    assert row['sha256'] == '32707d01b427e488af252b95c551989aad59f9fec611a694f5db6bde7f0f1f6c'


@pytest.fixture
def photo(tmp_path, monkeypatch):
    # Isolate runtime registration only, not geometry or image processing.
    registry = types.ModuleType('workbench.registry')
    registry.choices = lambda *a, **k: []
    registry.check_selection = lambda *a, **k: a[-1]
    registry.lora_choices = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, 'workbench.registry', registry)
    name = '_watermark_photo_cpu'
    for key in list(sys.modules):
        if key == name or key.startswith(name + '.'):
            monkeypatch.delitem(sys.modules, key)
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src/nodes/qwen_photo/__init__.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    folders = types.ModuleType('folder_paths')
    out = tmp_path / 'output'
    out.mkdir()
    folders.get_output_directory = lambda: str(out)
    folders.get_temp_directory = lambda: str(tmp_path / 'temp')
    def save_path(prefix, root, width, height):
        path = Path(root) / prefix
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path.parent), path.name, 1, str(path.parent.relative_to(root)), prefix
    folders.get_save_image_path = save_path
    folders.get_full_path = lambda *a: str(tmp_path / 'sam.safetensors')
    monkeypatch.setitem(sys.modules, 'folder_paths', folders)
    monkeypatch.setenv('WB_SAVE_METADATA', '0')
    return module, out


def widget_parameters(node, cls):
    spec = cls.INPUT_TYPES()
    # ComfyUI serializes only unconnected widget inputs in widgets_values.
    connected = {p['name'] for p in node['inputs'] if p.get('link') is not None}
    names = []
    for category in ('required', 'optional'):
        for name, definition in spec.get(category, {}).items():
            if name in connected:
                continue
            kind = definition[0]
            if isinstance(kind, (list, tuple)) or kind in ('STRING', 'BOOLEAN', 'INT', 'FLOAT'):
                names.append(name)
    assert len(names) == len(node['widgets_values'])
    return dict(zip(names, node['widgets_values']))


def final_mask(edit):
    # Same square, non-tapered dilation and subtract/clamp as the pinned core nodes.
    grown = F.max_pool2d(edit[:, None], 33, stride=1, padding=16)[:, 0]
    protected = 1 - grown
    return (edit - protected).clamp(0, 1), protected


def source_and_mask(h=513, w=769, edge=False):
    rng = np.random.default_rng(1647)
    rgb = rng.integers(25, 210, (h, w, 3), dtype=np.uint8)
    source = torch.from_numpy(rgb.astype(np.float32) / 255).unsqueeze(0)
    mask = torch.zeros((1, h, w))
    if edge:
        mask[:, :h//2, :w//2] = 1
    else:
        mask[:, h//4:3*h//4, w//4:3*w//4] = 1
        mask[:, h//2:h//2+12, w//2:w//2+12] = 0
    return rgb, source, mask


@pytest.mark.parametrize('resolution', [256, 512, 2048])
@pytest.mark.parametrize('rgba', [False, True])
@pytest.mark.parametrize('edge', [False, True])
def test_real_png_is_full_size_opaque_and_preserves_outside(photo, graph, resolution, rgba, edge):
    module, out = photo
    rgb, source, mask = source_and_mask(edge=edge)
    effective, protected = final_mask(mask)
    assert torch.equal(effective, mask)
    params = widget_parameters(one(graph, 'Q21PhotoCrop'), module.Q21PhotoCrop)
    params['resolution'] = resolution
    data, working = module.Q21PhotoCrop().crop(source, effective, '{}', **params)
    generated = torch.full_like(working, .9)
    if rgba:
        alpha = torch.ones((*working.shape[:3], 1))
        alpha[:, :working.shape[1]//3] = 0  # A transparent generated region must not erase the photo.
        generated = torch.cat((generated, alpha), -1)
    stitch_params = widget_parameters(one(graph, 'Q21PhotoStitch'), module.Q21PhotoStitch)
    result = module.Q21PhotoStitch().stitch(data, generated, '{}', **stitch_params)[0]
    save_params = widget_parameters(one(graph, 'Q21PhotoReviewSave'), module.Q21PhotoReviewSave)
    response = module.Q21PhotoReviewSave().review(data, edited_result=result, **save_params)
    files = response['ui']['images']
    assert len(files) == 1 and files[0]['type'] == 'output'
    path = out / files[0]['subfolder'] / files[0]['filename']
    with Image.open(path) as image:
        assert image.mode == 'RGB'
        assert image.size == (rgb.shape[1], rgb.shape[0])
        saved = np.asarray(image)
    outside = effective[0].numpy() == 0
    assert np.array_equal(saved[outside], rgb[outside])
    assert np.array_equal(saved[protected[0].numpy() == 1], rgb[protected[0].numpy() == 1])
    assert np.any(saved[~outside] != rgb[~outside])
    assert result['report']['outside_mask_changed_channels'] == 0
    assert not list(out.rglob('*_overlay.png'))
    assert all(Image.open(p).mode != 'RGBA' for p in out.rglob('*.png'))


def test_actual_2048px_source_keeps_full_dimensions(photo):
    module, _ = photo
    rgb, source, mask = source_and_mask(2048, 1365)
    data, work = module.Q21PhotoCrop().crop(source, mask, '{}', resolution=512)
    result = module.Q21PhotoStitch().stitch(data, torch.ones_like(work), '{}')[0]['image']
    assert result.shape == source.shape
    assert torch.equal(result[mask == 0], source[mask == 0])


def test_soft_mask_does_not_expand_the_paste_region(photo):
    module, _ = photo
    _, source, mask = source_and_mask(119, 137)
    mask[:, 8:15, 10:15] = .3
    effective, protected = final_mask(mask)
    assert torch.all(effective <= mask)
    data, work = module.Q21PhotoCrop().crop(source, effective, '{}', resolution=256)
    image = module.Q21PhotoStitch().stitch(data, torch.ones_like(work), '{}')[0]['image']
    assert torch.equal(image[mask == 0], source[mask == 0])


def test_manual_mask_mode_still_works(photo):
    module, _ = photo
    _, source, mask = source_and_mask(119, 137)
    selected, report = module.Q21PhotoMask().make_mask(source, auto_mask=False, manual_mask=mask)
    assert torch.equal(selected, mask)
    assert json.loads(report)['auto_mask'] is False


def test_sam_receives_watermark_and_selects_all_instances(photo, graph, monkeypatch):
    module, _ = photo
    _, source, mask = source_and_mask(119, 137)
    second_mask = torch.zeros_like(mask)
    second_mask[:, 2:7, 2:7] = 1
    expected_mask = torch.maximum(mask, second_mask)
    events = []
    n = types.ModuleType('nodes')
    n.CheckpointLoaderSimple = lambda: types.SimpleNamespace(load_checkpoint=lambda file: ('sam', 'clip', None))
    n.CLIPTextEncode = lambda: types.SimpleNamespace(encode=lambda clip, text: (events.append(text) or 'conditioning',))
    detect = types.ModuleType('comfy_extras.nodes_sam3')
    class SAM:
        @classmethod
        def execute(cls, **kwargs):
            assert kwargs['individual_masks'] is True
            return (torch.cat([mask, second_mask]),)
    detect.SAM3_Detect = SAM
    extras = types.ModuleType('comfy_extras')
    comfy = types.ModuleType('comfy')
    management = types.ModuleType('comfy.model_management')
    management.unload_all_models = lambda: events.append('unload')
    comfy.model_management = management
    for name, value in [('nodes', n), ('comfy_extras', extras), ('comfy_extras.nodes_sam3', detect),
                        ('comfy', comfy), ('comfy.model_management', management)]:
        monkeypatch.setitem(sys.modules, name, value)
    params = widget_parameters(one(graph, 'Q21PhotoMask'), module.Q21PhotoMask)
    selected, report = module.Q21PhotoMask().make_mask(source, manual_mask=torch.zeros(1, 64, 64), **params)
    assert torch.equal(selected, expected_mask)
    assert events == ['watermark:64', 'unload']
    assert json.loads(report)['instance_index'] == -1


def test_no_sam_detection_never_falls_back_to_full_image(photo):
    module, _ = photo
    with pytest.raises(ValueError, match='empty mask'):
        module.choose_instances(torch.zeros(1, 119, 137), -1)


@pytest.mark.parametrize('bad', ['size', 'nan', 'empty_mask'])
def test_invalid_inputs_stop_before_saving(photo, bad):
    module, out = photo
    _, source, mask = source_and_mask(119, 137)
    if bad == 'empty_mask':
        with pytest.raises(ValueError, match='Empty mask'):
            module.Q21PhotoCrop().crop(source, torch.zeros_like(mask), '{}')
    else:
        data, work = module.Q21PhotoCrop().crop(source, mask, '{}')
        generated = torch.ones_like(work)
        if bad == 'size':
            generated = generated[:, :-1]
        else:
            generated[0, 0, 0, 0] = float('nan')
        with pytest.raises(ValueError):
            module.Q21PhotoStitch().stitch(data, generated, '{}')
    assert not list(out.rglob('*.png'))


def test_preview_switch_remains_lazy(photo):
    module, _ = photo
    node = module.Q21PhotoReviewSave()
    assert node.check_lazy_status({}, run_edit=False) == []
    assert node.check_lazy_status({}, run_edit=True) == ['edited_result']
    assert node.check_lazy_status({}, run_edit=True, edited_result={}) == []
