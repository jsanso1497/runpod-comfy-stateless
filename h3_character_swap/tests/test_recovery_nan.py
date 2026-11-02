"""Cache-fingerprint export regressions. No H3 GPU inference is exercised."""
import copy
import json
import math
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from swapkit import recovery


@pytest.fixture(autouse=True)
def metadata_enabled(monkeypatch):
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA', '1')


def prompt(marker):
    return {'start': {'class_type': 'StartLoop',
                     'inputs': {'mode': 'simple', 'mode.num_iterations': 1,
                                'cache_iterations': False},
                     'is_changed': marker},
            'noise': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': 904239}}}


def strict_load(text):
    def bad(value):
        raise AssertionError('Invalid JSON constant: ' + value)
    return json.loads(text, parse_constant=bad)


@pytest.mark.parametrize('marker', [float('nan'), [float('nan')],
                                   float('inf'), [float('-inf')], None, [False]])
def test_export_omits_only_node_cache_marker(marker):
    source = prompt(marker)
    workflow = {'nodes': [{'id': 'start', 'type': 'StartLoop',
                          'widgets_values': ['simple', 1, False]}], 'links': []}
    saved = recovery.media_metadata(source, {'workflow': workflow})
    assert 'is_changed' not in saved['prompt']['start']
    assert saved['prompt']['start']['inputs'] == source['start']['inputs']
    assert saved['prompt']['noise'] == source['noise']
    assert saved['workflow'] == workflow
    assert saved['workflow'] is not workflow
    assert source['start']['is_changed'] is marker
    assert strict_load(recovery.json_text(saved)) == saved


def test_nested_real_input_named_is_changed_is_preserved():
    source = prompt(float('nan'))
    source['start']['inputs']['is_changed'] = 'this is a real input'
    saved = recovery.media_metadata(source)
    assert saved['prompt']['start']['inputs']['is_changed'] == 'this is a real input'


def test_graph_metadata_and_loop_connections_remain_unchanged():
    source = prompt([float('nan')])
    source['start'].update(_loop_body=['noise'], _loop_end='end',
                           _meta={'title': 'Native loop - one H3 window'})
    before = copy.deepcopy(source['start']['inputs'])
    saved = recovery.media_metadata(source)
    for key in ('_loop_body', '_loop_end', '_meta'):
        assert saved['prompt']['start'][key] == source['start'][key]
    assert source['start']['inputs'] == before
    assert math.isnan(source['start']['is_changed'][0])
    assert saved['prompt']['start']['inputs'] is not source['start']['inputs']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_invalid_numeric_generation_input_is_not_silently_changed(value):
    source = prompt(float('nan'))
    source['noise']['inputs']['noise_seed'] = value
    with pytest.raises(ValueError, match='Out of range float'):
        recovery.media_metadata(source)
    assert source['noise']['inputs']['noise_seed'] is value


def test_invalid_editable_canvas_stays_an_error():
    with pytest.raises(ValueError, match='Out of range float'):
        recovery.media_metadata(prompt(float('nan')), {'workflow': {
            'nodes': [{'id': 1, 'widgets_values': [float('nan')]}]}})


def test_disabled_metadata_remains_disabled(monkeypatch):
    monkeypatch.setenv('H3_SWAP_SAVE_METADATA', '0')
    assert recovery.media_metadata(prompt(float('nan'))) == {}


def test_sidecars_are_strictly_loadable(tmp_path):
    saved = recovery.media_metadata(prompt([float('nan')]),
                                    {'workflow': {'nodes': [], 'links': []}})
    media = tmp_path / 'result.mp4'
    recovery.sidecars(media, saved)
    assert strict_load((tmp_path / 'result.mp4.api.json').read_text()) == saved['prompt']
    assert strict_load((tmp_path / 'result.mp4.workflow.json').read_text()) == saved['workflow']


def test_journal_retains_request_and_writes_finite_records(tmp_path, monkeypatch):
    monkeypatch.setenv('H3_SWAP_DATA', str(tmp_path))
    source = prompt(float('nan'))
    source['noise']['class_type'] = 'MiniMaxH3ReferenceToVideo'
    data = {'prompt': source, 'extra_data': {'extra_pnginfo': {'workflow': {'nodes': []}}}}
    assert recovery.journal_request(data) is data
    assert math.isnan(source['start']['is_changed'])
    files = list((tmp_path / 'output').rglob('*.api.json'))
    assert len(files) == 1
    assert 'is_changed' not in strict_load(files[0].read_text())['start']


def test_decoded_preview_with_loop_nan_returns_same_pixels(monkeypatch):
    torch = pytest.importorskip('torch')
    from swapkit.preview_nodes import H3SwapWindowPreview
    received = []

    def save(images, **kwargs):
        strict_load(recovery.json_text(kwargs.get('prompt')))
        received.append(kwargs)
        return {'ui': {'images': [{'filename': 'test.png', 'type': 'temp', 'subfolder': ''}]}}

    nodes = ModuleType('nodes')
    nodes.PreviewImage = lambda: SimpleNamespace(save_images=save)
    comfy = ModuleType('comfy')
    mm = ModuleType('comfy.model_management')
    utils = ModuleType('comfy.utils')
    mm.throw_exception_if_processing_interrupted = Mock()
    utils.ProgressBar = lambda total: SimpleNamespace(update_absolute=Mock())
    comfy.model_management = mm
    comfy.utils = utils
    for name, module in [('nodes', nodes), ('comfy', comfy),
                         ('comfy.model_management', mm), ('comfy.utils', utils)]:
        monkeypatch.setitem(sys.modules, name, module)
    images = torch.linspace(0, 1, 7 * 16 * 16 * 3).reshape(7, 16, 16, 3)
    before = images.clone()
    source = prompt([float('nan')])
    result = H3SwapWindowPreview().inspect(images, hold_seconds=0,
                   prompt=source, extra_pnginfo={'workflow': {'nodes': []}})
    assert result['result'][0] is images
    assert torch.equal(images, before)
    assert math.isnan(source['start']['is_changed'][0])
    assert 'is_changed' not in received[0]['prompt']['start']


def test_preview_cancellation_still_raises(monkeypatch):
    torch = pytest.importorskip('torch')
    from swapkit.preview_nodes import H3SwapWindowPreview
    nodes = ModuleType('nodes')
    nodes.PreviewImage = lambda: SimpleNamespace(save_images=Mock())
    comfy = ModuleType('comfy')
    mm = ModuleType('comfy.model_management')
    utils = ModuleType('comfy.utils')
    mm.throw_exception_if_processing_interrupted = Mock(side_effect=InterruptedError('cancel'))
    comfy.model_management, comfy.utils = mm, utils
    for name, module in [('nodes', nodes), ('comfy', comfy),
                         ('comfy.model_management', mm), ('comfy.utils', utils)]:
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(InterruptedError, match='cancel'):
        H3SwapWindowPreview().inspect(torch.zeros(1, 16, 16, 3), prompt=prompt(float('nan')))
