"""Qwen-only policy and real native graph projections; no model inference."""
from __future__ import annotations
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from workbench import qwen_only as q
from workbench.graph_audit import audit_graph
from workbench.factory_sync import sync_factory
from workbench.audit_preparation import request_preparation, run_preparation

ROOT = Path(__file__).resolve().parents[1]
KEEP_PATH = 'workflows/qwen/QKEEP1/Qwen21_Keep_One_Person_Protected_Overlay_BF16.json'
BASE_FILES = {'qwen_image_2.1_bf16.safetensors', 'qwen3vl_8b_bf16.safetensors',
              'qwen_image_2.1_vae_bf16.safetensors'}


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv('WB_WORKSPACE', 'qwen')
    monkeypatch.setenv('WB_QWEN_ONLY', '1')
    return q


def native_graph():
    # Real committed QKEEP1 embeds the exact native Qwen sampler/encoder graph.
    return json.loads((ROOT / KEEP_PATH).read_text())


def catalogs():
    tasks = [{'id': 'Q%02d' % i, 'family': 'qwen', 'workspaces': ['qwen'],
              'requires': ['qwen'], 'description': 'Qwen editing',
              'optional_prompt_enhancement': {'default': 'off', 'model': 'helper'},
              'variants': []} for i in range(1, 29)]
    for name in ('QNATIVE', 'QOVERLAY', 'QKEEP1'):
        tasks.append({'id': name, 'family': 'qwen', 'workspaces': ['qwen'],
                      'requires': ['qwen', 'qwen_stock_pe', 'sam'], 'variants': []})
    tasks += [{'id': 'QDEV', 'family': 'qwen', 'workspaces': ['qwen'],
               'requires': ['qwen', 'qwen_identity'], 'variants': []},
              {'id': 'QGS1', 'family': 'qwen', 'workspaces': ['qwen'],
               'requires': ['qwen', 'sam', 'seedvr2'], 'variants': []},
              {'id': 'R01', 'family': 'restoration', 'workspaces': ['qwen', 'restoration'],
               'requires': ['seedvr2'], 'variants': []},
              {'id': 'U10', 'family': 'utility', 'workspaces': ['qwen'],
               'requires': ['ollama'], 'variants': []},
              {'id': 'U11', 'family': 'utility', 'workspaces': ['qwen', 'krea'],
               'requires': ['ollama'], 'variants': []},
              {'id': 'K01', 'family': 'krea', 'workspaces': ['krea'],
               'requires': ['krea'], 'variants': []}]
    tasks += [{'id': name, 'family': 'moss', 'workspaces': ['qwen'],
               'requires': ['moss_image'], 'variants': []} for name in ('QMOSS', 'QMOSS_I2I')]
    assets = [{'id': 'asset-' + g, 'group': g, 'destination': g + '.safetensors'}
              for g in ['qwen', 'sam', 'seedvr2', 'qwen_identity', 'krea',
                        'qwen_pe_i2i', 'qwen_stock_pe', 'moss_image']]
    models = [{'id': 'qwen', 'default_groups': ['qwen', 'ollama', 'moss_image'],
               'toolbox': ['sam', 'seedvr2', 'qwen_pe_i2i', 'qwen_stock_pe', 'ollama', 'moss_image'],
               'local_nodes': ['qwen_native', 'qwen_photo', 'qwen_prompt', 'runtime_compat'],
               'default_tasks': ['Q01'], 'experimental_moss': {}, 'comfy_commit': 'unchanged'},
              {'id': 'krea', 'default_groups': ['krea', 'ollama'], 'toolbox': ['ollama']}]
    return {'tasks.json': {'tasks': tasks}, 'models.json': {'models': models},
            'assets.json': {'assets': assets, 'ollama': {'models': ['large-helper']},
                            'ollama_by_workspace': {'krea': {'models': ['keep-this']}}}}


@pytest.mark.parametrize('flag,workspace,result', [('0', 'qwen', False), ('1', 'qwen', True),
                                                ('1', 'krea', False), ('1', 'flux', False)])
def test_strict_opt_in(monkeypatch, flag, workspace, result):
    monkeypatch.setenv('WB_QWEN_ONLY', flag)
    monkeypatch.setenv('WB_WORKSPACE', workspace)
    assert q.enabled() is result


@pytest.mark.parametrize('flag', ['yes', 'true', '', '01'])
def test_invalid_flags_fail(monkeypatch, flag):
    monkeypatch.setenv('WB_QWEN_ONLY', flag)
    with pytest.raises(ValueError):
        q.enabled()


def test_default_off_is_exact_identity(monkeypatch):
    monkeypatch.delenv('WB_QWEN_ONLY', raising=False)
    graph = native_graph()
    assert q.project_workflow(graph) is graph
    raw = b'{"nodes": []} \n'
    assert q.workflow_bytes(raw) == raw
    for name, value in catalogs().items():
        assert q.apply_catalog(name, value) is value


def test_no_moss_or_helper_groups_and_no_other_family_changes(on):
    source = catalogs()
    expected_krea = copy.deepcopy(source['models.json']['models'][1])
    projected = {n: q.apply_catalog(n, copy.deepcopy(v)) for n, v in source.items()}
    model = projected['models.json']['models'][0]
    assert model['default_groups'] == ['qwen']
    assert model['toolbox'] == ['sam', 'seedvr2']
    assert model['comfy_commit'] == 'unchanged'
    assert model['local_nodes'] == source['models.json']['models'][0]['local_nodes']
    assert projected['models.json']['models'][1] == expected_krea
    active = {t['id']: t for t in projected['tasks.json']['tasks'] if 'qwen' in t['workspaces']}
    assert set(active) >= {'Q%02d' % i for i in range(1, 29)} | {'QNATIVE', 'QOVERLAY', 'QKEEP1', 'QDEV', 'QGS1'}
    assert not (set(active) & q.HIDDEN_TASKS)
    assert all(not (set(t['requires']) & q.EXCLUDED_GROUPS) for t in active.values())
    studio = next(t for t in projected['tasks.json']['tasks'] if t['id'] == 'U11')
    assert studio['workspaces'] == ['krea'] and studio['requires'] == ['ollama']
    assert projected['assets.json']['ollama']['models'] == []
    assert projected['assets.json']['ollama_by_workspace']['krea'] == {'models': ['keep-this']}
    assert not ({a['group'] for a in projected['assets.json']['assets']} & q.EXCLUDED_GROUPS)
    assert {a['group'] for a in projected['assets.json']['assets']} >= {'qwen', 'sam', 'seedvr2', 'qwen_identity'}


def test_direct_prompt_is_exact_with_no_settings_or_image_inspection(on):
    prompt = '<image1> Keep "Literal text" and <lora:example:0.8> unchanged.\nNo rewriting.'
    result, report = q.direct_prompt_result(prompt)
    assert result is prompt and report['effective_prompt'] == prompt
    assert report['enabled'] is False and report['fallback_used'] is False


def test_real_native_graph_has_no_helper_loader_and_preserves_sampling(on):
    graph = native_graph()
    before = copy.deepcopy(graph)
    out = q.project_workflow(graph)
    assert graph == before
    assert audit_graph(out) == []
    old = graph['definitions']['subgraphs'][0]
    new = out['definitions']['subgraphs'][0]
    assert old['id'] != new['id']
    assert len([n for n in new['nodes'] if n['type'] == 'CLIPLoader']) == 1
    assert not ({n['type'] for n in new['nodes']} & {'TextGenerate', 'PreviewAny', 'BatchImagesNode'})
    for kind in ('KSampler', 'QwenImage21Cache', 'UNETLoader', 'VAELoader'):
        a = next(n for n in old['nodes'] if n['type'] == kind)
        b = next(n for n in new['nodes'] if n['type'] == kind)
        assert a.get('widgets_values') == b.get('widgets_values')
    text = json.dumps(out)
    assert all(name in text for name in BASE_FILES)
    assert 'qwen3.5_9b_qwen_image_2.1_pe_i2i' not in text
    encoder = next(n for n in new['nodes'] if n['type'] == 'TextEncodeQwenImage21')
    slot = next(i for i, p in enumerate(encoder['inputs']) if p['name'] == 'prompt')
    link = next(r for r in new['links'] if r['target_id'] == encoder['id'] and r['target_slot'] == slot)
    assert link['origin_id'] == new['inputNode']['id']
    assert new['inputs'][link['origin_slot']]['name'] == 'prompt'
    assert q.project_workflow(out) == out


def test_parent_widget_alignment_and_input_connection_preserved(on):
    original = native_graph()
    output = q.project_workflow(original)
    old = next(n for n in original['nodes'] if n.get('widgets_values_named', {}).get('unet_name'))
    new = next(n for n in output['nodes'] if n.get('widgets_values_named', {}).get('unet_name'))
    for key, value in new['widgets_values_named'].items():
        assert value == old['widgets_values_named'][key]
    assert list(new['widgets_values_named'].values()) == new['widgets_values']
    assert {'switch_1', 'clip_name_1', 'thinking', 'max_length'}.isdisjoint(new['widgets_values_named'])
    for port in new['inputs']:
        old_port = next(p for p in old['inputs'] if p['name'] == port['name'])
        assert port['link'] == old_port['link']
    assert old['inputs'] != new['inputs']


def test_native_stock_preview_branch_is_removed(on):
    graph = native_graph()
    sub = graph['definitions']['subgraphs'][0]
    encoder = next(n for n in sub['nodes'] if n['type'] == 'TextEncodeQwenImage21')
    link = next(r for r in sub['links'] if r['target_id'] == encoder['id'] and r['target_slot'] == 3)
    old_source, old_slot = link['origin_id'], link['origin_slot']
    link['origin_id'], link['origin_slot'] = 9000, 0
    sub['links'].append({'id': 9001, 'origin_id': old_source, 'origin_slot': old_slot,
                         'target_id': 9000, 'target_slot': 0, 'type': 'STRING'})
    sub['nodes'].append({'id': 9000, 'type': 'PreviewAny',
                         'inputs': [{'name': 'source', 'type': '*', 'link': 9001}],
                         'outputs': [{'name': 'STRING', 'type': 'STRING', 'links': [link['id']]}]})
    q._rebuild_subgraph_ports(sub)
    out = q.project_workflow(graph)
    assert not audit_graph(out)
    assert 'PreviewAny' not in {n['type'] for n in out['definitions']['subgraphs'][0]['nodes']}


def test_enabled_legacy_controls_forced_off_without_removing_edit_nodes(on):
    graph = {'nodes': [{'id': 1, 'type': 'WBQwenPromptControl', 'widgets_values': [True, 42, ''],
                        'widgets_values_named': {'enabled': True, 'seed': 42}},
                       {'id': 2, 'type': 'Q21PhotoEncode', 'inputs': []}], 'links': []}
    out = q.project_workflow(graph)
    assert out['nodes'][0]['widgets_values'][0] is False
    assert out['nodes'][0]['widgets_values_named']['enabled'] is False
    assert out['nodes'][1] == graph['nodes'][1]
    assert graph['nodes'][0]['widgets_values'][0] is True


@pytest.mark.parametrize('failure', ['widgets', 'prompt', 'duplicate-prompt'])
def test_unknown_native_layout_stops_instead_of_miswiring(on, failure):
    graph = native_graph()
    sub = graph['definitions']['subgraphs'][0]
    if failure == 'widgets':
        node = next(n for n in graph['nodes'] if n.get('widgets_values_named', {}).get('unet_name'))
        node['widgets_values'].append('unexpected')
    elif failure == 'prompt':
        sub['inputs'][1]['name'] = 'not-the-prompt'
    else:
        sub['inputs'].append(copy.deepcopy(sub['inputs'][1]))
    with pytest.raises(ValueError):
        q.project_workflow(graph)


def test_factory_and_served_bytes_match_without_altering_sources(on, tmp_path):
    source = tmp_path / 'source'; data = tmp_path / 'data'
    path = source / KEEP_PATH
    path.parent.mkdir(parents=True)
    raw = (ROOT / KEEP_PATH).read_bytes()
    path.write_bytes(raw)
    tasks = [{'id': 'QKEEP1', 'workspaces': ['qwen'], 'variants': [{'file': KEEP_PATH}]}]
    report = sync_factory(source, data, tasks, 'qwen')
    result = data / report['workflows'][0]['path']
    assert result.read_bytes() == q.workflow_bytes(raw)
    assert json.loads(result.read_bytes()) == q.load_workflow(path)
    assert path.read_bytes() == raw
    result.write_text('{"user_edit":true}')
    second = sync_factory(source, data, tasks, 'qwen')
    assert (data / second['workflows'][0]['path']).read_bytes() == q.workflow_bytes(raw)
    assert result.read_text() == '{"user_edit":true}'


def ns_for_requests():
    source = {name: q.apply_catalog(name, value) for name, value in catalogs().items()}
    return {'read_catalog': lambda name: source[name],
            'model_config': lambda: source['models.json']['models'][0], 'workspace': lambda: 'qwen'}


@pytest.mark.parametrize('data', [{'tasks': ['QMOSS']}, {'tasks': ['QMOSS_I2I']},
                                   {'features': ['moss_image']}, {'features': ['ollama']},
                                   {'features': ['qwen_stock_pe']}, {'features': ['qwen_pe_i2i']},
                                   {'tasks': ['U10']}, {'tasks': ['U11']}])
def test_requests_cannot_prepare_excluded_models(on, data):
    runtime = SimpleNamespace(job=None, prepare=AsyncMock())
    request = SimpleNamespace(json=AsyncMock(return_value=data))
    with pytest.raises(ValueError):
        asyncio.run(request_preparation(runtime, request, ns_for_requests()))
    runtime.prepare.assert_not_called()


def test_qwen_request_does_not_reintroduce_optional_helpers(on):
    runtime = SimpleNamespace(job=None, prepare=AsyncMock())
    request = SimpleNamespace(json=AsyncMock(return_value={'tasks': ['QKEEP1', 'QNATIVE', 'QDEV'], 'private': True}))
    async def execute():
        response = await request_preparation(runtime, request, ns_for_requests())
        await runtime.job
        return json.loads(response.body)
    result = asyncio.run(execute())
    assert result['groups'] == ['qwen', 'qwen_identity', 'sam']
    runtime.prepare.assert_awaited_once_with(['qwen', 'qwen_identity', 'sam'], True)


def test_preparation_does_not_start_or_contact_ollama(on, tmp_path):
    runtime = SimpleNamespace(job_state={}, session=Mock(), start_ollama=AsyncMock(side_effect=AssertionError('Ollama called')))
    standard, private = Mock(return_value={'failures': []}), Mock(return_value={'failures': []})
    def atomic(path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(value))
    ns = {**ns_for_requests(), 'state_root': lambda: tmp_path, 'atomic_json': atomic,
          'prepare_standard': standard, 'prepare_private': private}
    groups = ['qwen', 'qwen_identity', 'sam', 'seedvr2']
    asyncio.run(run_preparation(runtime, groups, True, ns))
    assert runtime.job_state['state'] == 'finished' and runtime.job_state['failures'] == 0
    standard.assert_called_once_with(groups)
    private.assert_called_once()
    runtime.start_ollama.assert_not_called()
    runtime.session.post.assert_not_called()


def test_each_installed_qwen_workflow_projects_cleanly(on):
    # Runs over all Qwen and developer graphs present in the checkout.
    # Full CI builds QNATIVE/QOVERLAY and the developer workflows before pytest.
    paths = list((ROOT / 'workflows/qwen').rglob('*.json'))
    paths += list((ROOT / 'workflows/developer/qwen').rglob('*.json'))
    assert paths
    for path in paths:
        graph = q.load_workflow(path)
        assert not audit_graph(graph, str(path)), path
        for sub in graph.get('definitions', {}).get('subgraphs', []):
            if any(n['type'] == 'TextEncodeQwenImage21' for n in sub['nodes']):
                assert not any(n['type'] == 'TextGenerate' for n in sub['nodes']), path
