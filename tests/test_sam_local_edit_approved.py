"""CPU/static contracts for the approved workflow; no inference or Docker build."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from workbench import sam_local_edit_profile as profile
from workbench.sam_local_edit_entrypoint import configure_minimal_environment

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('approved_graph_validator', ROOT / 'scripts/validate_sam_local_edit.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def graph():
    return json.loads((ROOT / profile.WORKFLOW_PATH).read_text())


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv('WB_APPROVED_ONLY', '1')
    monkeypatch.setenv('WB_WORKSPACE', 'qwen')


@pytest.fixture
def catalogs(tmp_path):
    folder = tmp_path / 'catalog'
    folder.mkdir()
    destinations = sorted(profile.EXPECTED_ASSETS)
    assets = [{'id': str(i), 'group': 'sam' if p.startswith('checkpoints/') else 'qwen',
               'destination': p, 'sha256': 'test-preserve-this-value', 'revision': 'preserve-revision'}
              for i, p in enumerate(destinations)]
    assets += [{'id': 'extra-' + group, 'group': group, 'destination': group + '/unused'}
               for group in ('moss_image', 'qwen_stock_pe', 'qwen_pe_i2i', 'ollama', 'seedvr2', 'krea')]
    (folder / 'assets.json').write_text(json.dumps({'assets': assets, 'ollama': {'models': ['unused']}}))
    (folder / 'models.json').write_text(json.dumps({'models': [
        {'id': 'qwen', 'precision': 'BF16', 'comfy_commit': 'old', 'toolbox': ['seedvr2'],
         'local_nodes': ['extra'], 'default_tasks': ['Q01'], 'external_nodes': ['ausboss']},
        {'id': 'krea', 'default_groups': ['krea']}], 'external_nodes': {'ausboss': {}}}))
    (folder / 'approved.workflows.json').write_bytes((ROOT / 'catalog/approved.workflows.json').read_bytes())
    destination = tmp_path / profile.WORKFLOW_PATH
    destination.parent.mkdir(parents=True)
    destination.write_bytes((ROOT / profile.WORKFLOW_PATH).read_bytes())
    return tmp_path


def test_original_v2_behavior_preserved():
    reference = json.loads((ROOT / 'tests/fixtures/sam_local_edit/approved_v2_source.json').read_text())
    assert validator.normalized(graph()) == validator.normalized(reference)
    validator.validate_graph(graph())


def test_no_private_upload_in_public_workflow():
    g = graph()
    node = next(n for n in g['nodes'] if n['id'] == 1)
    assert node['widgets_values'][0] == ''
    assert node['widgets_values_named']['image'] == ''


def test_approval_is_not_gpu_certification():
    info = graph()['extra']['workbench']
    assert info['approval'] == 'user-approved'
    assert info['live_gpu_validated'] is False


@pytest.mark.parametrize('node,key,value', [(6, 'steps', 4), (6, 'cfg', 7), (9, 'run_edit', False)])
def test_critical_setting_changes_fail(node, key, value):
    g = graph()
    target = next(n for n in g['nodes'] if n['id'] == node)
    target['widgets_values_named'][key] = value
    target['widgets_values'] = list(target['widgets_values_named'].values())
    with pytest.raises(ValueError): validator.validate_graph(g)


@pytest.mark.parametrize('workspace,flag,expected', [('qwen','1',True),('qwen','0',False),('krea','1',False)])
def test_profile_opt_in(monkeypatch, workspace, flag, expected):
    monkeypatch.setenv('WB_WORKSPACE', workspace)
    monkeypatch.setenv('WB_APPROVED_ONLY', flag)
    assert profile.enabled() is expected


@pytest.mark.parametrize('bad', ['true', 'yes', '01', ''])
def test_invalid_profile_flag(monkeypatch, bad):
    monkeypatch.setenv('WB_APPROVED_ONLY', bad)
    with pytest.raises(ValueError): profile.enabled()


def test_disabled_profile_does_not_read_files(monkeypatch, tmp_path):
    monkeypatch.setenv('WB_APPROVED_ONLY', '0')
    assert profile.catalog_override('tasks.json', tmp_path) is None


def test_only_approved_task(enabled, catalogs):
    tasks = profile.catalog_override('tasks.json', catalogs)['tasks']
    assert [t['id'] for t in tasks] == ['QLOCAL']
    assert tasks[0]['approved'] and tasks[0]['sort_order'] == 0
    assert tasks[0]['title'].startswith('00 | SAM-Guided Local Edit')


def test_assets_are_unmodified_and_only_qwen_sam(enabled, catalogs):
    original = json.loads((catalogs / 'catalog/assets.json').read_text())
    result = profile.catalog_override('assets.json', catalogs)
    assert result['assets'] == [a for a in original['assets'] if a['group'] in {'qwen','sam'}]
    assert len(result['assets']) == 4
    assert result['ollama']['models'] == []
    assert result['ollama']['estimated_bytes'] == 0


def test_extra_qwen_asset_fails_closed(enabled, catalogs):
    p = catalogs / 'catalog/assets.json'
    data = json.loads(p.read_text())
    data['assets'].append({'group':'qwen','destination':'extra.safetensors'})
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError): profile.catalog_override('assets.json', catalogs)


def test_no_external_packs_or_extra_workspaces(enabled, catalogs):
    catalog = profile.catalog_override('models.json', catalogs)
    assert len(catalog['models']) == 1
    model = catalog['models'][0]
    assert model['local_nodes'] == ['qwen_photo', 'runtime_compat', 'sam_local_edit_ui']
    assert model['external_nodes'] == [] and catalog['external_nodes'] == {}
    assert model['default_tasks'] == ['QLOCAL']
    assert model['default_groups'] == ['qwen','sam']
    assert model['toolbox'] == []
    assert model['precision'] == 'BF16'


def test_approved_factory_is_first_named_folder_and_idempotent(enabled, catalogs, tmp_path):
    data = tmp_path / 'volume'
    tasks = profile.approved_tasks(catalogs)
    a = profile.sync_approved(catalogs, data, tasks, 'qwen')
    b = profile.sync_approved(catalogs, data, tasks, 'qwen')
    assert a == b
    assert a['workflows'][0]['path'] == 'comfy-user/default/workflows/00_Approved/00_SAM_Guided_Local_Edit.json'
    assert (data / a['workflows'][0]['path']).read_bytes() == (catalogs / profile.WORKFLOW_PATH).read_bytes()


def test_user_workflow_edits_and_old_workflows_survive(enabled, catalogs, tmp_path):
    data = tmp_path / 'volume'
    tasks = profile.approved_tasks(catalogs)
    first = profile.sync_approved(catalogs, data, tasks, 'qwen')
    existing = data / first['workflows'][0]['path']
    existing.write_text('{"user_edit":true}')
    old = data / 'comfy-user/default/workflows/Old_workflow.json'
    old.write_text('{"old":true}')
    new = profile.sync_approved(catalogs, data, tasks, 'qwen')
    assert existing.read_text() == '{"user_edit":true}' and old.exists()
    assert '__factory_' in new['workflows'][0]['path']


def test_factory_refuses_symlink_destination(enabled, catalogs, tmp_path):
    data, outside = tmp_path / 'volume', tmp_path / 'outside'
    outside.mkdir()
    data.mkdir()
    (data / 'comfy-user').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError): profile.sync_approved(catalogs, data, profile.approved_tasks(catalogs), 'qwen')


def test_unapproved_tasks_rejected_by_factory(enabled, catalogs, tmp_path):
    with pytest.raises(ValueError):
        profile.sync_approved(catalogs, tmp_path / 'volume', [{'id':'Q01','workspaces':['qwen']}], 'qwen')


def test_sanitizes_inherited_extra_library_variables(monkeypatch):
    defaults = {'WB_WORKSPACE':'qwen','WB_APPROVED_ONLY':'1','WB_QWEN_ONLY':'1','WB_TASKS':'QLOCAL','WB_TOOLBOX':''}
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)
    for key in ('WB_LORA_URLS','WB_LORA_FILE','WB_CHECKPOINTS','WB_CHECKPOINT_FILE','CIVITAI_TOKEN'):
        monkeypatch.setenv(key, 'unused-private-value')
    monkeypatch.setenv('HF_TOKEN', 'preserve-private-value')
    configure_minimal_environment()
    import os
    assert os.environ['WB_TASKS'] == 'QLOCAL' and os.environ['WB_TOOLBOX'] == ''
    assert os.environ['HF_TOKEN'] == 'preserve-private-value'
    assert 'WB_LORA_URLS' not in os.environ and 'CIVITAI_TOKEN' not in os.environ


@pytest.mark.parametrize('key,value', [('WB_WORKSPACE','krea'),('WB_TASKS','Q01'),('WB_TOOLBOX','ollama'),('WB_QWEN_ONLY','0')])
def test_minimal_entrypoint_rejects_extras(monkeypatch, key, value):
    defaults = {'WB_WORKSPACE':'qwen','WB_APPROVED_ONLY':'1','WB_QWEN_ONLY':'1','WB_TASKS':'QLOCAL','WB_TOOLBOX':''}
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(key,value)
    with pytest.raises(ValueError): configure_minimal_environment()


def test_deployment_has_no_optional_installer_or_model_download():
    docker = (ROOT / 'Dockerfile.sam-local-edit').read_text()
    build = (ROOT / 'deploy/build_sam_local_edit.py').read_text()
    assert 'WB_APPROVED_ONLY=1' in docker and 'WB_QWEN_ONLY=1' in docker
    assert 'ollama/releases' not in docker and 'curl' not in build
    assert 'ComfyUI-SeedVR2' not in build and 'ComfyUI-AusBoss' not in build
    assert 'prepare_standard(' not in build and 'hf_hub_download(' not in build


def test_ci_uses_dedicated_dockerfile_and_tag():
    workflow = (ROOT / '.github/workflows/build-sam-local-edit.yml').read_text()
    assert 'file: Dockerfile.sam-local-edit' in workflow
    assert ':qwen-local-edit-v1' in workflow
    assert 'uses: ./.github/workflows/ci.yml' in workflow


def test_full_validation_with_catalog_fixtures(enabled, catalogs):
    report = validator.run(catalogs)
    assert report['graph_connections'] == 31 and report['model_assets'] == 4
