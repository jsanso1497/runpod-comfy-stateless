"""CPU-only contracts. These tests do not claim GPU or actual frontend execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from workbench import qwen_mobile as m
from workbench import qwen_mobile_launcher as j


def test_graph_links_and_app_contract():
    m.validate_workflow(m.make_workflow())


def test_checked_in_files_equal_generator():
    assert json.loads((ROOT / m.WORKFLOW_PATH).read_text()) == m.make_workflow()
    assert json.loads((ROOT / m.API_PATH).read_text()) == m.make_api()


def test_native_only_no_enhancer_sam_or_upscaler():
    kinds = {n['type'] for n in m.make_workflow()['nodes']}
    assert kinds == set(m.task_row()['required_nodes'])
    assert not any(any(word in kind for word in ('TextGenerate', 'WBQwen', 'SAM', 'SeedVR', 'Lora')) for kind in kinds)


def test_no_quantized_model_files():
    values = [v for n in m.make_workflow()['nodes'] for v in n['widgets_values'] if isinstance(v, str)]
    weights = [v for v in values if v.endswith('.safetensors')]
    assert set(weights) == {m.MODEL, m.ENCODER, m.VAE}
    assert all('bf16' in v for v in weights)


def test_native_sampling_and_lossless_cache():
    api = m.make_api()
    sampler = api['8']['inputs']
    assert (sampler['steps'], sampler['cfg'], sampler['sampler_name'], sampler['scheduler'], sampler['denoise']) == (25, 1, 'euler', 'simple', 1)
    assert api['7']['inputs']['device'] == 'auto'
    assert api['7']['inputs']['dtype'] == 'default'


def test_native_image_size_latent_not_empty_latent():
    api = m.make_api()
    assert api['8']['inputs']['latent_image'] == ['6', 2]
    assert api['6']['inputs']['images.image_1'] == ['1', 0]
    assert api['6']['inputs']['resolution'] == 2048
    assert api['6']['inputs']['prompt'] == ['2', 0]


def test_one_source_and_one_saved_output():
    graph = m.make_workflow()
    assert sum(n['type'] == 'LoadImage' for n in graph['nodes']) == 1
    assert graph['extra']['linearData']['outputs'] == [10]
    assert graph['extra']['linearMode'] is True
    assert m.make_api()['10']['inputs']['filename_prefix'] == 'Qwen_Mobile/Qwen21'


def test_api_drops_frontend_only_controls():
    api = m.make_api()
    assert 'upload' not in api['1']['inputs']
    assert 'control_after_generate' not in api['8']['inputs']
    assert api['1']['inputs']['image'] == ''


def test_app_bindings_only_resolve_real_native_widgets():
    graph = m.make_workflow()
    assert [(r[0], r[1]) for r in graph['extra']['linearData']['inputs']] == [
        (1, 'image'), (2, 'value'), (6, 'resolution'), (8, 'seed'), (8, 'control_after_generate')]
    assert not graph.get('definitions')


def test_catalog_additive_no_existing_mutations():
    original = {'tasks': [{'id': 'Q01', 'requires': ['qwen', 'sam'], 'variants': []}]}
    saved = copy.deepcopy(original['tasks'][0])
    result = m.apply_catalog('tasks.json', original)
    assert result['tasks'][0] == saved
    assert result['tasks'][-1]['id'] == 'QAPP'
    assert result['tasks'][-1]['requires'] == ['qwen']


@pytest.mark.parametrize('name', ['models.json', 'assets.json', 'release.json'])
def test_other_catalogs_unchanged(name):
    value = {'example': ['unchanged']}
    assert m.apply_catalog(name, value) is value
    assert value == {'example': ['unchanged']}


def test_duplicate_catalog_rejected():
    with pytest.raises(ValueError):
        m.apply_catalog('tasks.json', {'tasks': [m.task_row()]})


@pytest.mark.parametrize('workspace', ['flux', 'h3', 'restoration', 'krea'])
def test_jupyter_never_starts_for_other_families(workspace):
    assert not j.jupyter_enabled({'WB_WORKSPACE': workspace, 'WB_JUPYTER': '1'})


def test_jupyter_default_and_opt_out():
    assert j.jupyter_enabled({'WB_WORKSPACE': 'qwen'})
    assert not j.jupyter_enabled({'WB_WORKSPACE': 'qwen', 'WB_JUPYTER': '0'})
    assert not j.jupyter_enabled({'WB_WORKSPACE': 'qwen', 'WB_APPROVED_ONLY': '1'})


def test_invalid_enable_value_rejected():
    with pytest.raises(ValueError):
        j.jupyter_enabled({'WB_WORKSPACE': 'qwen', 'WB_JUPYTER': 'yes'})


def test_explicit_token_and_password_fallback():
    assert j.resolve_token({'JUPYTER_TOKEN': 'a' * 20, 'WB_PASSWORD': 'b' * 20}) == 'a' * 20
    assert j.resolve_token({'WB_PASSWORD': 'b' * 20}) == 'b' * 20


@pytest.mark.parametrize('value', ['', 'short', ' ' * 20, '{{ RUNPOD_SECRET_comfy_password }}', 'a' * 20 + '\n'])
def test_bad_tokens_rejected_even_with_valid_fallback(value):
    with pytest.raises(ValueError):
        j.resolve_token({'JUPYTER_TOKEN': value, 'WB_PASSWORD': 'b' * 20})


def test_no_secret_in_error_message():
    value = 'unresolved\nvery-private-secret'
    with pytest.raises(ValueError) as exc:
        j.resolve_token({'JUPYTER_TOKEN': value})
    assert value not in str(exc.value)


def test_child_credentials_are_separated(tmp_path):
    env = {k: 'secret' for k in j.SECRET_KEYS}
    env.update({'PATH': '/usr/bin', 'PYTHONPATH': '/opt/workbench/src', 'RUNPOD_SECRET_something': 'secret'})
    child = j.jupyter_environment(env, 'token-only-for-jupyter', tmp_path)
    assert child['JUPYTER_TOKEN'] == 'token-only-for-jupyter'
    assert all(key not in child for key in j.SECRET_KEYS - {'JUPYTER_TOKEN'})
    assert 'RUNPOD_SECRET_something' not in child and 'PYTHONPATH' not in child
    comfy = j.comfy_environment(env)
    assert 'JUPYTER_TOKEN' not in comfy and 'JUPYTER_TOKEN_FILE' not in comfy
    assert comfy['WB_PASSWORD'] == 'secret'


def test_private_log_permissions_and_symlink_rejection(tmp_path):
    p = tmp_path / 'server.log'
    p.touch(mode=0o644)
    with j.private_file(p) as out:
        out.write(b'local test\n')
    assert p.stat().st_mode & 0o777 == 0o600
    link = tmp_path / 'link.log'
    link.symlink_to(p)
    with pytest.raises(ValueError):
        j.private_file(link)


def test_status_contains_no_credentials(tmp_path):
    path = tmp_path / 'status.json'
    j.write_status(path, 'process-running', pid=123)
    assert json.loads(path.read_text()) == {'service': 'jupyterlab', 'state': 'process-running', 'port': 8888, 'pid': 123}
    assert path.stat().st_mode & 0o777 == 0o600


def test_cleanup_terminates_then_escalates(monkeypatch):
    p = MagicMock()
    p.pid = 123
    p.poll.return_value = None
    p.wait.side_effect = [subprocess.TimeoutExpired('mock', 1), 0]
    signals = []
    monkeypatch.setattr(os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    j.stop_process(p, timeout=0.01)
    assert signals == [(123, signal.SIGTERM), (123, signal.SIGKILL)]


def test_jupyter_config_keeps_auth_and_csrf_and_consumes_token(monkeypatch, tmp_path):
    c = SimpleNamespace(IdentityProvider=SimpleNamespace(), ServerApp=SimpleNamespace(), LabApp=SimpleNamespace())
    token = 'test-secret-not-a-real-credential'
    monkeypatch.setenv('JUPYTER_TOKEN', token)
    monkeypatch.setenv('WB_DATA_ROOT', str(tmp_path))
    source = (ROOT / 'deploy/jupyter_workbench_config.py').read_text()
    scope = {'get_config': lambda: c}
    exec(compile(source, 'jupyter_workbench_config.py', 'exec'), scope)
    assert c.IdentityProvider.token == token
    assert 'JUPYTER_TOKEN' not in os.environ
    assert c.ServerApp.disable_check_xsrf is False
    assert c.ServerApp.allow_unauthenticated_access is False
    assert c.ServerApp.port_retries == 0 and c.ServerApp.port == 8888
    assert c.ServerApp.allow_origin == ''
    assert c.ServerApp.cookie_options['secure'] is True
    assert 'token' not in scope


def test_jupyter_config_refuses_empty_token(monkeypatch):
    monkeypatch.delenv('JUPYTER_TOKEN', raising=False)
    with pytest.raises(ValueError):
        exec((ROOT / 'deploy/jupyter_workbench_config.py').read_text(), {'get_config': lambda: SimpleNamespace()})


def test_disabled_jupyter_still_starts_existing_runtime(tmp_path, monkeypatch):
    calls = []
    runtime = MagicMock()
    runtime.poll.return_value = 0
    runtime.returncode = 0
    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return runtime
    monkeypatch.setattr(subprocess, 'Popen', popen)
    result = j.run_supervisor({'WB_DATA_ROOT': str(tmp_path), 'WB_WORKSPACE': 'qwen', 'WB_JUPYTER': '0'})
    assert result == 0
    assert len(calls) == 1
    assert calls[0][0] == [sys.executable, '-m', 'workbench.runtime']
    assert json.loads((tmp_path / 'workbench/jupyter/status.json').read_text())['state'] == 'disabled'


def test_invalid_token_fails_closed_without_stopping_comfy(tmp_path, monkeypatch):
    calls = []
    runtime = MagicMock()
    runtime.poll.return_value = 0
    runtime.returncode = 0
    monkeypatch.setattr(subprocess, 'Popen', lambda args, **kw: calls.append((args, kw)) or runtime)
    result = j.run_supervisor({'WB_DATA_ROOT': str(tmp_path), 'WB_WORKSPACE': 'qwen',
                               'JUPYTER_TOKEN': 'short', 'WB_PASSWORD': 'x' * 20})
    assert result == 0 and len(calls) == 1
    assert calls[0][0][-1] == 'workbench.runtime'
    assert 'JUPYTER_TOKEN' not in calls[0][1]['env']
    assert json.loads((tmp_path / 'workbench/jupyter/status.json').read_text())['state'] == 'unavailable'


def test_docker_installs_jupyter_after_original_software_build():
    text = (ROOT / 'Dockerfile').read_text()
    assert text.index('python deploy/build.py') < text.index('python deploy/build_qwen_mobile.py')
    assert 'EXPOSE 8188 8888' in text
    assert '"workbench.qwen_mobile_launcher"]' in text
    assert 'COPY . ' not in text


def test_catalog_hook_runs_before_qwen_only_filter():
    text = (ROOT / 'src/workbench/config.py').read_text()
    assert text.index('value = apply_qwen_mobile(name, value)') < text.index('return apply_qwen_only(name, value)')
    assert 'JUPYTER_TOKEN' in text
