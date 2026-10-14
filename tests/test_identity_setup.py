"""Offline contracts only. No claim of upstream download or GPU execution."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('identity_setup', Path(__file__).resolve().parents[1] / 'scripts/setup_identity_edits.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def krea_fixture():
    specs = [(53, 'KSampler', [1, 'fixed', 10, 1, 'euler', 'simple', 1]),
             (55, 'UNETLoader', ['krea2_turbo_fp8_scaled.safetensors', 'default']),
             (56, 'CLIPLoader', ['qwen3vl_4b_fp8_scaled.safetensors', 'krea2', 'default']),
             (71, 'LoraLoaderModelOnly', ['Krea2/' + m.KEDIT, 1]),
             (79, 'Krea2EditModelPatch', [4, 1, 'fit']),
             (82, 'EmptySD3LatentImage', [1024, 1024, 1]),
             (84, 'Krea2EditGroundedEncode', ['edit', 768, '']),
             (85, 'Krea2EditGroundedEncode', ['', 768, '']),
             (90, 'LoadImage', ['person.png', 'image']), (92, 'VAEEncode', [])]
    graph = {'nodes': [dict(id=i, type=t, widgets_values=w, mode=4 if i in (90, 92) else 0, inputs=[]) for i, t, w in specs],
             'links': [[24, 82, 0, 53, 3, 'LATENT'], [27, 82, 0, 79, 7, 'LATENT']]}
    by_id = {n['id']: n for n in graph['nodes']}
    by_id[53]['inputs'] = [{'name': 'latent_image', 'link': 24}]
    by_id[79]['inputs'] = [{'name': 'target_latent', 'link': 27}]
    return graph


def qwen_fixture():
    specs = [('UNETLoader', ['qwen_image_2.1_int8_convrot.safetensors', 'default']),
             ('CLIPLoader', ['qwen3vl_8b_int8_convrot.safetensors', 'qwen_image', 'default']),
             ('VAELoader', [Path(m.BASE['qwen'][2]).name]), ('TextEncodeQwenImage21', ['edit', 0]),
             ('KSampler', [1, 'fixed', 25, 1, 'euler', 'simple', 1]),
             ('AUSBOSS_NODES_LoraLoader', [json.dumps([{'name': m.Q1500, 'enabled': True},
                  {'name': m.Q2000, 'enabled': False}, {'name': 'speed-lora', 'enabled': True}])])]
    return {'nodes': [dict(id=i, type=t, widgets_values=w, widgets_values_named={}) for i, (t, w) in enumerate(specs)], 'links': []}


@pytest.mark.parametrize('raw,two', [(False, False), (False, True), (True, False), (True, True)])
def test_krea_presets(raw, two):
    graph = krea_fixture()
    snapshot = copy.deepcopy(graph)
    result = m.adapt_krea(graph, raw, two)
    by_id = {n['id']: n for n in result['nodes']}
    assert graph == snapshot and result['links'] == graph['links']
    assert by_id[53]['widgets_values'][2:4] == ([20, 3] if raw else [10, 1])
    assert by_id[90]['mode'] == by_id[92]['mode'] == (0 if two else 4)
    assert by_id[79]['widgets_values'] == [4, 1, 'fit']
    assert by_id[85]['widgets_values'][0] == ''
    assert by_id[71]['widgets_values'] == [m.KEDIT, 1]
    assert 'fp8_scaled' not in json.dumps(result)
    assert ('raw' if raw else 'turbo') in by_id[55]['widgets_values'][0]


def test_missing_target_latent_rejected():
    graph = krea_fixture()
    graph['links'][1][1] = 999
    with pytest.raises(ValueError, match='same output'):
        m.adapt_krea(graph)


@pytest.mark.parametrize('selected', [m.Q1500, m.Q2000, m.OUTFIT, None])
def test_qwen_selection_and_nested_graphs(selected):
    fixture = qwen_fixture()
    original = copy.deepcopy(fixture)
    graph = {'nodes': [{'id': 100, 'type': 'embedded'}], 'definitions': {'subgraphs': [dict(fixture, id='embedded')]}}
    result = m.adapt_qwen(graph, selected)
    assert fixture == original
    rows = json.loads(next(n for n in m.nodes(result) if n['type'] == 'AUSBOSS_NODES_LoraLoader')['widgets_values'][0])
    assert [r['name'] for r in rows if r['enabled']] == ([selected] if selected else [])
    assert 'embedded' not in m.required_types(result)
    assert 'int8_convrot' not in json.dumps(result)
    sampler = next(n for n in m.nodes(result) if n['type'] == 'KSampler')
    assert sampler['widgets_values'] == [1, 'fixed', 25, 1, 'euler', 'simple', 1]
    loader = next(n for n in m.nodes(result) if n['type'] == 'AUSBOSS_NODES_LoraLoader')
    assert loader['widgets_values_named']['loras'] == loader['widgets_values'][0]


def test_download_integrity_and_reuse(tmp_path, monkeypatch):
    data = b'test payload'
    sha = hashlib.sha256(data).hexdigest()
    calls = []
    def fake_open(*a, **k):
        calls.append(1)
        return io.BytesIO(data)
    monkeypatch.setattr(m, 'urlopen', fake_open)
    target = tmp_path / 'weight'
    m.download('https://example.invalid/test', target, sha)
    m.download('https://example.invalid/test', target, sha)
    assert len(calls) == 1 and target.read_bytes() == data
    with pytest.raises(ValueError, match='different hash'):
        m.download('https://example.invalid/test', target, '0' * 64)
    assert target.read_bytes() == data


def test_bad_download_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'urlopen', lambda *a, **k: io.BytesIO(b'bad'))
    with pytest.raises(ValueError, match='hash mismatch'):
        m.download('https://example.invalid/test', tmp_path / 'weight', '0' * 64)
    assert not list(tmp_path.iterdir())


def test_user_edits_and_symlinks_preserved(tmp_path):
    path = tmp_path / 'workflow.json'
    path.write_bytes(b'user edit')
    dest = m.publish(path, b'factory copy')
    assert dest != path and path.read_bytes() == b'user edit'
    assert m.publish(path, b'factory copy') == dest
    symlink = tmp_path / 'link.json'
    symlink.symlink_to(path)
    with pytest.raises(ValueError, match='symlink'):
        m.publish(symlink, b'new')


@pytest.mark.parametrize('family', ['qwen', 'krea'])
def test_dry_run_no_writes(family, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('WB_WORKSPACE', family)
    assert m.main([family, '--dry-run', '--data-root', str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)['family'] == family
    assert not list(tmp_path.iterdir())


def test_wrong_workspace_rejected(monkeypatch):
    monkeypatch.setenv('WB_WORKSPACE', 'qwen')
    with pytest.raises(SystemExit):
        m.main(['krea', '--dry-run'])


def test_all_sources_pinned():
    for family in ('qwen', 'krea'):
        for repo, rev, filename, sha in m.WEIGHTS[family]:
            assert len(rev) == 40 and len(sha) == 64
            assert '/resolve/' + rev + '/' in m.hf_url(repo, rev, filename)
    with pytest.raises(ValueError):
        m.hf_url('example/model', 'main', 'file')


def test_workbench_nodes_without_git_use_build_provenance(tmp_path, monkeypatch):
    comfy = tmp_path / 'ComfyUI'
    repo, directory, revision = m.PACKS['krea']
    target = comfy / 'custom_nodes' / directory
    target.mkdir(parents=True)
    (target / '__init__.py').write_text('# fixture')
    root = tmp_path / 'workbench'
    root.mkdir()
    (root / 'build-provenance.json').write_text(json.dumps({'external_nodes': {'krea2edit':
        {'directory': directory, 'commit': revision, 'url': f'https://github.com/{repo}.git'}}}))
    monkeypatch.setenv('WB_ROOT', str(root))
    def no_git(*a, **k):
        raise m.subprocess.CalledProcessError(128, 'git')
    monkeypatch.setattr(m.subprocess, 'check_output', no_git)
    assert not m.ensure_pack(comfy, 'krea')
    record = json.loads((root / 'build-provenance.json').read_text())
    record['external_nodes']['krea2edit']['commit'] = '0' * 40
    (root / 'build-provenance.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='not the verified revision'):
        m.ensure_pack(comfy, 'krea')
