"""CPU mechanics only. Ollama replies are fixtures, never claimed model results."""
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from PIL import Image
from swapkit import akatz_analysis as a, analysis_options as opt, prompts, media, graphs


def evidence():
    return {'face_hair': 'Visible face and curly dark hair.',
            'body_proportions': 'Visible shoulders, torso, arms and legs.',
            'wardrobe': 'UNIQUE REFERENCE OUTFIT.',
            'source_scene': 'UNIQUE ROOM DESCRIPTION.',
            'action_timeline': 'S01 at +0.000s: UNIQUE ACTION. S16 at +5.125s: hand at counter.',
            'camera_lighting': 'UNIQUE CAMERA DESCRIPTION.',
            'other_people_objects': 'Another person holds a cup.',
            'uncertainties': 'UNIQUE HUMAN REVIEW ONLY.'}


def test_defaults_are_long_but_not_unbounded():
    options = opt.AnalysisOptions().validate()
    assert options.max_output_tokens == 16384
    assert options.context_tokens == 32768
    assert options.sampled_frames == 16
    assert options.max_output_tokens > 9 * 1800
    assert opt.AnalysisOptions(max_output_tokens=32768, context_tokens=65536, sampled_frames=32).validate()


@pytest.mark.parametrize('kwargs', [
    {'max_output_tokens': -1}, {'max_output_tokens': True}, {'max_output_tokens': 40000},
    {'context_tokens': 262144}, {'context_tokens': 32768.0}, {'sampled_frames': 33},
    {'sampled_frames': 7}, {'timeout_seconds': 0}, {'detail_level': 'fake'},
    {'reference_layout': 'fake'}, {'context_tokens': 16384}, {'sampled_frames': 32},
    {'max_output_tokens': 32768},
])
def test_invalid_or_insufficient_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError): opt.AnalysisOptions(**kwargs).validate()


def test_large_field_and_prompt_preserved_not_silently_trimmed():
    obj = evidence(); obj['action_timeline'] = 'observed visible motion ' * 1700
    parsed = a.parse_analysis(json.dumps(obj))
    clip, chain = a.build_prompts(parsed, 'the left performer', 'Reference clothing', opt.LAYOUTS[0])
    assert len(clip) > 20000
    assert obj['action_timeline'].strip() in clip
    assert a.parse_analysis(json.dumps(parsed)) == parsed
    assert len(clip) < opt.MAX_PROMPT_CHARS


@pytest.mark.parametrize('case', ['missing', 'extra', 'tag', 'empty', 'long', 'wrong_type'])
def test_schema_rejects_invalid_analysis(case):
    obj = evidence()
    if case == 'missing': del obj['face_hair']
    elif case == 'extra': obj['bfs_prompt'] = 'no'
    elif case == 'tag': obj['wardrobe'] = 'Use <Picture 9>'
    elif case == 'empty': obj['wardrobe'] = ' '
    elif case == 'long': obj['wardrobe'] = 'a' * 64001
    else: obj['wardrobe'] = []
    with pytest.raises(ValueError): a.parse_analysis(json.dumps(obj))


def test_chain_prompt_does_not_replay_sample_timeline_camera_or_scene():
    clip, chain = a.build_prompts(evidence(), 'person in blue', 'Reference clothing', opt.LAYOUTS[0])
    for text in ('UNIQUE ACTION', 'UNIQUE ROOM', 'UNIQUE CAMERA'):
        assert text in clip and text not in chain
    assert 'UNIQUE HUMAN REVIEW' not in clip + chain
    assert 'UNIQUE REFERENCE OUTFIT' in clip and 'UNIQUE REFERENCE OUTFIT' in chain
    assert '<Picture 1>' in clip and '<Video 1>' in clip
    assert '<Picture 2>' not in clip


def test_source_clothing_mode_does_not_transfer_reference_outfit():
    clip, chain = a.build_prompts(evidence(), 'person in blue', 'Source clothing (experimental)', opt.LAYOUTS[0])
    assert 'UNIQUE REFERENCE OUTFIT' not in clip + chain
    assert "retain the selected source performer's clothing" in chain


def test_style_sheet_is_explicitly_one_character():
    clip, chain = a.build_prompts(evidence(), 'person', 'Reference clothing', opt.LAYOUTS[1])
    assert 'two-panel' in clip and 'ONE replacement character' in chain


def test_all_samples_visible_on_contact_sheet():
    for count in (8, 16, 24, 32):
        sheet = prompts.contact_sheet([Image.new('RGB', (30, 50))] * count, list(range(count)))
        assert sheet.size == (1536, math.ceil(count / 4) * 284)


def test_cache_separates_shape_digest_output_budget_and_sampler_settings(tmp_path):
    source = tmp_path / 'source.mp4'; source.write_bytes(b'test')
    def key(options=None, shape=(12, 6, 3), digest='one', trim=0):
        return a.cache_key(source, b'same pixels', shape, 0, 124, trim, 'person',
                           'Reference clothing', 'local', digest, options or opt.AnalysisOptions())
    keys = [key(), key(shape=(6, 12, 3)), key(digest='two'), key(trim=6),
            key(opt.AnalysisOptions(max_output_tokens=8192)),
            key(opt.AnalysisOptions(sampled_frames=8)),
            key(opt.AnalysisOptions(detail_level='Detailed'))]
    assert len(set(keys)) == len(keys)


class Reply:
    def __init__(self, obj=None, lines=None): self.obj, self.lines = obj, lines
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def json(self): return self.obj
    def iter_lines(self, **kwargs):
        for row in self.lines: yield json.dumps(row).encode()


class FakeOllama:
    def __init__(self, text=None, reason='stop', cleanup_fails=False):
        self.text = json.dumps(evidence()) if text is None else text
        self.reason, self.cleanup_fails, self.calls = reason, cleanup_fails, []
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith('/api/show'):
            return Reply({'capabilities': ['vision'], 'model_info': {'qwen.context_length': 262144}})
        if url.endswith('/api/generate'):
            if self.cleanup_fails: raise RuntimeError('cleanup fixture')
            return Reply({})
        return Reply(lines=[{'message': {'content': self.text[:50]}, 'done': False},
                            {'message': {'content': self.text[50:]}, 'done': True,
                             'done_reason': self.reason, 'eval_count': 6000, 'prompt_eval_count': 8000}])
    def get(self, url, **kwargs):
        if url.endswith('/api/tags'): return Reply({'models': [{'name': 'qwen3-vl:8b', 'digest': 'abc'}]})
        return Reply({'models': []})


def run_query(monkeypatch, fake, **kwargs):
    monkeypatch.setattr(prompts, 'session', lambda: fake)
    return a.query([Image.new('RGB', (40, 40))] * 17,
                   [i * 123 / 15 / 24 for i in range(16)], 'person', opt.AnalysisOptions(), **kwargs)


def test_real_request_schema_budget_metrics_and_unloading(monkeypatch):
    fake = FakeOllama(); obj, stats = run_query(monkeypatch, fake)
    payload = next(kw['json'] for url, kw in fake.calls if url.endswith('/api/chat'))
    assert payload['options']['num_predict'] == 16384
    assert payload['options']['num_ctx'] == 32768
    assert payload['think'] is False and payload['keep_alive'] == 0
    assert payload['format'] == a.SCHEMA
    assert len(payload['messages'][1]['images']) == 17
    assert stats['eval_count'] == 6000 and stats['done_reason'] == 'stop'
    assert fake.calls[-1][0].endswith('/api/generate')
    assert fake.calls[-1][1]['json']['keep_alive'] == 0


def test_partial_output_retained_but_not_accepted(monkeypatch, tmp_path):
    fake = FakeOllama(text='{"face_hair": "unfinished', reason='length')
    with pytest.raises(ValueError, match='output/context limit'):
        run_query(monkeypatch, fake, diagnostic_prefix=tmp_path / 'draft')
    assert (tmp_path / 'draft_partial.txt').read_text() == fake.text
    report = json.loads((tmp_path / 'draft_failure.json').read_text())
    assert report['used_as_generation_prompt'] is False
    assert fake.calls[-1][0].endswith('/api/generate')


def test_cleanup_error_does_not_mask_original_failure(monkeypatch):
    with pytest.raises(ValueError, match='output/context limit') as caught:
        run_query(monkeypatch, FakeOllama(reason='length', cleanup_fails=True))
    assert 'cleanup' in ' '.join(caught.value.__notes__)


def test_successful_analysis_fails_closed_if_unload_fails(monkeypatch):
    with pytest.raises(RuntimeError, match='cleanup failed'):
        run_query(monkeypatch, FakeOllama(cleanup_fails=True))


def test_cancel_still_unloads(monkeypatch):
    def cancel(): raise InterruptedError('user canceled')
    fake = FakeOllama()
    with pytest.raises(InterruptedError): run_query(monkeypatch, fake, check_interrupt=cancel)
    assert fake.calls[-1][0].endswith('/api/generate')


def test_live_digest_not_stale_boot_receipt(monkeypatch):
    monkeypatch.setattr(prompts, 'session', lambda: FakeOllama())
    assert a.current_model_digest('qwen3-vl:8b') == 'abc'
    with pytest.raises(ValueError, match='not installed'): a.current_model_digest('other:8b')


def test_new_factory_is_akatz_only_and_paired():
    canvas, api = graphs.prompt_studio()
    assert graphs.validate_graph(canvas)
    node = next(n for n in api.values() if n['class_type'] == 'H3SwapAnalyzeAkatz')
    assert node['inputs']['max_output_tokens'] == 16384
    assert not any('bfs' in n['_meta']['title'].lower() for n in api.values())
    assert not {'CLIPLoader', 'UNETLoader', 'SamplerCustomAdvanced'} & {n['class_type'] for n in api.values()}
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / 'workflows/00_Analyze_Clip_Build_Prompts.api.json').read_text())
    assert saved == api


def test_force_run_enters_disk_cache_instead_of_comfy_execution_cache():
    from swapkit.analysis_nodes import H3SwapAnalyzeAkatz
    assert math.isnan(H3SwapAnalyzeAkatz.IS_CHANGED(reuse_cached_draft=False))
    assert math.isnan(H3SwapAnalyzeAkatz.IS_CHANGED(reuse_cached_draft=True))


def test_analysis_and_chain_cannot_hold_same_lock(monkeypatch, tmp_path):
    import fcntl
    from swapkit import analysis_nodes as n
    monkeypatch.setattr(n, 'data_root', lambda: tmp_path)
    folder = tmp_path / 'output/h3_character_swap/disk_chain_v2'; folder.mkdir(parents=True)
    with (folder / '.controller.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='chain is running'):
            with n.analysis_lock(): pass
        fcntl.flock(handle, fcntl.LOCK_UN)
    with n.analysis_lock(): pass


def test_new_chain_accepts_long_prompt_unchanged(tmp_path):
    from swapkit import disk_chain as d
    inputs, outputs = tmp_path / 'input', tmp_path / 'output'
    inputs.mkdir(); outputs.mkdir()
    source = inputs / 'test.mp4'; source.write_bytes(b'no decode needed for create_run')
    reference = inputs / 'test.png'; Image.new('RGB', (32, 32)).save(reference)
    text = 'Keep the specified character and follow the current source. ' * 600
    assert len(text) > 20000
    run = d.create_run(source=source, reference=reference, input_root=inputs,
                       output_root=outputs, run_name='long_prompt', prompt=text)
    assert json.loads((run / 'config.json').read_text())['prompt'] == text
    with pytest.raises(d.ChainError, match='never silently truncated'):
        d.create_run(source=source, reference=reference, input_root=inputs,
                     output_root=outputs, run_name='too_long', prompt='x' * (opt.MAX_PROMPT_CHARS + 1))


def test_no_generation_model_or_history_recipe_change():
    root = Path(__file__).resolve().parents[1]
    graph = json.loads((root / 'swapkit/data/akatz_disk_base.api.json').read_text())
    assert graph['adapter']['inputs']['lora_name'] == 'h3_character_swap_v1.5.safetensors'
    assert graph['schedule']['inputs']['steps'] == 8
    assert graph['plan']['inputs']['history_frames'] == '18'


def test_actual_sampling_16_frames_and_anamorphic_display(tmp_path):
    import shutil
    import subprocess
    if not shutil.which('ffmpeg'): pytest.skip('FFmpeg missing')
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'testsrc2=size=120x96:rate=24:duration=6', '-vf', 'setsar=2', '-c:v', 'libx264',
        '-preset', 'ultrafast', '-threads', '2', str(source)], check=True)
    images, times, report = media.sample_frames(source, frames=124, sample_count=16)
    assert len(images) == 16 and len(times) == 16
    assert times[0] == 0 and times[-1] == 123/24
    assert images[0].size == (240, 96)
    assert report['audio_analyzed'] is False


def test_bfs_hidden_from_default_factory_but_legacy_files_not_deleted(tmp_path, monkeypatch):
    from swapkit import runtime
    root = tmp_path / 'kit'; (root / 'workflows').mkdir(parents=True)
    for name in ('00_Analyze.json', '03_BFS_V11_Guide_Comparison.json'):
        (root / 'workflows' / name).write_text('{}')
    monkeypatch.setattr(runtime, 'ROOT', root)
    monkeypatch.setenv('H3_SWAP_VARIANTS', 'akatz')
    folder = runtime.sync_workflows(tmp_path / 'workspace')
    assert (folder / '00_Analyze.json').is_file()
    assert not (folder / '03_BFS_V11_Guide_Comparison.json').exists()
    assert (root / 'workflows/03_BFS_V11_Guide_Comparison.json').is_file()


def test_node_cache_controls_and_full_outputs_with_cpu_tensor(tmp_path, monkeypatch):
    import sys
    import types
    torch = pytest.importorskip('torch')
    from swapkit import analysis_nodes as n
    mm = types.ModuleType('comfy.model_management')
    mm.throw_exception_if_processing_interrupted = lambda: None
    mm.unload_all_models = Mock(); mm.soft_empty_cache = Mock()
    comfy = types.ModuleType('comfy'); comfy.__path__ = []; comfy.model_management = mm
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.model_management', mm)
    monkeypatch.setattr(n, 'data_root', lambda: tmp_path)
    source = tmp_path / 'source.mp4'; source.write_bytes(b'fixture')
    video = SimpleNamespace(get_stream_source=lambda: str(source), get_active_trim_window=lambda: (0., 0.))
    monkeypatch.setattr(a, 'current_model_digest', lambda model: 'fixture-digest')
    calls = []
    def query(images, times, target, options, *args):
        calls.append(options)
        return evidence(), {'requested': options.record(), 'eval_count': 6000, 'done_reason': 'stop'}
    monkeypatch.setattr(a, 'query', query)
    def sample(*args, sample_count=8, **kwargs):
        return [Image.new('RGB', (40, 50))] * sample_count, list(range(sample_count)), {'audio_analyzed': False}
    monkeypatch.setattr(media, 'sample_frames', sample)
    character = torch.zeros(1, 64, 32, 3)
    def execute(**kwargs):
        return n.H3SwapAnalyzeAkatz().analyze(video, character, 'performer', 0., 124,
                                             'Reference clothing', **kwargs)
    first = execute()
    cached = execute()
    assert len(calls) == 1
    assert json.loads(cached[2])['cached_draft'] is True
    assert cached[3].shape == (1, 1136, 1536, 3)
    assert 'bfs' not in cached[2].lower()
    execute(reuse_cached_draft=False)
    execute(reuse_cached_draft=False)
    assert len(calls) == 3
    execute(max_output_tokens=8192)
    assert len(calls) == 4
    report = json.loads(first[2])
    assert Path(report['files']['clip_prompt']).read_text().strip() == first[0]
    assert Path(report['files']['chain_prompt']).read_text().strip() == first[1]
    # A corrupt record is regenerated rather than making the workflow unusable.
    Path(report['files']['report']).write_text('{broken')
    rerun = execute()
    assert len(calls) == 5 and any('Ignored invalid' in v for v in json.loads(rerun[2])['warnings'])
    video._VideoFromFile__crop = {'width': 100}
    with pytest.raises(ValueError, match='cropped clip first'): execute()


def test_authored_canvas_widget_order_matches_new_node_schema():
    from swapkit.analysis_nodes import H3SwapAnalyzeAkatz
    canvas, api = graphs.prompt_studio()
    node = next(n for n in canvas['nodes'] if n['type'] == 'H3SwapAnalyzeAkatz')
    names = [k for k in H3SwapAnalyzeAkatz.INPUT_TYPES()['required'] if k not in ('video', 'character')]
    assert list(node['widgets_values_named']) == names
    assert node['widgets_values'] == [api[str(node['id'])]['inputs'][k] for k in names]


def test_character_reference_keeps_prior_768_pixel_budget(monkeypatch):
    import base64, io
    fake = FakeOllama()
    monkeypatch.setattr(prompts, 'session', lambda: fake)
    images = [Image.new('RGB', (768, 1200))] * 17
    a.query(images, [i / 3 for i in range(16)], 'person', opt.AnalysisOptions())
    payload = next(kw['json'] for url, kw in fake.calls if url.endswith('/api/chat'))
    encoded = payload['messages'][1]['images']
    sizes = [Image.open(io.BytesIO(base64.b64decode(value))).size for value in encoded]
    assert max(sizes[0]) == 768
    assert all(max(size) == 576 for size in sizes[1:])
