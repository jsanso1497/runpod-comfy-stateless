"""Regression tests for the local Qwen enhancer. GPU and Ollama are mocked."""
import base64
import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock
import pytest
import torch
from PIL import Image
from workbench import qwen_prompt as pe, qwen_prompt_local as local, qwen_prompt_catalog, config

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    pe._CACHE.clear()
    monkeypatch.delenv('WB_QWEN_PROMPT_MODEL', raising=False)
    monkeypatch.delenv('WB_QWEN_PROMPT_FAILURE', raising=False)
    monkeypatch.setenv('WB_DATA_ROOT', str(tmp_path))
    monkeypatch.setenv('OLLAMA_MODELS', str(tmp_path / 'ollama'))
    yield
    pe._CACHE.clear()


def answer(text):
    return json.dumps({'rewritten_prompt': text, 'wh_ratio': '', 'ratio_follow': '<image1>'})


def images(n=2):
    return {f'image_{i}': torch.full((1, 16+i, 24-i, 3), i / 12) for i in range(1, n+1)}


def test_off_does_not_resolve_model_inspect_images_or_call_service(monkeypatch):
    monkeypatch.setenv('WB_QWEN_PROMPT_MODEL', 'not-a-model')
    monkeypatch.setattr(pe, '_ordered_images', Mock(side_effect=AssertionError('image inspection')))
    monkeypatch.setattr(local, 'request_json', Mock(side_effect=AssertionError('network')))
    original = '  Keep <image1>.\n\tEXACT  '
    result, report = pe.enhance_edit_prompt(original, None, {'enabled': False})
    assert result == original and report['status'].startswith('OFF')


def test_missing_tag_is_repaired_once(monkeypatch):
    seen = []
    def generate(prompt, ordered, options):
        seen.append((prompt, options.repair_error))
        return answer('Edit <image1>.' if len(seen) == 1 else 'Edit <image1> using <image2>.')
    monkeypatch.setattr(pe, '_generate', generate)
    original = 'Edit <image1> using <image2>.'
    text, report = pe.enhance_edit_prompt(original, images(), {'enabled': True})
    assert text == original and report['attempts'] == 2
    assert report['fallback_used'] is False and 'repaired' in report['status']
    assert seen[0][1] == '' and 'reference tag' in seen[1][1]
    assert all(row[0] == original for row in seen)


@pytest.mark.parametrize('invalid', [answer('Edit <image1>.'), answer('Use <image9>.'), 'not JSON', answer(''), '<think>unfinished'])
def test_invalid_rewrite_uses_original_not_blindly_appended_tags(monkeypatch, invalid, capsys):
    generate = Mock(return_value=invalid)
    monkeypatch.setattr(pe, '_generate', generate)
    original = '  Edit <image1> using <image2>.\n  '
    result, report = pe.enhance_edit_prompt(original, images(), {'enabled': True})
    assert result == original and report['fallback_used'] is True
    assert report['retry_count'] == 1 and generate.call_count == 2
    assert report['validation_warning'] and 'WARNING' in report['status']
    assert 'ORIGINAL' in capsys.readouterr().out
    assert not pe._CACHE


@pytest.mark.parametrize('original,lock', [('Write "EXACT WORDS" on <image1>.',''), ('Use <lora:identity:0.8> on <image1>.',''), ('Use exact_TRIGGER on <image1>.','exact_TRIGGER')])
def test_literal_and_trigger_omission_falls_back_safely(monkeypatch, original, lock):
    monkeypatch.setattr(pe, '_generate', lambda *a: answer('Edit <image1>.'))
    result, report = pe.enhance_edit_prompt(original, images(1), {'enabled': True, 'keep_exact': lock})
    assert result == original and report['fallback_used']


def test_strict_mode_remains_available(monkeypatch):
    monkeypatch.setenv('WB_QWEN_PROMPT_FAILURE', 'error')
    monkeypatch.setattr(pe, '_generate', lambda *a: answer('Edit <image1>.'))
    with pytest.raises(pe.EnhancementError, match='twice'):
        pe.enhance_edit_prompt('Use <image1> with <image2>.', images(), {'enabled': True})


def test_service_error_is_not_a_silent_original_prompt_fallback(monkeypatch):
    generate = Mock(side_effect=local.LocalServiceError('Prepare U10.'))
    monkeypatch.setattr(pe, '_generate', generate)
    with pytest.raises(pe.EnhancementError, match='U10'):
        pe.enhance_edit_prompt('Edit <image1>.', images(1), {'enabled': True})
    assert generate.call_count == 1


@pytest.mark.parametrize('prompt', ['Edit <image0>.', 'Edit <image11>.'])
def test_original_bad_reference_is_not_repaired_or_sent(prompt, monkeypatch):
    generate = Mock()
    monkeypatch.setattr(pe, '_generate', generate)
    with pytest.raises(pe.EnhancementError, match='missing reference'):
        pe.enhance_edit_prompt(prompt, images(), {'enabled': True})
    generate.assert_not_called()


@pytest.mark.parametrize('n', [1, 2, 4, 10])
def test_exact_input_order_and_pixels_are_retained(n, monkeypatch):
    source = images(n)
    snapshots = {k: v.clone() for k, v in source.items()}
    captured = []
    prompt = 'Edit ' + ' using '.join(f'<image{i}>' for i in range(1, n+1)) + '.'
    def generate(text, ordered, options):
        captured.extend(ordered)
        return answer(prompt)
    monkeypatch.setattr(pe, '_generate', generate)
    pe.enhance_edit_prompt(prompt, dict(reversed(list(source.items()))), {'enabled': True})
    for i, value in enumerate(captured, 1):
        assert value.data_ptr() == source[f'image_{i}'].data_ptr()
        assert torch.equal(value, snapshots[f'image_{i}'])


def test_success_cache_is_model_aware(monkeypatch):
    generate = Mock(return_value=answer('Edit <image1>.'))
    monkeypatch.setattr(pe, '_generate', generate)
    refs = images(1)
    pe.enhance_edit_prompt('Edit <image1>.', refs, {'enabled': True})
    assert pe.enhance_edit_prompt('Edit <image1>.', refs, {'enabled': True})[1]['cache_hit']
    monkeypatch.setenv('WB_QWEN_PROMPT_MODEL', local.ALTERNATE_MODEL)
    _, report = pe.enhance_edit_prompt('Edit <image1>.', refs, {'enabled': True})
    assert not report['cache_hit'] and report['model'] == local.ALTERNATE_MODEL
    assert generate.call_count == 2


def test_manifest_replacement_invalidates_cache_identity(tmp_path):
    original = local.cache_identity()
    path = tmp_path / 'ollama/manifests/registry.ollama.ai/huihui_ai/qwen3-vl-abliterated/32b-instruct-fp16'
    path.parent.mkdir(parents=True)
    path.write_text('{"layers":[]}')
    changed = local.cache_identity()
    path.write_text('{"layers":[1]}')
    assert len({original, changed, local.cache_identity()}) == 3


@pytest.mark.parametrize('model,precision', [(local.DEFAULT_MODEL,'F16'),(local.ALTERNATE_MODEL,'BF16')])
def test_selected_model_metadata_verification(model, precision, monkeypatch):
    monkeypatch.setenv('WB_QWEN_PROMPT_MODEL', model)
    assert local.selected_model() == model
    assert local.verify_model({'capabilities':['completion','vision'],'details':{'quantization_level':precision}},model)==precision
    assert local.helper_config()['models'] == [model]


@pytest.mark.parametrize('caps,quant', [(['completion'],'F16'),(['vision'],'Q4_K_M'),(['vision'],'Q8_0'),(['vision'],''),(['vision'],'BF16')])
def test_incorrect_or_text_only_32b_model_is_rejected(caps, quant):
    with pytest.raises(local.LocalServiceError):
        local.verify_model({'capabilities':caps,'details':{'quantization_level':quant}},local.DEFAULT_MODEL)


@pytest.mark.parametrize('bad', ['qwen3.8vl:32b', 'huihui_ai/qwen3-vl-abliterated:32b-instruct', 'http://remote.invalid/model'])
def test_unknown_or_quantized_alias_is_never_substituted(bad, monkeypatch):
    monkeypatch.setenv('WB_QWEN_PROMPT_MODEL', bad)
    with pytest.raises(local.LocalServiceError): local.selected_model()


@pytest.fixture
def fake_ollama(monkeypatch):
    requests = []
    events = []
    state = {'content':json.dumps({'rewritten_prompt':'Edit <image1> using <image2>.'}),
             'done':True,'done_reason':'stop','caps':['vision','completion']}
    def request(path, payload=None, **kwargs):
        requests.append((path,payload));events.append(path)
        if path == '/api/show':
            return {'capabilities':state['caps'],'details':{'quantization_level':'F16'}}
        if path == '/api/chat':
            return {k:state[k] for k in ('done','done_reason')} | {'message':{'content':state['content'],'thinking':'PRIVATE REASONING'}}
        if path == '/api/generate':return {'done':True}
        if path == '/api/ps':return {'models':[]}
        raise AssertionError(path)
    monkeypatch.setattr(local,'request_json',request)
    comfy=types.ModuleType('comfy');mm=types.ModuleType('comfy.model_management')
    mm.unload_all_models=lambda:events.append('comfy-unload')
    mm.soft_empty_cache=lambda:events.append('comfy-cache')
    comfy.model_management=mm
    monkeypatch.setitem(sys.modules,'comfy',comfy);monkeypatch.setitem(sys.modules,'comfy.model_management',mm)
    return requests,events,state


def test_vision_payload_schema_sequence_and_verified_unload(fake_ollama):
    requests,events,state=fake_ollama
    source=list(images().values())
    result=local.generate('Edit <image1> using <image2>.',source,pe.Options(True,42,''))
    payload=next(data for path,data in requests if path=='/api/chat')
    assert payload['model']==local.DEFAULT_MODEL and payload['format']==local.SCHEMA
    assert payload['keep_alive']==0 and 'think' not in payload
    assert payload['options']['num_ctx']==16384
    assert len(payload['messages'][1]['images'])==2
    for i,encoded in enumerate(payload['messages'][1]['images']):
        png=Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert png.size==(source[i].shape[2],source[i].shape[1])
    text=json.loads(payload['messages'][1]['content'])
    assert text['attachment_order']==['<image1>','<image2>']
    assert events.index('comfy-unload') < events.index('/api/chat') < events.index('/api/generate') < events.index('/api/ps')
    assert 'PRIVATE REASONING' not in result
    assert pe.parse_response(result,'Edit <image1> using <image2>.',2)


def test_thinking_disabled_only_when_supported(fake_ollama):
    requests,events,state=fake_ollama
    state['caps'].append('thinking')
    local.generate('Edit <image1> using <image2>.',list(images().values()),pe.Options(True))
    assert next(data for path,data in requests if path=='/api/chat')['think'] is False


@pytest.mark.parametrize('field,value', [('content',''),('content','{'),('done_reason','length'),('done',False)])
def test_bad_model_answer_still_unloads(fake_ollama, field, value):
    requests,events,state=fake_ollama
    state[field]=value
    with pytest.raises(local.InvalidReply):local.generate('Edit <image1>.',list(images(1).values()),pe.Options(True))
    assert events[-2:]==['/api/ps','comfy-cache']


def test_residency_failure_stops_generation(monkeypatch):
    def request(path,*a,**k):
        return {'models':[{'name':local.DEFAULT_MODEL}]} if path=='/api/ps' else {'done':True}
    monkeypatch.setattr(local,'request_json',request)
    with pytest.raises(local.LocalServiceError,match='still resident'):
        local.unload_and_verify(local.DEFAULT_MODEL,timeout=0)


def test_vision_copy_is_bounded_without_mutating_reference():
    image=torch.rand(1,1000,2000,3);before=image.clone()
    encoded=local.vision_png(image)
    assert Image.open(io.BytesIO(base64.b64decode(encoded))).size==(768,384)
    assert torch.equal(image,before)


def test_catalog_migration_only_prepares_selected_qwen_model(monkeypatch):
    models={'models':[{'id':'qwen','default_groups':['qwen','qwen_pe_i2i'],'toolbox':['sam','seedvr2']},
                      {'id':'h3','default_groups':['h3'],'toolbox':['ollama']}]}
    changed=qwen_prompt_catalog.apply('models.json',models)
    assert changed['models'][0]['default_groups']==['qwen','ollama']
    assert changed['models'][1]['default_groups']==['h3']
    assets={'assets':[],'ollama':{'models':['h3-existing']},'ollama_by_workspace':{'h3':{'models':['h3-existing']}}}
    monkeypatch.setenv('WB_WORKSPACE','qwen')
    result=qwen_prompt_catalog.apply('assets.json',assets)
    assert result['ollama']['models']==[local.DEFAULT_MODEL]
    assert result['ollama_by_workspace']['h3']['models']==['h3-existing']
    task={'tasks':[{'id':'U10','workspaces':['qwen'],'requires':['qwen_pe_i2i']}]}
    assert qwen_prompt_catalog.apply('tasks.json',task)['tasks'][0]['requires']==['ollama']


def test_qwen_control_socket_and_widget_contract_is_unchanged():
    spec=importlib.util.spec_from_file_location('local_prompt_control',ROOT/'src/nodes/qwen_prompt/__init__.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cls=module.WBQwenPromptControl
    assert list(cls.INPUT_TYPES()['required'])==['enabled','seed','keep_exact']
    assert cls().configure(True,42,'Trigger')==({'enabled':True,'seed':42,'keep_exact':'Trigger'},)


def test_two_krea_variants_are_preserved_and_different():
    k=ROOT/'automation/K01'
    identity=json.loads((k/'Krea_Edit_Multi_Reference_HQ.api.json').read_text())
    standard=json.loads((k/'Krea_Standard_No_Identity_LoRA_HQ.api.json').read_text())
    assert any(n['class_type']=='WBKreaCondition' for n in identity.values())
    assert any(n['class_type']=='WBKreaLoRAs' for n in identity.values())
    assert not any(n['class_type'] in ('WBKreaCondition','WBKreaLoRAs') for n in standard.values())
    assert 'krea2_identity_edit_v1_2.safetensors' not in json.dumps(standard)
    assert any(n['class_type']=='WBKreaStandardCondition' for n in standard.values())
