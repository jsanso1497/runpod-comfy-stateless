"""Prompt Studio CPU contracts. Ollama, ComfyUI GPU work and responses are mocked."""
import base64
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from workbench import prompt_studio as ps
from workbench import qwen_prompt_local as local
from workbench import prompt_studio_catalog as catalog

ROOT = Path(__file__).resolve().parents[1]


def refs(n=3):
    return [{'image': torch.full((1, 40+i, 60-i, 3), (i+1)/12),
             'role': ('scene with dog A', 'dog B face', 'dog B body')[i] if i < 3 else ''}
            for i in range(n)]


def brief(n=3, task='edit'):
    return {'task': task, 'goal': 'Replace dog A with dog B.' if task == 'edit' else 'Describe the visible scene.',
            'canvas_image': 1 if task == 'edit' else 0, 'target': 'The dog in the scene.',
            'images': [{'index': i, 'observed': 'Visible test subject.', 'use_for': f'Role for image {i}'}
                       for i in range(1, n+1)],
            'integration': ['Retain the observed ground contact.'], 'preserve': ['Background.'],
            'uncertainties': []}


def reply(obj):
    return {'done': True, 'done_reason': 'stop', 'message': {'content': json.dumps(obj)}}


@pytest.fixture
def runtime(monkeypatch):
    comfy = ModuleType('comfy'); mm = ModuleType('comfy.model_management')
    mm.unload_all_models = Mock(); mm.soft_empty_cache = Mock()
    mm.throw_exception_if_processing_interrupted = Mock()
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, 'comfy', comfy)
    monkeypatch.setitem(sys.modules, 'comfy.model_management', mm)
    monkeypatch.setenv('WB_QWEN_PROMPT_MODEL', local.DEFAULT_MODEL)
    return mm


def install_api(monkeypatch, replies, *, vision=True, precision='BF16', thinking=True):
    queue = iter(replies); calls = []
    def api(path, payload=None, **kwargs):
        calls.append((path, payload, kwargs))
        if path == '/api/show':
            return {'capabilities': (['vision'] if vision else []) + (['thinking'] if thinking else []),
                    'details': {'quantization_level': precision}}
        if path == '/api/chat':
            result = next(queue)
            if isinstance(result, BaseException): raise result
            return result
        if path == '/api/generate': return {'done': True}
        if path == '/api/ps': return {'models': []}
        raise AssertionError(path)
    monkeypatch.setattr(local, 'request_json', api)
    return calls


def test_dog_swap_sees_individual_images_in_both_passes_and_exports_only_prompt(monkeypatch, runtime):
    source=refs(); saved=[r['image'].clone() for r in source]
    final='Replace the dog in <image1> using the head identity in <image2> and body in <image3>.'
    calls=install_api(monkeypatch,[reply(brief()),reply({'prompt':final})])
    prompt,report=ps.generate_prompt('Swap the dog in image 1 with dog B from image 2 and image 3.',source)
    assert prompt==final and report['task']=='edit' and report['image_count']==3
    chats=[p for path,p,_ in calls if path=='/api/chat']
    assert len(chats)==2
    for chat in chats:
        assert len(chat['messages'][1]['images'])==3
        assert chat['options']['num_ctx']==32768 and chat['think'] is False
        assert chat['keep_alive']=='5m'
        for encoded,ref in zip(chat['messages'][1]['images'],source):
            im=Image.open(io.BytesIO(base64.b64decode(encoded)))
            assert im.size==(ref['image'].shape[2],ref['image'].shape[1])
    assert json.loads(chats[0]['messages'][1]['content'])['image_order'][1]['role_hint']=='dog B face'
    assert 'visual_brief' in json.loads(chats[1]['messages'][1]['content'])
    assert chats[0]['messages'][0]['content']==ps.PLANNER_SYSTEM
    assert chats[1]['messages'][0]['content']==ps.WRITER_SYSTEM
    assert all(torch.equal(r['image'],before) for r,before in zip(source,saved))
    assert [c[0] for c in calls][-2:]==['/api/generate','/api/ps']
    assert calls[-2][1]['keep_alive']==0
    runtime.unload_all_models.assert_called_once()
    assert runtime.soft_empty_cache.call_count==2


@pytest.mark.parametrize('task', ['edit','new_scene','recreate','describe'])
def test_all_explicit_task_modes(monkeypatch,runtime,task):
    selection=next(k for k,v in ps.TASK_CODES.items() if v==task)
    final='A dog sits on grass in soft light.'
    calls=install_api(monkeypatch,[reply(brief(1,task)),reply({'prompt':final})])
    text,report=ps.generate_prompt('Use this image.',refs(1),ps.StudioOptions(task=selection))
    assert text==final and report['task']==task
    packet=json.loads(next(p for path,p,_ in calls if path=='/api/chat')['messages'][1]['content'])
    assert packet['selected_task']==task


def test_auto_can_choose_recreation_and_no_tags(monkeypatch,runtime):
    install_api(monkeypatch,[reply(brief(1,'recreate')),reply({'prompt':'A small tan dog sits on grass, facing the camera.'})])
    text,report=ps.generate_prompt('Describe an image that would recreate this photo.',refs(1))
    assert '<image' not in text and report['task']=='recreate'


def test_instruct_model_does_not_receive_unsupported_think_parameter(monkeypatch,runtime):
    calls=install_api(monkeypatch,[reply(brief(1,'describe')),reply({'prompt':'A dog on grass.'})],thinking=False)
    ps.generate_prompt('Describe the dog.',refs(1))
    assert all('think' not in p for path,p,_ in calls if path=='/api/chat')


def test_prompt_missing_image_tag_retried_not_blindly_appended(monkeypatch,runtime):
    final='Edit <image1> with <image2> head and <image3> body.'
    calls=install_api(monkeypatch,[reply(brief()),reply({'prompt':'Edit <image1> with <image2>.'}),reply({'prompt':final})])
    text,report=ps.generate_prompt('Swap the dogs.',refs())
    assert text==final and report['attempts']=={'analysis':1,'writing':2}
    chats=[p for path,p,_ in calls if path=='/api/chat']
    assert 'retry_requirement' in json.loads(chats[-1]['messages'][1]['content'])


def test_two_invalid_writes_raise_no_unchanged_original_fallback(monkeypatch,runtime):
    calls=install_api(monkeypatch,[reply(brief()),reply({'prompt':'A dog.'}),reply({'prompt':'A dog.'})])
    with pytest.raises(ps.StudioError,match='No unchanged-prompt fallback'):
        ps.generate_prompt('Swap the dogs.',refs())
    assert [x[0] for x in calls][-2:]==['/api/generate','/api/ps']


def test_bad_analysis_gets_exactly_one_retry(monkeypatch,runtime):
    bad=brief(); bad['images']=bad['images'][:2]
    calls=install_api(monkeypatch,[reply(bad),reply(brief()),reply({'prompt':'Edit <image1> with <image2> and <image3>.'})])
    _,report=ps.generate_prompt('Swap them.',refs())
    assert report['attempts']=={'analysis':2,'writing':1}
    chats=[p for path,p,_ in calls if path=='/api/chat']
    assert 'retry_requirement' in json.loads(chats[1]['messages'][1]['content'])


def test_two_bad_analysis_replies_stop_before_writer(monkeypatch,runtime):
    calls=install_api(monkeypatch,[reply({}),reply({})])
    with pytest.raises(ps.StudioError,match='Visual analysis failed twice'):
        ps.generate_prompt('Swap them.',refs())
    assert len([1 for path,_,_ in calls if path=='/api/chat'])==2


@pytest.mark.parametrize('vision,precision,fragment',[(False,'BF16','vision'),(True,'Q4_K_M','precision')])
def test_wrong_model_metadata_never_generates(monkeypatch,runtime,vision,precision,fragment):
    calls=install_api(monkeypatch,[],vision=vision,precision=precision)
    with pytest.raises(ps.StudioError,match=fragment): ps.generate_prompt('Describe.',refs(1))
    assert [p for p,_,_ in calls]==['/api/show']
    runtime.unload_all_models.assert_not_called()


def test_api_error_still_unloads(monkeypatch,runtime):
    calls=install_api(monkeypatch,[local.LocalServiceError('offline')])
    with pytest.raises(ps.StudioError,match='selected local vision model'):
        ps.generate_prompt('Describe.',refs(1))
    assert [p for p,_,_ in calls][-2:]==['/api/generate','/api/ps']


def test_failed_unload_blocks_export(monkeypatch,runtime):
    install_api(monkeypatch,[reply(brief(1,'describe')),reply({'prompt':'A dog.'})])
    monkeypatch.setattr(local,'unload_and_verify',Mock(side_effect=local.LocalServiceError('timeout')))
    with pytest.raises(ps.StudioError,match='No prompt was exported'):
        ps.generate_prompt('Describe.',refs(1))


def test_interrupt_between_passes_unloads(monkeypatch,runtime):
    calls=install_api(monkeypatch,[reply(brief(1,'describe'))])
    runtime.throw_exception_if_processing_interrupted.side_effect=[None,None,RuntimeError('cancelled')]
    with pytest.raises(RuntimeError,match='cancelled'):
        ps.generate_prompt('Describe.',refs(1))
    assert [p for p,_,_ in calls][-2:]==['/api/generate','/api/ps']


@pytest.mark.parametrize('user_request', ['', '   ', 'Use image 4', 'Use <image0>', 'Use image #9'])
def test_invalid_request_or_missing_image_stops_before_api(monkeypatch,runtime,user_request):
    api=Mock(side_effect=AssertionError('No API call'))
    monkeypatch.setattr(local,'request_json',api)
    with pytest.raises(ps.StudioError): ps.generate_prompt(user_request,refs(1))
    api.assert_not_called()


@pytest.mark.parametrize('opts',[
    {'task':'unknown'},{'detail':'unknown'},{'seed':-1},{'seed':True},{'seed':2**31},
    {'vision_edge':300},{'vision_edge':4096},{'max_tokens':12},{'max_tokens':9000},{'guidance':None}])
def test_options_are_validated(opts):
    with pytest.raises(ps.StudioError): ps.StudioOptions(**opts).validate()


@pytest.mark.parametrize('data',[
    {},{'done':False},{'done':True,'done_reason':'length'},
    {'done':True,'message':{'content':'not json'}},
    {'done':True,'message':{'content':'<think>unfinished'}},
    {'done':True,'message':{'content':'[]'}},
    {'done':True,'message':{'thinking':'not a final answer'}}])
def test_incomplete_replies_are_rejected(data):
    with pytest.raises(ps.InvalidOutput): ps.final_json(data)


def test_reasoning_not_forwarded():
    value=ps.final_json({'done':True,'message':{'content':'<think>private work</think>{"prompt":"A dog."}','thinking':'hidden'}})
    assert value=={'prompt':'A dog.'}


@pytest.mark.parametrize('text,task,n',[
    ('Use <image3>.','edit',2),('Use <image1>.','edit',2),
    ('Recreate <image1>.','recreate',1),('Describe image 1.','describe',1),
    ('<think>private</think>','edit',1),('', 'edit',1)])
def test_invalid_final_prompt(text,task,n):
    with pytest.raises(ps.InvalidOutput): ps.validate_prompt({'prompt':text},n,task,[])


def test_exact_terms_checked_but_no_forced_quote_inventory():
    assert ps.validate_prompt({'prompt':'A dog named tr1gger.'},1,'recreate',['tr1gger'])
    with pytest.raises(ps.InvalidOutput):
        ps.validate_prompt({'prompt':'A dog.'},1,'recreate',['tr1gger'])


@pytest.mark.parametrize('alter',[
    lambda x:x.update(task='bad'),lambda x:x.update(task='describe'),
    lambda x:x.update(canvas_image=4),lambda x:x.update(canvas_image=0),
    lambda x:x['images'][0].update(index=2),lambda x:x.update(integration='bad'),
    lambda x:x.update(goal=''),lambda x:x.update(extra='unexpected')])
def test_plan_validation(alter):
    p=brief();alter(p)
    with pytest.raises(ps.InvalidOutput):ps.validate_plan(p,3,'edit')


def test_vision_copy_respects_size_and_alpha_and_does_not_mutate():
    source=torch.ones(1,800,2400,4);source[...,0]=0;source[...,3]=0
    before=source.clone()
    value=ps.encode_images([{'image':source}],1024)[0]
    photo=Image.open(io.BytesIO(base64.b64decode(value)))
    assert photo.width==1024 and photo.height==341 and photo.getpixel((0,0))==(255,255,255)
    assert torch.equal(source,before)


@pytest.mark.parametrize('image',[torch.zeros(2,5,5,3),torch.zeros(1,5,5,1),torch.full((1,5,5,3),float('nan'))])
def test_bad_image_tensor(image):
    with pytest.raises(ps.StudioError):ps.encode_images([{'image':image}],1024)


def node_module():
    name='studio_node_under_test'
    spec=importlib.util.spec_from_file_location(name,ROOT/'src/workbench/prompt_studio_nodes.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_all_optional_slots_can_be_none_except_first():
    m=node_module();primary=refs(1)[0]
    assert m.collect_references(primary,{f'reference_{i}':None for i in range(2,9)})==[primary]
    assert m.WBPromptStudioImage().load('None','')==(None,)
    assert m.WBPromptStudioImage.VALIDATE_INPUTS('None') is True
    assert m.WBPromptStudioImage.IS_CHANGED('None')=='None'


def test_gaps_are_explicit_not_silently_renumbered():
    m=node_module()
    with pytest.raises(ps.StudioError,match='Image 3'):
        m.collect_references(refs(1)[0],{'reference_3':refs(1)[0]})
    with pytest.raises(ps.StudioError,match='Image 1'):
        m.collect_references(None,{})


@pytest.fixture
def output_dir(tmp_path,monkeypatch):
    fp=ModuleType('folder_paths');fp.get_output_directory=lambda:str(tmp_path/'output')
    monkeypatch.setitem(sys.modules,'folder_paths',fp)
    return tmp_path/'output'


def test_only_text_export_no_source_analysis_or_json(monkeypatch,output_dir):
    m=node_module()
    report={'task':'edit','image_count':1,'model':'test','precision':'BF16',
            'roles':[{'index':1,'use_for':'source'}],'uncertainties':[]}
    monkeypatch.setattr(m,'generate_prompt',lambda *a:('Replace the dog.',report))
    result=m.WBQwenPromptStudio().write_prompt(refs(1)[0],'Swap it.')
    files=list(output_dir.rglob('*.*'))
    assert len(files)==1 and files[0].suffix=='.txt'
    assert files[0].read_text()=='Replace the dog.\n'
    assert result['result']==('Replace the dog.',)
    assert 'roles' not in files[0].read_text()


def test_export_off_does_not_create_any_file(monkeypatch,output_dir):
    m=node_module()
    report={'task':'describe','image_count':1,'model':'test','precision':'BF16','roles':[],'uncertainties':[]}
    monkeypatch.setattr(m,'generate_prompt',lambda *a:('A dog.',report))
    result=m.WBQwenPromptStudio().write_prompt(refs(1)[0],'Describe.',save_txt=False)
    assert result['result']==('A dog.',) and not output_dir.exists()


@pytest.mark.parametrize('name',['../escape','x/y','/tmp/a','', 'a\\b'])
def test_export_prefix_cannot_escape(name,output_dir):
    with pytest.raises(ps.StudioError):node_module().save_text('A dog.',name)


def test_export_rejects_symlink(output_dir,tmp_path):
    output_dir.mkdir();(output_dir/'prompt_studio').symlink_to(tmp_path)
    with pytest.raises(ps.StudioError):node_module().save_text('A dog.')


def test_exports_never_overwrite(output_dir):
    m=node_module();m.save_text('First');m.save_text('Second')
    assert sorted(p.read_text() for p in output_dir.rglob('*.txt'))==['First\n','Second\n']


def test_workflow_is_prompt_only_preconnected_and_schema_matches(monkeypatch,tmp_path):
    fp=ModuleType('folder_paths');fp.get_input_directory=lambda:str(tmp_path)
    monkeypatch.setitem(sys.modules,'folder_paths',fp)
    m=node_module();g=json.loads((ROOT/'workflows/toolbox/U11_Qwen_Smart_Prompt_Studio.json').read_text())
    lookup={n['id']:n for n in g['nodes']}
    assert set(n['type'] for n in g['nodes'])=={'WBPromptStudioImage','WBQwenPromptStudio','Note'}
    assert len([n for n in g['nodes'] if n['type']=='WBPromptStudioImage'])==8
    assert len(g['links'])==8
    for lid,src,ss,dst,ds,kind in g['links']:
        assert lookup[src]['outputs'][ss]['links']==[lid]
        assert lookup[dst]['inputs'][ds]['link']==lid
        assert lookup[dst]['inputs'][ds]['name']==f'reference_{src}'
    writer=next(n for n in g['nodes'] if n['type']=='WBQwenPromptStudio')
    assert m.WBQwenPromptStudio.OUTPUT_NODE and m.WBQwenPromptStudio.RETURN_TYPES==('STRING',)
    assert writer['widgets_values'][3] is True
    assert writer['widgets_values'][1] in ps.TASKS
    assert all(n['widgets_values'][0]=='None' for n in g['nodes'] if n['type']=='WBPromptStudioImage')
    assert set(writer['inputs'][i]['name'] for i in range(8))=={
        'reference_1',*m.WBQwenPromptStudio.INPUT_TYPES()['optional']}


def test_catalog_additive_and_duplicate_guard(tmp_path):
    data={'schema_version':1,'tasks':[{'id':'Q01','title':'unchanged'}]}
    before=copy.deepcopy(data)
    catalog.append_tasks(data,tmp_path)
    assert data==before
    (tmp_path/'catalog').mkdir()
    (tmp_path/'catalog/tasks.prompt-studio.json').write_bytes((ROOT/'catalog/tasks.prompt-studio.json').read_bytes())
    catalog.append_tasks(data,tmp_path)
    assert data['tasks'][0]==before['tasks'][0] and data['tasks'][1]['id']=='U11'
    assert data['tasks'][1]['requires']==['ollama']
    with pytest.raises(ValueError,match='Duplicate'):catalog.append_tasks(data,tmp_path)


def test_old_enhancer_node_schema_unchanged():
    # New class registration is appended; no changes to old class definitions.
    text=(ROOT/'src/nodes/qwen_prompt/__init__.py').read_text()
    assert 'class WBQwenPromptControl:' in text and 'class WBQwenPromptPreview:' in text
    assert 'NODE_CLASS_MAPPINGS.update(_STUDIO_CLASSES)' in text
    assert 'prompt_studio' not in (ROOT/'src/workbench/qwen_prompt.py').read_text()


def test_frontend_contract():
    import shutil
    import subprocess
    executable=shutil.which('node')
    if executable is None:
        pytest.skip('Node.js is not installed locally; GitHub CI installs it.')
    subprocess.run([executable,str(ROOT/'tests/test_prompt_studio_ui.cjs')],check=True,capture_output=True,text=True)


def test_literal_image_number_is_not_a_reference_requirement():
    assert ps.referenced_indices('Render the text "image 8" on image 1.')=={1}
    assert ps.validate_prompt({'prompt':'A white sign bearing "image 8" in black lettering.'},1,'recreate',['image 8'])
