"""Graph, catalog and build contracts; no live model inference is asserted."""
import copy
import json
from pathlib import Path
import pytest
from workbench import moss_catalog as c, moss_workflows as b, moss_i2i_workflows as i

ROOT = Path(__file__).resolve().parents[1]


def stock():
    return json.loads((ROOT/'tests/fixtures/moss_stock_runtime_contract.json').read_text())


def graph():
    return i.make_img2img(stock())


def node(g, ident):
    return next(n for n in g['nodes'] if n['id']==ident)


def source_of(g, ident, name):
    port=next(p for p in node(g,ident)['inputs'] if p['name']==name)
    return next(row[1:3] for row in g['links'] if row[0]==port['link'])


def test_stock_graph_never_mutated():
    raw=stock();before=copy.deepcopy(raw)
    i.make_img2img(raw)
    assert raw==before


def test_serialized_file_matches_generator():
    assert (ROOT/c.I2I_WORKFLOW).read_bytes()==i.encode_graph(graph())


def test_native_input_latent_chain_is_complete():
    g=graph()
    assert source_of(g,26,'image')==[25,0]
    assert source_of(g,27,'pixels')==[26,0]
    assert source_of(g,27,'vae')==[3,0]
    assert source_of(g,7,'latent_image')==[27,0]
    assert source_of(g,8,'samples')==[7,0]
    assert source_of(g,8,'vae')==[3,0]
    assert source_of(g,28,'images')==[8,0]
    assert not any(n['type'].startswith('Empty') for n in g['nodes'])


def test_working_size_and_output_explicit():
    g=graph();resize=node(g,26)
    assert resize['widgets_values_named']=={'upscale_method':'lanczos','megapixels':1.0,'resolution_steps':16}
    assert node(g,28)['widgets_values']==['MOSS_Image/I2I']
    assert sum(n['type']=='SaveImage' for n in g['nodes'])==1
    assert not any(n['type']=='PreviewImage' for n in g['nodes'])


def test_unused_load_image_mask_and_no_overlay():
    g=graph();assert node(g,25)['outputs'][1]['links']==[]
    assert not any(n['type'] in {'WBOverlayExport','Q21PhotoMask','Q21PhotoStitch','ImageCompositeMasked'} for n in g['nodes'])


def test_same_loader_files_and_no_qwen_encoder_substitution():
    g=graph()
    assert node(g,1)['widgets_values']==['moss_image_sft_int8_convrot.safetensors','default']
    assert node(g,2)['widgets_values']==['qwen3.5_9b_w4a8_gptq.safetensors','moss_image','default']
    assert node(g,3)['widgets_values']==['flux2-vae_small.safetensors']
    assert c.MODEL_FILES==frozenset(x[1].split('/',1)[1] for x in c.ASSET_SPECS)


def test_sampler_retains_stock_values_except_denoise():
    g=graph();values=node(g,7)['widgets_values_named'];outer=stock()['nodes'][0]['widgets_values_named']
    assert values['denoise']==0.35
    for name in ('seed','steps','cfg','sampler_name','scheduler'):
        assert values[name]==outer[name]
    assert node(g,7)['widgets_values']==list(values.values())


def test_all_widget_serialization_agrees():
    for n in graph()['nodes']:
        assert n['widgets_values']==list(n['widgets_values_named'].values()),n['type']


def test_outer_negative_not_stale_empty_subgraph_default():
    g=graph()
    assert node(g,5)['widgets_values_named']['text']=='nsfw, worst quality, bad anatomy, watermark, jpeg artifact'
    assert [p['name'] for p in node(g,5)['inputs']]==['clip']


def test_promoted_inputs_return_to_editable_widgets():
    g=graph()
    assert node(g,1)['inputs']==[]
    assert node(g,2)['inputs']==[]
    assert node(g,3)['inputs']==[]
    assert node(g,17)['inputs']==[]
    assert [p['name'] for p in node(g,7)['inputs']]==['model','positive','negative','latent_image']
    assert [p['name'] for p in node(g,16)['inputs']]==['on_false','on_true']


def test_stock_prompt_upsampler_text_only_and_unchanged():
    g=graph();original={n['id']:n for n in stock()['definitions']['subgraphs'][0]['nodes']}
    assert node(g,14)['widgets_values']==original[14]['widgets_values']
    assert node(g,11)['widgets_values']==original[11]['widgets_values']
    assert node(g,16)['widgets_values']==[True]
    for name in ('image','video','audio'):
        assert next(p for p in node(g,11)['inputs'] if p['name']==name)['link'] is None
    assert source_of(g,11,'clip')==[2,0]
    assert source_of(g,11,'system_prompt')==[14,0]
    assert source_of(g,11,'prompt')==[17,0]
    assert source_of(g,16,'on_false')==[17,0]
    assert source_of(g,16,'on_true')==[11,0]
    assert source_of(g,4,'text')==[13,0]


def test_stock_slg_switch_retained_off():
    g=graph();assert node(g,24)['widgets_values']==[False]
    assert source_of(g,24,'on_false')==[1,0]
    assert source_of(g,24,'on_true')==[15,0]
    assert source_of(g,7,'model')==[24,0]


def test_link_types_dag_and_flat_native_nodes():
    g=graph();i.validate_graph(g)
    assert not g.get('definitions')
    order={n['id']:n['order'] for n in g['nodes']}
    assert all(order[r[1]]<order[r[3]] for r in g['links'])
    assert len(order)==19 and len(g['links'])==22


@pytest.mark.parametrize('case',['subgraph','node','layout','widget','outer','output'])
def test_adapter_rejects_changed_source_contract(case):
    raw=stock();sg=raw['definitions']['subgraphs'][0]
    if case=='subgraph':raw['definitions']['subgraphs']=[]
    elif case=='node':sg['nodes'].pop()
    elif case=='layout':sg['nodes'][0]['type']='UnrecognizedPromptNode'
    elif case=='widget':next(n for n in sg['nodes'] if n['id']==7)['widgets_values'].pop()
    elif case=='outer':del raw['nodes'][0]['widgets_values_named']['seed']
    elif case=='output':sg['links']=[r for r in sg['links'] if r['target_id']!=-20]
    with pytest.raises(ValueError):i.make_img2img(raw)


@pytest.mark.parametrize('case',['dangling','type','input','output','latent','empty','subgraph'])
def test_graph_validator_rejects_broken_output_or_latent_route(case):
    g=graph()
    if case=='dangling':g['links'][0][1]=999
    elif case=='type':g['links'][0][5]='IMAGE'
    elif case=='input':next(p for p in node(g,7)['inputs'] if p['name']=='latent_image')['link']=999
    elif case=='output':node(g,25)['outputs'][0]['links']=[999]
    elif case=='latent':node(g,27)['type']='EmptyFlux2LatentImage'
    elif case=='empty':node(g,29)['type']='EmptyLatentImage'
    else:g['definitions']={'subgraphs':[{}]}
    with pytest.raises(ValueError):i.validate_graph(g)


def test_both_tasks_share_exactly_one_asset_bundle(tmp_path):
    (tmp_path/'catalog').mkdir();(tmp_path/'catalog/tasks.moss.json').write_bytes((ROOT/'catalog/tasks.moss.json').read_bytes())
    tasks=c.apply('tasks.json',{'tasks':[]},tmp_path)['tasks']
    assert [t['id'] for t in tasks]==['QMOSS','QMOSS_I2I']
    assert all(t['requires']==['moss_image'] for t in tasks)
    rows=c.apply('assets.json',{'assets':[]},tmp_path)['assets']
    assert len(rows)==3 and len({r['destination'] for r in rows})==3
    assert all(r['family']=='moss' for r in rows)


@pytest.mark.parametrize('file',sorted(c.MODEL_FILES))
def test_i2i_precision_allowlist_reuses_only_stock_files(file):
    assert c.is_stock_model_selection('QMOSS_I2I',file)
    assert not c.is_stock_model_selection('Q01',file)
    assert not c.is_stock_model_selection('QMOSS_I2I','untrusted_int8.safetensors')


def test_deterministic_build_retains_t2i_bytes_and_i2i_distinction(tmp_path,monkeypatch):
    (tmp_path/'catalog').mkdir();(tmp_path/'catalog/tasks.moss.json').write_bytes((ROOT/'catalog/tasks.moss.json').read_bytes())
    raw=(ROOT/'tests/fixtures/moss_stock_runtime_contract.json').read_bytes()
    monkeypatch.setattr(b,'TEMPLATE_BLOB',b.git_blob(raw))
    archive=tmp_path/'vendor/moss-stock'/(b.TEMPLATE_REVISION+'.json');archive.parent.mkdir(parents=True);archive.write_bytes(raw)
    report=b.build(tmp_path,offline=True)
    assert report['copied_byte_for_byte'] is True
    assert report['image_to_image']['official_template'] is False
    assert report['image_to_image']['additional_models']==[]
    assert (tmp_path/c.WORKFLOW).read_bytes()==raw
    image=(tmp_path/c.I2I_WORKFLOW).read_bytes()
    assert image==i.encode_graph(graph())
    assert report==b.build(tmp_path,offline=True)
    assert image==(tmp_path/c.I2I_WORKFLOW).read_bytes()


def test_existing_same_pins_and_group_are_unchanged():
    assert c.COMFY_COMMIT=='bab80ef793ac42ad3f25589a6b0507ca537b1dd1'
    assert c.MODEL_REVISION=='f28da8f93449c16cfa7efa07dcd1303a1c0c44d1'
    assert sum(r['estimated_bytes'] for r in c.asset_rows())==13006811386


def test_unsupported_instruction_edit_claim_not_made():
    meta=graph()['extra']['workbench_source']
    assert meta['official_img2img_workflow'] is False
    assert meta['reference_conditioned_edit_model'] is False
    assert meta['source_to_prompt_upsampler'] is False
    assert meta['gpu_tested'] is False
