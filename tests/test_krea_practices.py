"""Static/CPU contracts for reviewed graph changes, not GPU quality benchmarks."""
import copy
import pytest
from test_identity_release import graph_fixture, builder
from workbench.krea_practices import adapt_developer_runtime, update_k01_defaults


@pytest.mark.parametrize('raw,two',[(False,False),(False,True),(True,False),(True,True)])
def test_developer_pixel_path_and_srgb(raw,two):
    graph=builder.adapter.adapt_krea(graph_fixture(),raw,two)
    graph=builder.add_optional_krea_loras(adapt_developer_runtime(graph))
    builder.validate_topology(graph)
    byid={n['id']:n for n in graph['nodes']}
    assert not any(n['type']=='VAEEncode' for n in graph['nodes'])
    assert byid[72]['type']==byid[90]['type']=='WBKreaLoadImage'
    assert byid[29]['type']=='WBKreaSaveImage'
    assert byid[90]['mode']==(0 if two else 4)
    assert byid[53]['widgets_values'][2:4]==([20,3] if raw else [10,1])
    links={l[0]:l for l in graph['links']}
    ports={p['name']:p['link'] for p in byid[79]['inputs']}
    assert links[ports['source_latent']][1:3]==[82,0]
    assert links[ports['target_latent']][1:3]==[82,0]
    assert ports['source_latent_b'] is None
    assert ports['source_image_b'] is not None
    assert byid[83]['widgets_values']==['1:1 (Square)',1,16]
    assert byid[85]['widgets_values']==['',768,'']
    assert len([n for n in byid.values() if n.get('title','').startswith('Optional LoRA')])==2


def test_missing_pixel_path_is_not_silently_discarded():
    graph=builder.adapter.adapt_krea(graph_fixture())
    next(p for n in graph['nodes'] if n['id']==79 for p in n['inputs'] if p['name']=='source_image')['link']=None
    with pytest.raises(ValueError,match='complete pixel path'):adapt_developer_runtime(graph)


@pytest.mark.parametrize('standard',[False,True])
def test_k01_factory_defaults_and_native_loader(standard):
    graph={'nodes':[
        {'id':8,'type':'WBKreaModels','properties':{},'widgets_values':['raw','encoder','vae','Auto',20,3]},
        {'id':11,'widgets_values':[1,'randomize',20,3,'euler','simple',1]},
        {'id':15,'widgets_values':['Existing note.']}], 'links':[['unchanged']]}
    api={'8':{'class_type':'WBKreaModels','inputs':{'checkpoint':'raw'}},
         '10':{'class_type':'WBKreaStandardCondition' if standard else 'WBKreaCondition'}}
    original_links=copy.deepcopy(graph['links'])
    graph,api=update_k01_defaults(graph,api)
    byid={n['id']:n for n in graph['nodes']}
    assert byid[8]['widgets_values'][0]=='krea2_turbo_bf16.safetensors'
    assert api['8']['inputs']['checkpoint']==byid[8]['widgets_values'][0]
    assert byid[11]['widgets_values'][2:4]==[8,1]
    assert api['8']['class_type']==('WBKreaStandardModels' if standard else 'WBKreaModels')
    assert graph['links']==original_links
    note=byid[15]['widgets_values'][0]
    update_k01_defaults(graph,api)
    assert byid[15]['widgets_values'][0]==note
