import copy
import json
from pathlib import Path
import pytest
from workbench.factory_sync import sync_factory
from workbench.graph_audit import audit_graph


def source(tmp_path):
    root=tmp_path/'repo';data=tmp_path/'volume';root.mkdir();data.mkdir()
    (root/'w.json').write_text('{"one":1}')
    tasks=[{'id':'QKEEP1','workspaces':['qwen'],'variants':[{'file':'w.json'}]}]
    return root,data,tasks


def test_factory_updates_version_not_users_copy(tmp_path):
    root,data,tasks=source(tmp_path)
    report=sync_factory(root,data,tasks,'qwen')
    old=data/report['workflows'][0]['path'];old.write_text('user edit')
    # Same source gets an alternate official file; user edit survives.
    again=sync_factory(root,data,tasks,'qwen')
    assert old.read_text()=='user edit'
    assert (data/again['workflows'][0]['path']).read_text()=='{"one":1}'
    (root/'w.json').write_text('{"one":2}')
    changed=sync_factory(root,data,tasks,'qwen')
    assert changed['directory']!=report['directory']
    assert old.read_text()=='user edit'


def test_factory_idempotent_and_workspace_scoped(tmp_path):
    root,data,tasks=source(tmp_path)
    first=sync_factory(root,data,tasks,'qwen');second=sync_factory(root,data,tasks,'qwen')
    assert first==second
    assert len(list(data.rglob('*.json')))==1
    assert sync_factory(root,data,tasks,'krea')['workflows']==[]


def graph():
    return {'nodes':[{'id':1,'inputs':[],'outputs':[{'links':[1]}]},
                     {'id':2,'inputs':[{'link':1}],'outputs':[]}],
            'links':[[1,1,0,2,0,'IMAGE']]}


@pytest.mark.parametrize('broken',['src','dst','slot','dangling','duplicate','output'])
def test_recursive_validator_catches_nested_errors(broken):
    g=graph()
    if broken=='src':g['links'][0][1]=99
    elif broken=='dst':g['links'][0][3]=99
    elif broken=='slot':g['links'][0][2]=-1
    elif broken=='dangling':g['nodes'][1]['inputs'][0]['link']=99
    elif broken=='duplicate':g['links'].append(g['links'][0])
    else:g['nodes'][0]['outputs'][0]['links']=[]
    g['name']='inner';g['inputNode']={'id':-10};g['outputNode']={'id':-20}
    outer={'nodes':[],'links':[],'definitions':{'subgraphs':[g]}}
    assert audit_graph(outer)


def test_real_qkeep1_nested_links():
    root=Path(__file__).resolve().parents[1]
    path=root/'workflows/qwen/QKEEP1/Qwen21_Keep_One_Person_Protected_Overlay_BF16.json'
    assert audit_graph(json.loads(path.read_text()))==[]


def test_real_moss_i2i_links():
    root=Path(__file__).resolve().parents[1]
    path=root/'workflows/experimental/moss/QMOSS_I2I/MOSS_Image_I2I.json'
    assert audit_graph(json.loads(path.read_text()))==[]


def test_alternate_factory_symlink_is_rejected(tmp_path):
    from workbench.factory_sync import sync_factory
    import hashlib
    root=tmp_path/'repo';root.mkdir();data=tmp_path/'volume';data.mkdir()
    (root/'graph.json').write_text('{}')
    tasks=[{'id':'ONE','workspaces':['qwen'],'variants':[{'file':'graph.json'}]}]
    report=sync_factory(root,data,tasks,'qwen')
    dest=data/report['workflows'][0]['path'];dest.write_text('user edit')
    other=tmp_path/'private';other.write_text('never replace')
    digest=hashlib.sha256(b'{}').hexdigest()[:12]
    alt=dest.with_name(dest.stem+'__factory_'+digest+dest.suffix);alt.symlink_to(other)
    with pytest.raises(ValueError,match='alternate destination'):sync_factory(root,data,tasks,'qwen')
    assert other.read_text()=='never replace'
