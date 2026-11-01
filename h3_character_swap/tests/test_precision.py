"""Source/fixture-only tests. None runs the full H3 model or claims visual quality."""
import copy
import json
import shutil
import struct
from pathlib import Path
import pytest
from swapkit import precision as p
from swapkit.graphs import validate_graph
from swapkit.disk_common import validate_api_graph

ROOT=Path(__file__).resolve().parents[1]


def image_root(tmp_path):
    dest=tmp_path/'image'
    shutil.copytree(ROOT,dest,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    return dest


def seed_native(root):
    from swapkit.preview_graph import native_preview
    from swapkit.graphs import adapt_akatz
    canvas=json.loads((root/'tests/fixtures/native_canvas_for_preview.json').read_text())
    api=json.loads((root/'swapkit/data/akatz_disk_base.api.json').read_text())
    # The fixture has upstream names and the worker already has canonical files.
    canvas,api,mask=adapt_akatz(canvas,api)
    canvas,api=native_preview(canvas,api)
    (root/'workflows/02B_Akatz_Native_Simple_Preview.json').write_text(json.dumps(canvas))
    (root/'workflows/02B_Akatz_Native_Simple_Preview.api.json').write_text(json.dumps(api))


def test_baseline_does_not_change_assets(tmp_path):
    root=image_root(tmp_path);before=(root/'assets.spec.json').read_bytes()
    p.prepare_build('baseline',root);assert p.profile(root)=='baseline'
    assert (root/'assets.spec.json').read_bytes()==before
    assert p.cli_flags(root)==[] and p.folder_suffix(root)==''


def test_full_source_not_quantized_conversion(tmp_path):
    root=image_root(tmp_path);p.prepare_build(p.BF16,root)
    rows=json.loads((root/'assets.spec.json').read_text())['assets']
    common=[x for x in rows if x['id'] in {v['id'] for v in p.UPGRADES.values()}]
    assert len(common)==2
    assert all(x['revision']==p.REVISION and len(x['expected_sha256'])==64 for x in common)
    assert all('pruned' not in x['file'] and 'int8' not in x['file'] and 'nvfp4' not in x['file'] for x in common)
    assert p.profile(root)==p.BF16 and p.folder_suffix(root)=='_BF16'
    assert p.cli_flags(root)==['--bf16-unet','--bf16-text-enc','--use-pytorch-cross-attention']


def test_build_profile_idempotent_and_not_toggle(tmp_path):
    root=image_root(tmp_path);p.prepare_build(p.BF16,root)
    before=(root/'assets.spec.json').read_bytes();p.prepare_build(p.BF16,root)
    assert before==(root/'assets.spec.json').read_bytes()
    with pytest.raises(ValueError,match='separate clean'):p.prepare_build('baseline',root)


@pytest.mark.parametrize('bad',['','full','fp32','auto',None])
def test_unknown_build_profile_rejected(tmp_path,bad):
    with pytest.raises(ValueError):p.prepare_build(bad,tmp_path)


def test_api_only_precision_and_attention_change():
    original=json.loads((ROOT/'swapkit/data/akatz_disk_base.api.json').read_text())
    g=p.transform(original);p.validate_recipe(g);validate_api_graph(g)
    assert g['model']['inputs']['unet_name']==p.DIFFUSION
    assert g['text']['inputs']['clip_name']==p.ENCODER
    for key in original:
        if key not in {'model','text','attention'}:assert g[key]==original[key]
    assert original['model']['inputs']['unet_name']==p.DIFFUSION_OLD


def test_canvas_widgets_and_links_match():
    orig=json.loads((ROOT/'tests/fixtures/native_canvas_for_preview.json').read_text())
    g=p.transform(orig);validate_graph(g)
    assert g['links']==orig['links']
    for n in g['nodes']:
        named=n.get('widgets_values_named')
        if named:assert list(named.values())==n['widgets_values']
    assert next(n for n in g['nodes'] if n['type']=='ModelAttentionBackend')['widgets_values_named']['attention']=='pytorch attention'


def test_finalize_both_native_and_worker_and_optional_highref(tmp_path):
    root=image_root(tmp_path);seed_native(root)
    p.prepare_build(p.BF16,root);p.finalize_build(root)
    api=json.loads((root/'swapkit/data/akatz_disk_base.api.json').read_text());p.validate_recipe(api)
    for filename in ['02B_Akatz_Native_Simple_Preview','02C_Akatz_BF16_HighRef_Preview']:
        g=json.loads((root/'workflows'/(filename+'.json')).read_text());validate_graph(g)
        a=json.loads((root/'workflows'/(filename+'.api.json')).read_text());p.validate_recipe(a);validate_api_graph(a)
    low=json.loads((root/'workflows/02B_Akatz_Native_Simple_Preview.api.json').read_text())
    high=json.loads((root/'workflows/02C_Akatz_BF16_HighRef_Preview.api.json').read_text())
    assert low['reference_conditioning']['inputs']['ref_image_size']=='match'
    assert high['reference_conditioning']['inputs']['ref_image_size']=='max'
    assert low['reference_conditioning']['inputs']['prompt']==high['reference_conditioning']['inputs']['prompt']==p.SIMPLE
    assert all(high[k]==low[k] for k in low if k not in {'reference_conditioning','save'})


@pytest.mark.parametrize('node,field,value',[('attention','attention','comfy kitchen attention'),
 ('model','unet_name',p.DIFFUSION_OLD),('text','clip_name',p.ENCODER_OLD),
 ('schedule','steps',20),('sampler','sampler_name','euler'),('guider','cfg',7.),('adapter','strength_model',.5)])
def test_misconfigured_bf16_recipe_rejected(node,field,value):
    g=p.transform(json.loads((ROOT/'swapkit/data/akatz_disk_base.api.json').read_text()))
    g[node]['inputs'][field]=value
    with pytest.raises(ValueError):p.validate_recipe(g)


def header_file(path,dtype='BF16'):
    header=json.dumps({'layer.weight':{'dtype':dtype,'shape':[2,2],'data_offsets':[0,8]}}).encode()
    path.write_bytes(struct.pack('<Q',len(header))+header+bytes(8))


def test_header_inspection_no_tensor_load(tmp_path):
    f=tmp_path/'w.safetensors';header_file(f)
    r=p.read_header(f)
    assert r['weight_dtypes']==['BF16'] and r['parameter_elements_by_dtype']=={'BF16':4}


@pytest.mark.parametrize('content',[b'',struct.pack('<Q',1000),struct.pack('<Q',2)+b'{}'])
def test_invalid_headers_refused(tmp_path,content):
    f=tmp_path/'w.safetensors';f.write_bytes(content)
    with pytest.raises(ValueError):p.read_header(f)


def test_real_selected_dtype_report_and_no_quantized_fallback(tmp_path):
    root=image_root(tmp_path);p.prepare_build(p.BF16,root)
    store=tmp_path/'workspace';target=store/'models/diffusion_models'/p.DIFFUSION
    target.parent.mkdir(parents=True);header_file(target)
    (root/'assets.lock.json').write_text(json.dumps({'assets':[
      {'id':'h3-ref2va-bf16-full','group':'common','destination':'diffusion_models/'+p.DIFFUSION,'sha256':'a'*64,'bytes':target.stat().st_size}]}))
    report=p.audit_weights(root,store)
    assert report['assets'][0]['weight_dtypes']==['BF16']
    assert (store/'h3-swap/precision-audit.json').exists()
    header_file(target,'I8')
    with pytest.raises(ValueError,match='quantized'):p.audit_weights(root,store)


def test_bf16_job_first_continuation_and_retry_keep_full_precision():
    from swapkit.disk_chain import make_job, plan_segments
    api=p.transform(json.loads((ROOT/'swapkit/data/akatz_disk_base.api.json').read_text()))
    config={'resolution':'768','target_person':'person','selection':'largest','selection_index':0,
      'mask_expansion':64,'prompt':p.SIMPLE,'window_frames':124,'seed':904239}
    for seg in plan_segments(1152):
        for attempt in [0,1]:
            prev={'checkpoint':'h.npz','checkpoint_sha256':'a'*64} if seg['index'] else None
            g=make_job(api,config,seg,'clip.mkv','ref.png','attempt',prev,attempt)
            p.validate_recipe(g);validate_api_graph(g)
            assert g['reference_conditioning']['inputs']['prompt']==p.SIMPLE


def test_separate_ci_tag_and_build_arg():
    action=(ROOT.parent/'.github/workflows/build-h3-bf16.yml').read_text()
    assert 'H3_BUILD_PROFILE=bf16_full' in action
    assert 'docker push "$IMAGE:h3-character-swap-bf16"' in action
    assert 'docker push "$IMAGE:h3-character-swap"' not in action
    docker=(ROOT/'Dockerfile').read_text()
    assert docker.index('swapkit.precision --prepare-build')<docker.index('python -m swapkit.build')


def test_resume_rejects_cross_profile_before_queue(tmp_path,monkeypatch):
    from swapkit import disk_chain as d
    run=tmp_path/'output/h3_character_swap/disk_chain_v2/run';run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps({'version':d.VERSION,'model_profile':'bf16_full','input_root':str(tmp_path/'input'),'output_root':str(tmp_path/'output'),'run_name':'run'}))
    with pytest.raises(d.ChainError,match='different precision'):d.execute(run)


def test_profiles_recorded_on_recovery(tmp_path):
    from swapkit.recovery import control_workflow
    inp=tmp_path/'input';inp.mkdir()
    cfg={'model_profile':'bf16_full','run_name':'test','input_root':str(inp),'source':str(inp/'s.mp4'),
      'reference':str(inp/'r.png'),'duration_seconds':10,'start_seconds':0,'resolution':'768',
      'window_frames':124,'target_person':'person','selection':'largest','selection_index':0,
      'mask_expansion':64,'seed':904239,'review_mode':'Review every chunk','max_retries':0,'prompt':p.SIMPLE}
    canvas,_=control_workflow(cfg)
    assert canvas['extra']['h3_recovery']['model_profile']=='bf16_full'
