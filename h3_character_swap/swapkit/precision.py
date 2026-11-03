"""Build-time, explicit H3 precision profiles. Never dequantizes a quantized file.

The BF16 image is published under its own tag. Native and disk-worker graphs are
both rewritten during the build; the baseline image and existing weights survive.
Hardware/visual quality is not established by these source-level checks.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import struct
import uuid

from .config import ROOT, atomic_json, data_root

BASELINE = 'baseline'
BF16 = 'bf16_full'
ECONOMY = 'bf16_economy'
PROFILES = (BASELINE, BF16, ECONOMY)
PRUNED = 'minimax_h3_ref2va_pruned_bf16.safetensors'
REVISION = 'e5eb578a89295337b8ff433a035929ce0279e0b6'
DIFFUSION_OLD = 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'
ENCODER_OLD = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
DIFFUSION = 'minimax_h3_ref2va_bf16.safetensors'
ENCODER = 'qwen3vl_32b_minimax_h3_bf16.safetensors'
DEFAULT_HELPER = 'huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16'
SIMPLE = ('Replace the selected person with the reference character. '
          'Preserve source performance, scene and audio timing.')
UPGRADES = {
    'h3-ref2va-int8': dict(id='h3-ref2va-bf16-full', group='common',
        repo='Comfy-Org/MiniMax-H3', revision=REVISION,
        file='diffusion_models/'+DIFFUSION, destination='diffusion_models/'+DIFFUSION,
        expected_sha256='e32c54c1a7b4f5f397f195cea267ccb18806303bb665678c4bee60953bdf3026'),
    'h3-encoder-nvfp4': dict(id='h3-encoder-bf16', group='common',
        repo='Comfy-Org/MiniMax-H3', revision=REVISION,
        file='text_encoders/'+ENCODER, destination='text_encoders/'+ENCODER,
        expected_sha256='600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d'),
}


def is_bf16(root=None):
    return profile(root) in (BF16, ECONOMY)


def upgrades_for(name):
    result = copy.deepcopy(UPGRADES)
    if name == ECONOMY:
        result['h3-ref2va-int8'] = dict(
            id='h3-ref2va-bf16-pruned', group='common', repo='Comfy-Org/MiniMax-H3',
            revision='3792b245ac4685dda00ebf43a33c646f7ce98064',
            file='diffusion_models/'+PRUNED, destination='diffusion_models/'+PRUNED,
            expected_sha256='37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe')
    return result


def profile(root=None):
    p = Path(root or ROOT)/'precision-profile.json'
    if not p.exists():
        return BASELINE
    value = json.loads(p.read_text())['profile']
    if value not in PROFILES:
        raise ValueError('Unknown compiled H3 precision profile')
    return value


def prepare_build(name, root=None):
    root = Path(root or ROOT)
    if name not in PROFILES:
        raise ValueError('Unknown H3 build profile: '+str(name))
    marker = root/'precision-profile.json'
    if marker.exists() and profile(root) != name:
        raise ValueError('Build profiles must use separate clean image layers')
    specs_path = root/'assets.spec.json'
    data = json.loads(specs_path.read_text())
    upgrades = upgrades_for(name)
    if name in (BF16, ECONOMY):
        ids = {x['id'] for x in data['assets']}
        required = set(upgrades)
        upgraded = {x['id'] for x in upgrades.values()}
        if not (required <= ids or upgraded <= ids):
            raise ValueError('Source asset definitions changed; cannot upgrade safely')
        data['assets'] = [copy.deepcopy(upgrades.get(x['id'], x)) for x in data['assets']]
        atomic_json(specs_path, data)
    if name == ECONOMY:
        data['assets'] = [x for x in data['assets'] if x['group'] != 'bfs']
        atomic_json(specs_path, data)
    atomic_json(marker, {'profile':name, 'version':1,
        'diffusion_file':(PRUNED if name==ECONOMY else DIFFUSION) if name in (BF16, ECONOMY) else DIFFUSION_OLD,
        'encoder_file':ENCODER if name in (BF16, ECONOMY) else ENCODER_OLD,
        'attention':'pytorch attention' if name in (BF16, ECONOMY) else 'comfy kitchen attention',
        'checkpoint_precision':'BF16 source weights, not recast INT8/NVFP4' if name in (BF16, ECONOMY) else 'existing quantized recipe',
        'generation_recipe':'Akatz V1.5 + FL2V Turbo8, both 1.0, 8/er_sde/normal/CFG1',
        'reference_sizing':'match; optional high-reference native canvas uses max',
        'visual_quality_validated':False, 'peak_memory_measured':False})


def _replace(value):
    if isinstance(value, dict):
        return {k:_replace(v) for k,v in value.items()}
    if isinstance(value, list):
        return [_replace(v) for v in value]
    if value == DIFFUSION_OLD: return DIFFUSION
    if value == ENCODER_OLD: return ENCODER
    return value


def _widget(node, key, value):
    named = node.get('widgets_values_named', {})
    if key not in named:
        raise ValueError(f"Cannot safely change {node.get('type')} widget {key}")
    position = list(named).index(key)
    if node['widgets_values'][position] != named[key]:
        raise ValueError('Canvas widget order changed')
    node['widgets_values'][position] = value
    named[key] = value


def transform(document, *, diffusion=DIFFUSION, build_profile=BF16):
    """Only model files, attention and presentation; never sampler/mask/timeline."""
    g = _replace(copy.deepcopy(document))
    if diffusion != DIFFUSION:
        def swap(value):
            if isinstance(value, dict): return {k: swap(v) for k, v in value.items()}
            if isinstance(value, list): return [swap(v) for v in value]
            return diffusion if value == DIFFUSION else value
        g = swap(g)
    if isinstance(g.get('nodes'), list):
        for node in g['nodes']:
            kind = node.get('type')
            if kind == 'ModelAttentionBackend':
                _widget(node, 'attention', 'pytorch attention')
            if kind == 'UNETLoader':
                node['title'] = ('ECONOMY BF16 H3 Ref2VA: pruned BF16, NOT INT8' if build_profile==ECONOMY else 'FULL BF16 H3 Ref2VA: original unpruned weights')
            elif kind == 'CLIPLoader' and node.get('widgets_values_named', {}).get('type') == 'minimax':
                node['title'] = 'BF16 H3 Qwen reference encoder: no NVFP4/AWQ'
        g.setdefault('extra', {})['h3_build_profile'] = build_profile
        g['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(g.get('id'))+'/'+build_profile))
    else:
        for node in g.values():
            if not isinstance(node, dict): continue
            if node.get('class_type') == 'ModelAttentionBackend':
                node['inputs']['attention'] = 'pytorch attention'
    return g


def validate_recipe(api, *, diffusion=DIFFUSION):
    models = [n['inputs']['unet_name'] for n in api.values() if n['class_type']=='UNETLoader']
    encoders = [n['inputs']['clip_name'] for n in api.values() if n['class_type']=='CLIPLoader']
    if models != [diffusion] or encoders != [ENCODER]:
        raise ValueError('BF16 recipe uses unexpected diffusion/text checkpoints')
    for n in api.values():
        v = n['inputs']; k=n['class_type']
        if k=='ModelAttentionBackend' and v['attention']!='pytorch attention':
            raise ValueError('Quantized attention must not be enabled in BF16 profile')
        if k=='KSamplerSelect' and v['sampler_name']!='er_sde': raise ValueError('Sampler changed')
        if k=='BasicScheduler' and (v['steps'],v['scheduler'],v['denoise'])!=(8,'normal',1.):
            raise ValueError('Qualified eight-step schedule changed')
        if k=='CFGGuider' and v['cfg']!=1.: raise ValueError('CFG changed')
    loras = {n['inputs']['lora_name']:n['inputs']['strength_model'] for n in api.values()
             if n['class_type']=='LoraLoaderModelOnly'}
    if loras != {'h3_character_swap_v1.5.safetensors':1.,
        'minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors':1.}:
        raise ValueError('Matched character/Turbo adapters changed')


def finalize_build(root=None):
    """Called after upstream native canvases are assembled in the image build."""
    root=Path(root or ROOT)
    selected = profile(root)
    if selected not in (BF16, ECONOMY): return
    diffusion = PRUNED if selected==ECONOMY else DIFFUSION
    from .graphs import validate_graph
    for p in sorted((root/'workflows').glob('*.json')):
        doc = transform(json.loads(p.read_text()), diffusion=diffusion, build_profile=selected)
        if isinstance(doc.get('nodes'),list): validate_graph(doc)
        atomic_json(p,doc)
    recipe=root/'swapkit/data/akatz_disk_base.api.json'
    api=transform(json.loads(recipe.read_text()), diffusion=diffusion, build_profile=selected); validate_recipe(api, diffusion=diffusion)
    atomic_json(recipe,api)
    # Native 02B retains match for an isolated precision comparison. 02C is an
    # explicit second test: more pixels in each reference, no upscaling invented.
    canvas_path=root/'workflows/02B_Akatz_Native_Simple_Preview.json'
    api_path=canvas_path.with_name('02B_Akatz_Native_Simple_Preview.api.json')
    if not canvas_path.exists() or not api_path.exists():
        raise ValueError('Native preview factory files were not built')
    canvas=json.loads(canvas_path.read_text()); native=json.loads(api_path.read_text())
    validate_recipe(native, diffusion=diffusion)
    for node in canvas['nodes']:
        if node['type']=='MiniMaxH3ReferenceToVideo': _widget(node,'ref_image_size','max')
        if node['type']=='H3SwapSaveVideo':
            _widget(node,'filename_prefix','h3_character_swap/native_bf16_highref/akatz')
    for node in native.values():
        if node['class_type']=='MiniMaxH3ReferenceToVideo': node['inputs']['ref_image_size']='max'
        if node['class_type']=='H3SwapSaveVideo':
            node['inputs']['filename_prefix']='h3_character_swap/native_bf16_highref/akatz'
    canvas['id']=str(uuid.uuid5(uuid.NAMESPACE_URL,canvas['id']+'/highref'))
    canvas['extra']['h3_highref_note']='Optional higher-reference-resolution test; increases memory/time. Same output resolution.'
    validate_graph(canvas)
    atomic_json(root/'workflows/02C_Akatz_BF16_HighRef_Preview.json',canvas)
    atomic_json(root/'workflows/02C_Akatz_BF16_HighRef_Preview.api.json',native)
    record_path=root/'workflow-provenance.json'
    record=json.loads(record_path.read_text()) if record_path.exists() else {}
    record['precision_profile']=json.loads((root/'precision-profile.json').read_text())
    atomic_json(record_path,record)


def cli_flags(root=None):
    # Do not force VAE precision or full GPU residency. Preserve the original
    # FP16 video and FP32 audio checkpoint handling and stage offloading.
    return ['--bf16-unet','--bf16-text-enc','--use-pytorch-cross-attention'] if is_bf16(root) else []


def folder_suffix(root=None):
    return '_ECONOMY' if profile(root)==ECONOMY else ('_BF16' if profile(root)==BF16 else '')


def read_header(path):
    """Read only the safetensors JSON header, not tens of GB of tensor storage."""
    path=Path(path)
    with path.open('rb') as f:
        first=f.read(8)
        if len(first)!=8: raise ValueError('Truncated safetensors header')
        size=struct.unpack('<Q',first)[0]
        if not 2<=size<=64*1024**2 or size+8>path.stat().st_size:
            raise ValueError('Invalid safetensors header size')
        header=json.loads(f.read(size))
    if not isinstance(header,dict): raise ValueError('Invalid safetensors header')
    histogram={}; weight_types=set(); weight_count=0
    for name,entry in header.items():
        if name=='__metadata__': continue
        shape=entry['shape'];dtype=entry['dtype']
        if not isinstance(shape,list) or any(type(x)!=int or x<0 for x in shape):
            raise ValueError('Invalid tensor dimensions')
        histogram[dtype]=histogram.get(dtype,0)+math.prod(shape)
        if name.endswith('.weight') or '.lora_' in name or name.endswith('_weight'):
            weight_types.add(dtype);weight_count+=1
    if not histogram: raise ValueError('No safetensors tensors found')
    return {'parameter_elements_by_dtype':histogram, 'weight_dtypes':sorted(weight_types),
            'weight_tensors':weight_count, 'header_bytes':size}


def audit_weights(root=None, storage=None, selected=('akatz',)):
    root=Path(root or ROOT);storage=Path(storage or data_root())
    rows=json.loads((root/'assets.lock.json').read_text())['assets']
    rows=[r for r in rows if r['group'] in {'common',*selected}]
    records=[]
    for row in rows:
        p=storage/'models'/row['destination'];report=read_header(p)
        if is_bf16(root):
            if any(x not in {'BF16','F16','F32','F64'} for x in report['weight_dtypes']):
                raise ValueError('Unexpected quantized weight dtype: '+row['destination'])
            if row['id'] in {x['id'] for x in upgrades_for(profile(root)).values()}:
                if 'BF16' not in report['weight_dtypes']:
                    raise ValueError('Expected source BF16 weight tensors: '+row['destination'])
        records.append({'id':row['id'],'file':row['destination'],'sha256':row['sha256'],
                        'bytes':row['bytes'],**report})
    result={'profile':profile(root),'check':'Verified-file header audit, not inference',
            'assets':records,'model_bytes':sum(x['bytes'] for x in records),
            'runtime_flags':cli_flags(root),'gpu_render_tested':False}
    atomic_json(storage/'h3-swap/precision-audit.json',result)
    print('[H3 PRECISION] '+json.dumps({'profile':result['profile'],
          'model_bytes':result['model_bytes'],'files':[x['file'] for x in records]}),flush=True)
    return result


def gpu_smoke():
    """Tiny BF16 CUDA/SDPA check. Not an H3 inference or memory-fit benchmark."""
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU unavailable')
    props=torch.cuda.get_device_properties(0)
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('GPU/runtime does not support BF16')
    with torch.inference_mode():
        x=torch.randn((1,4,32,64),device='cuda',dtype=torch.bfloat16)
        y=F.scaled_dot_product_attention(x,x,x)
        z=x[0,0]@x[0,0].T
        torch.cuda.synchronize()
        if not torch.isfinite(y).all() or not torch.isfinite(z).all():
            raise RuntimeError('BF16 CUDA smoke returned non-finite values')
    result={'gpu':props.name,'vram_gib':round(props.total_memory/1024**3,2),
            'compute_capability':list(torch.cuda.get_device_capability(0)),
            'torch':torch.__version__,'cuda':torch.version.cuda,
            'scope':'tiny BF16 matmul and SDPA only; NOT H3 inference/memory validation'}
    print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-build',choices=PROFILES)
    p.add_argument('--audit',action='store_true');p.add_argument('--gpu-smoke',action='store_true')
    a=p.parse_args()
    if a.prepare_build: prepare_build(a.prepare_build)
    elif a.audit: audit_weights()
    elif a.gpu_smoke: gpu_smoke()
    else: print(json.dumps({'profile':profile(),'flags':cli_flags()},indent=2))

if __name__=='__main__': main()
