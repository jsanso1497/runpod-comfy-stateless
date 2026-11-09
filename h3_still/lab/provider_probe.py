"""Check real pinned provider imports/contracts on CPU. No H3 weights or CUDA kernels are loaded."""
import importlib.util
import inspect
import json
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
sys.path.insert(0,'/opt/ComfyUI');sys.path.insert(0,str(P))
sys.argv=[sys.argv[0],'--cpu']
import comfy.options
comfy.options.enable_args_parsing()
from comfy.cli_args import args
args.cpu=True
import nodes
from nodepack.common import provider,provider_node,required_defaults


def main():
    report={'weights_loaded':False,'gpu_tested':False,'providers':{}}
    spectrum=provider_node('spectrum','SpectrumApplyMiniMaxH3')
    values=required_defaults(spectrum,{'model':object(),'enabled':True,'debug':True,
        'model_aware_mode':'full','offline_smoothing_replay':False,'model_aware_trust_shrinkage':False,
        'history_storage':'system_ram','audio_blend_weight':0.,'degree':1,'warmup_steps':1,'tail_actual_steps':1})
    inspect.signature(getattr(spectrum(),spectrum.FUNCTION)).bind(**values)
    report['providers']['spectrum']={'import':'passed','schema_and_call_signature':'passed','execution':'requires GPU test'}
    refs=provider('refmods').NODE_CLASS_MAPPINGS
    for name in ('MiniMaxH3RefModExtract','MiniMaxH3RefModsLoader','MiniMaxH3RefModTextEncode'):
        if name not in refs:raise RuntimeError('Missing RefMod interface: '+name)
    inspect.signature(refs['MiniMaxH3RefModExtract'].execute).bind(name='test',mode='encode',vae=object(),refs_image={'ref_image_1':None},subfolder='studio_lab',budget_policy='error')
    inspect.signature(refs['MiniMaxH3RefModsLoader']().load).bind(show_info=True,max_total_tokens=65536,mod_1='test',strength_1=1.,copies_1=1,components_1='Visual')
    inspect.signature(refs['MiniMaxH3RefModTextEncode']().encode).bind(clip=object(),mods=[],prompt='test',reference_fps=24.,max_total_tokens=65536,vae=object())
    report['providers']['refmods']={'import':'passed','call_signatures':'passed','model_encode_decode':'requires real weights'}
    report['providers']['spark']={'cpu_test':'not applicable','reason':'SM86 GPU and optional compiled backend required'}
    (P/'provider-contract-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
