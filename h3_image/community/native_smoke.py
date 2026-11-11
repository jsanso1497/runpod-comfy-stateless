#!/usr/bin/env python3
"""Build-time native CPU constructor checks; no models or GPU inference."""
import json
from pathlib import Path
import sys


def unpack(value):
    if hasattr(value,'result'):return value.result
    if isinstance(value,dict):return value['result']
    return value


def main():
    core=Path('/opt/ComfyUI')
    if not (core/'comfy').is_dir():raise RuntimeError('Run inside the workspace image.')
    sys.path.insert(0,str(core))
    sys.argv=['community-native-smoke','--cpu','--cache-none','--disable-pinned-memory']
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.cli_args import args
    if not args.cpu:raise RuntimeError('CPU-only arguments were not applied.')
    import torch
    from comfy import model_management
    if model_management.get_torch_device().type!='cpu':raise RuntimeError('CPU mode required.')
    from comfy.ldm.minimax.sampling import MiniMaxH3ImageModelSampling
    from comfy_extras.nodes_custom_sampler import BasicScheduler,KSamplerSelect
    from comfy_extras.nodes_advanced_samplers import SamplerLCM
    package=Path(__file__).resolve().parents[1]
    expected=json.loads((package/'pdd_config.json').read_text())['sigmas']
    class Config:
        unet_config={'pdd_sigmas':expected}
        sampling_settings={}
    sampling=MiniMaxH3ImageModelSampling(Config())
    class Model:
        def get_model_object(self,name):
            if name!='model_sampling':raise ValueError(name)
            return sampling
    report={'device':'cpu','model_weights_loaded':False,'gpu_generation_tested':False,
            'scope':'Real native scheduler/sampler constructors only, not image generation.', 'schedules':{}}
    for count in (4,6):
        sigmas=unpack(BasicScheduler.execute(Model(),'simple',count,1.0))[0]
        assert len(sigmas)==count+1
        assert torch.isfinite(sigmas).all()
        assert bool(torch.all(sigmas[:-1]>=sigmas[1:]))
        assert float(sigmas[-1])==0.0
        report['schedules'][str(count)]=sigmas.tolist()
    reference=torch.tensor(expected,dtype=torch.float32)
    actual=torch.tensor(report['schedules']['6'],dtype=torch.float32)
    report['simple6_max_difference_from_uploaded_grid']=float(torch.max(torch.abs(reference-actual)))
    report['native_samplers']={}
    for name in ('euler','er_sde'):
        obj=unpack(KSamplerSelect.execute(name))[0]
        assert callable(obj.sampler_function)
        report['native_samplers'][name]=getattr(obj.sampler_function,'__name__','callable')
    for value in (0.95,1.0,1.1,1.2):
        obj=unpack(SamplerLCM.execute(value,value,0.0))[0]
        assert callable(obj.sampler_function)
        assert obj.extra_options['s_noise']==value
        assert obj.extra_options['s_noise_end']==value
        assert obj.extra_options['noise_clip_std']==0.0
    report['lcm_constant_noise_values_constructed']=[0.95,1.0,1.1,1.2]
    (package/'community/native-constructor-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
