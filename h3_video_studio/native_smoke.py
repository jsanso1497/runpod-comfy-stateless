"""Build-time real upstream API contracts with synthetic encoder/VAE fixtures; no weights."""
import importlib.util
import inspect
import json
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,'/opt/ComfyUI');sys.argv=[sys.argv[0],'--cpu']
import comfy.options
comfy.options.enable_args_parsing()
from comfy.cli_args import args
args.cpu=True
import torch
import nodes

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    nodes.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS);return module

class Encoder:
    def tokenize(self,prompt,**kwargs):self.last=(prompt,kwargs);return {}
    def encode_from_tokens_scheduled(self,tokens):return [[torch.zeros(1,1,8),{}]]
class VAE:
    def encode(self,image):return torch.zeros(1,24,1,image.shape[1]//16,image.shape[2]//16)

def main():
    studio=load('video_contract_studio',Path('/opt/ComfyUI/custom_nodes/H3_Still_Studio'))
    fizgig=load('video_contract_fizgig',Path('/opt/ComfyUI/custom_nodes/ComfyUI-Fizgig-H3-Still'))
    custom=load('video_contract_new',Path('/opt/ComfyUI/custom_nodes/H3_Video_Still_Studio'))
    from comfy_extras.nodes_minimax_h3 import (MiniMaxH3ImageToVideo,MiniMaxH3ReferenceToVideo,MiniMaxH3SigmaShift,PH3ImageConditioning)
    from comfy_extras.nodes_custom_sampler import RandomNoise,BasicGuider,BasicScheduler,SamplerCustomAdvanced
    get=custom.nodes.result
    import comfy.lora
    import comfy.sd
    assert callable(comfy.lora.model_lora_keys_unet) and callable(comfy.lora.load_lora)
    assert callable(comfy.sd.load_bypass_lora_for_models)
    assert get(RandomNoise.execute(123)).__len__()==1
    for cls in (BasicGuider,BasicScheduler,SamplerCustomAdvanced,MiniMaxH3SigmaShift):assert callable(cls.execute)
    image=torch.rand(1,96,160,3);enc=Encoder();vae=VAE();checks=[]
    cond=get(MiniMaxH3ImageToVideo.execute(enc,vae,'A still photograph.',160,96,5))[0]
    assert len(cond)==1;checks.append('native text conditioning')
    for with_video in (False,True):
        output=MiniMaxH3ReferenceToVideo.execute(clip=enc,vae=vae,prompt='Use <Picture 1>.',width=160,height=96,
            length=22 if with_video else 5,ref_image_size='max',ref_images={'ref_image_0':image},
            ref_videos={'ref_video_0':image.repeat(22,1,1,1)} if with_video else {})
        assert len(get(output)[0][0][1].get('minimax_refs',[]))==1+int(with_video)
        checks.append('native photo plus video conditioning' if with_video else 'native photo conditioning')
    output=PH3ImageConditioning.execute(clip=enc,prompt='Edit <Picture 1>. Change the shirt.',width=160,height=96,
        vae=vae,images={'image_1':image})
    assert get(output)[1]['samples'].tensors[0].shape[2]==1;checks.append('native P-H3 single-frame conditioning')
    latent=custom.nodes.node_call('FizgigH3StillLatent',width=160,height=96,batch_size=1)[0]
    assert latent['samples'].tensors[0].shape[2]==1;checks.append('Fizgig T=1 latent contract')
    decoder=nodes.NODE_CLASS_MAPPINGS['FizgigH3StillDecode']
    inputs=decoder.INPUT_TYPES()
    assert {'samples','vae'}.issubset(inputs['required']);checks.append('Fizgig decoder input contract')
    report=dict(checks=checks,real_encoder_weights=False,real_diffusion_weights=False,gpu_inference=False)
    (P/'native-contract-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
