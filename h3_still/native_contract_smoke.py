"""Real pinned native conditioning functions with synthetic encoder/VAE outputs.

This checks calling conventions and still latent geometry, NOT model inference.
It is a separate CPU process so native imports honor --cpu before initialization.
"""
from pathlib import Path
import importlib.util
import json
import sys

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,'/opt/ComfyUI')
import comfy.options
comfy.options.enable_args_parsing()
from comfy.cli_args import args
args.cpu=True
import torch


def load_package(name,path):
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

class EncoderFixture:
    def tokenize(self,prompt,**kwargs):
        self.last=(prompt,kwargs);return {'fixture':True}
    def encode_from_tokens_scheduled(self,tokens):
        return [[torch.zeros(1,1,8),{}]]

class VAEFixture:
    def encode(self,image):
        return torch.zeros(1,24,1,image.shape[1]//16,image.shape[2]//16)


def main():
    studio=load_package('h3_studio_contract',Path('/opt/ComfyUI/custom_nodes/H3_Still_Studio'))
    fizgig=load_package('fizgig_contract',Path('/opt/ComfyUI/custom_nodes/ComfyUI-Fizgig-H3-Still'))
    encoder=EncoderFixture();vae=VAEFixture();image=torch.zeros(1,96,160,3)
    results=[]
    for mode,count in [('Auto',0),('Auto',1),('Auto',2),('References',1)]:
        refs={'reference_%s'%i:{'name':'synthetic-%s.png'%i,'image':image} for i in range(1,count+1)}
        plan,w,h,steps=studio.H3StudioPlan().prepare(mode,'1:1',.25,.25,'Quality - 50 steps, no Turbo',.25,**refs)
        prompt='A natural photograph.' if count==0 else 'Use '+', '.join('<Picture %s>'%i for i in range(1,count+1))+'.'
        cond=studio.H3StudioConditioning().encode(plan,encoder,vae,prompt)[0]
        assert encoder.last[0]==prompt
        blocks=cond[0][1].get('minimax_refs',[])
        assert len(blocks)==count
        latent=fizgig.FizgigH3StillLatent().make(w,h,1)[0]['samples']
        video,audio=latent.unbind()
        assert tuple(video.shape)==(1,24,1,h//16,w//16)
        assert tuple(audio.shape)==(1,32,2,2)
        results.append({'mode':plan['mode'],'references':count,'single_frame_latent':True,
                        'native_conditioning_called':True,'encoder_and_vae':'synthetic fixtures'})
    # Real native video-conditioning path; only encoder/VAE weights are synthetic.
    media=sys.modules['h3_studio_contract.media_nodes']
    seen=[]
    class VideoVAEFixture:
        def encode(self,pixels):
            frames=int(pixels.shape[0]);seen.append(frames)
            latent_t=1 if frames==1 else 2+5*((frames-5)//17)
            return torch.zeros(1,24,latent_t,pixels.shape[1]//16,pixels.shape[2]//16)
    frames=torch.zeros(73,96,160,3)
    plan,w,h=media.H3StudioMediaPlan().prepare('References','2:3',.25,.25,
        'Quality - 50 steps, no Turbo',.25,False,
        video={'frames':frames,'summary':{'reference_frames':73,'fps':24}},
        reference_1={'name':'synthetic-face.png','image':image})
    cond=media.H3StudioMediaConditioning().encode(plan,encoder,VideoVAEFixture(),
        'Use <Video 1> for the person and <Picture 1> for the face.')[0]
    assert 73 in seen, seen
    assert plan['conditioning_length']==73
    blocks=cond[0][1]['minimax_refs']
    assert len(blocks)==2 and blocks[1]['kind']=='video'
    assert blocks[1]['latent_t']==22
    video_tokens=encoder.last[1]['minimax_ref_items'][1]
    assert video_tokens['type']=='video' and video_tokens['data'].shape[0]==7
    latent=fizgig.FizgigH3StillLatent().make(w,h,1)[0]['samples']
    assert latent.unbind()[0].shape[2]==1
    results.append({'mode':'video-plus-photo reference','reference_frames':73,'native_vae_received_frames':73,
                    'tokenizer_sampled_frames':7,'single_frame_latent':True,
                    'encoder_and_vae':'synthetic fixtures, native H3 calling convention'})
    target=PACKAGE/'native-contract-report.json'
    target.write_text(json.dumps({'checks':results,'model_inference':False},indent=2)+'\n')
    print(target.read_text())

if __name__=='__main__':main()
