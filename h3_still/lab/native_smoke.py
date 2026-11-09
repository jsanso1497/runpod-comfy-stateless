"""Run Lab reference routing against real ComfyUI conditioning with synthetic encoder/VAE fixtures."""
import importlib.util
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
import torch
import nodes


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    nodes.NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS);return module

class EncoderFixture:
    def tokenize(self,prompt,**kwargs):self.last=(prompt,kwargs);return {}
    def encode_from_tokens_scheduled(self,tokens):return [[torch.zeros(1,1,8),{}]]
class VAEFixture:
    def encode(self,image):return torch.zeros(1,24,1,image.shape[1]//16,image.shape[2]//16)


def main():
    studio=load('lab_contract_studio',Path('/opt/ComfyUI/custom_nodes/H3_Still_Studio'))
    lab=load('lab_contract_nodes',Path('/opt/ComfyUI/custom_nodes/H3_Studio_Lab'))
    img=torch.rand(1,96,160,3);ref={'image':img,'name':'source.png'};enc=EncoderFixture();vae=VAEFixture()
    prepare=lab.NODE_CLASS_MAPPINGS['H3LabPlan']();encode=lab.NODE_CLASS_MAPPINGS['H3LabConditioning']()
    reports=[]
    for mode,count in [('Auto',0),('Auto',1),('References',2),('Replace person',3)]:
        refs={'reference_%s'%i:dict(ref) for i in range(1,count+1)}
        plan,w,h=prepare.prepare(mode,'2:3',.25,'Quality - 50 steps, no Turbo',0,'match',False,**refs)
        prompt='Create a photograph.' if not count else 'Use <Picture 1> as the main reference.'
        cond,report=encode.encode(plan,enc,vae,prompt)
        assert len(cond[0][1].get('minimax_refs',[]))==count
        assert enc.last[0]==prompt
        reports.append({'mode':plan['mode'],'photos':count,'dimensions':[w,h]})
    # The default replacement remains equivalent at the conditioning boundary.
    refs={f'reference_{i}':dict(ref) for i in range(1,4)}
    a,w,h=prepare.prepare('Replace person','2:3',.25,'Quality - 50 steps, no Turbo',0,'match',False,**refs)
    b,_,_=studio.MEDIA_NODE_CLASS_MAPPINGS['H3StudioMediaPlan']().prepare('Replace person','2:3',.25,.25,'Quality - 50 steps, no Turbo',2.5,False,**refs)
    assert all(torch.equal(x,y) for x,y in zip(a['images'],b['images']))
    assert (a['model'],a['steps'],a['sampler'],a['scheduler'])==(b['model'],b['steps'],b['sampler'],b['scheduler'])
    crop=lab.NODE_CLASS_MAPPINGS['H3LabReference']().prepare(ref,'face',.25,25.,0.,25.,0.)
    assert crop[0]['lab_preprocess']['crop']==[40,0,120,96]
    source=img.clone();mask=torch.zeros(1,96,160);mask[:,:,40:120]=1
    plan,_,_=prepare.prepare('Replace person','2:3',.25,'Quality - 50 steps, no Turbo',0,'match',True,reference_1=dict(ref,mask=mask),reference_2=dict(ref))
    finish=studio.MEDIA_NODE_CLASS_MAPPINGS['H3StudioMediaFinish']().finish(plan,torch.zeros(1,64,96,3))[0]
    assert torch.equal(finish[mask==0],source[mask==0])
    profile=(P/'build-profile.txt').read_text().strip()
    if profile=='ph3':
        from comfy_extras.nodes_minimax_h3 import PH3ImageConditioning
        out=PH3ImageConditioning.execute(enc,'Edit <Picture 1>. Replace the subject.',width=160,height=96,vae=vae,images={'image_1':img})
        values=out.result if hasattr(out,'result') else out
        assert values[1]['samples'].tensors[0].shape[2]==1
    report={'checks':reports,'baseline_equivalence':'passed at reference tensors and sampling configuration',
        'unmasked_pixel_preservation':'passed','native_conditioning':'passed with synthetic encoder/VAE',
        'profile':profile,'real_model_weights':False,'gpu_inference':False}
    (P/'native-lab-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
