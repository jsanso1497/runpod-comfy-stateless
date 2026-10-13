"""Optional, composable utilities. No model is downloaded or loaded on import."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from workbench.registry import choices, check_selection, lora_choices

CATEGORY='Workbench / Toolbox'
FAMILIES=['qwen','flux','h3','restoration']
SPECS={
 'WBQwenUNETLoader':('qwen','UNETLoader','diffusion_models','unet_name',None),
 'WBQwenCLIPLoader':('qwen','CLIPLoader','text_encoders','clip_name','qwen_image'),
 'WBQwenVAELoader':('qwen','VAELoader','vae','vae_name',None),
 'WBFluxUNETLoader':('flux','UNETLoader','diffusion_models','unet_name',None),
 'WBFluxCLIPLoader':('flux','CLIPLoader','text_encoders','clip_name','flux2'),
 'WBFluxVAELoader':('flux','VAELoader','vae','vae_name',None),
 'SafeH3UNETLoader':('h3','UNETLoader','diffusion_models','unet_name',None),
 'SafeH3CLIPLoader':('h3','CLIPLoader','text_encoders','clip_name','minimax'),
 'SafeH3VideoVAELoader':('h3','VAELoader','vae','vae_name',None),
 'SafeH3AudioVAELoader':('h3','VAELoader','vae','vae_name',None),
}

class _FamilyLoader:
    CATEGORY='Workbench / Compatible models'
    FUNCTION='load'
    @classmethod
    def INPUT_TYPES(cls):
        family,native,kind,field,encoder=SPECS[cls.__name__]
        files=choices(family,kind)
        if cls.__name__=='SafeH3VideoVAELoader':files=[n for n in files if 'audio' not in n]
        if cls.__name__=='SafeH3AudioVAELoader':files=[n for n in files if 'audio' in n]
        required={field:(files or ['[prepare models]'],)}
        if native=='UNETLoader':required['weight_dtype']=(['default'],{'default':'default'})
        if native=='CLIPLoader':required['type']=([encoder],{'default':encoder})
        result={'required':required}
        if native=='CLIPLoader':result['optional']={'device':(['default','cpu'],{'default':'default'})}
        return result
    @classmethod
    def VALIDATE_INPUTS(cls,**kwargs):
        family,native,kind,field,encoder=SPECS[cls.__name__]
        # Linked filename inputs are evaluated later. Validate their actual value in load().
        if field not in kwargs:return True
        try:
            check_selection(family,kind,kwargs[field])
            if encoder and kwargs.get('type',encoder)!=encoder:raise ValueError('Encoder family mismatch.')
            if native=='UNETLoader' and kwargs.get('weight_dtype','default')!='default':raise ValueError('Automatic precision substitution is disabled.')
        except (ValueError,FileNotFoundError,KeyError) as exc:return str(exc)
        return True
    def load(self,**kwargs):
        error=self.VALIDATE_INPUTS(**kwargs)
        if error is not True:raise ValueError(error)
        import nodes
        _,native,_,field,encoder=SPECS[self.__class__.__name__]
        if native=='UNETLoader':return nodes.UNETLoader().load_unet(kwargs[field],'default')
        if native=='CLIPLoader':return nodes.CLIPLoader().load_clip(kwargs[field],encoder,kwargs.get('device','default'))
        return nodes.VAELoader().load_vae(kwargs[field])

NODE_CLASS_MAPPINGS={}
for name,(_,native,*_) in SPECS.items():
    cls=type(name,(_FamilyLoader,),{'RETURN_TYPES':({'UNETLoader':('MODEL',),'CLIPLoader':('CLIP',),'VAELoader':('VAE',)}[native])})
    globals()[name]=cls;NODE_CLASS_MAPPINGS[name]=cls

class WBCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names=list(dict.fromkeys(n for family in FAMILIES for n in choices(family,'checkpoints')))
        return {'required':{'family':(FAMILIES,), 'checkpoint':(names or ['[register a private checkpoint]'],)}}
    RETURN_TYPES=('MODEL','CLIP','VAE')
    FUNCTION='load';CATEGORY='Workbench / Compatible models'
    DESCRIPTION='Only for a declared all-in-one checkpoint that ComfyUI can load. Split diffusion checkpoints belong in their family UNET loader. Does not change Standard HQ workflows.'
    def load(self,family,checkpoint):
        import nodes
        check_selection(family,'checkpoints',checkpoint)
        try:return nodes.CheckpointLoaderSimple().load_checkpoint(checkpoint)
        except Exception as exc:raise ValueError('ComfyUI could not load this declared all-in-one checkpoint. Register split model components under diffusion_models, text_encoders and vae instead.') from exc

class WBOptionalLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        files=list(dict.fromkeys(n for f in FAMILIES for n in lora_choices(f)))
        return {'required':{'model':('MODEL',),'family':(FAMILIES,), 'enabled':('BOOLEAN',{'default':False}),
                'lora_name':(['None']+files,), 'strength':('FLOAT',{'default':1.,'min':-2.,'max':2.,'step':.05})}}
    RETURN_TYPES=('MODEL',);FUNCTION='apply';CATEGORY=CATEGORY
    def apply(self,model,family,enabled=False,lora_name='None',strength=1.):
        if not enabled or lora_name=='None' or strength==0:return (model,)
        if lora_name not in lora_choices(family):raise ValueError('LoRA is not registered for this model family.')
        import nodes
        return nodes.LoraLoaderModelOnly().load_lora_model_only(model,lora_name,strength)


def validate_mask(mask:torch.Tensor)->None:
    if mask.ndim!=3 or not torch.isfinite(mask).all() or mask.min()<0 or mask.max()>1:
        raise ValueError('Mask must be a finite [batch,height,width] tensor from 0 to 1.')

def feather_mask(mask:torch.Tensor,expand:int,feather:int,invert:bool=False)->torch.Tensor:
    validate_mask(mask)
    if not 0<=expand<=4096 or not 0<=feather<=4096:raise ValueError('Expansion and feather must be between 0 and 4096 pixels.')
    base=(1-mask if invert else mask).detach().float().cpu().numpy()
    if expand==0 and feather==0:return torch.from_numpy(base.copy()).to(mask.device)
    outputs=[]
    for plane in base:
        solid=plane>=0.5
        if not solid.any():outputs.append(np.zeros_like(plane));continue
        distance=distance_transform_edt(~solid)
        if feather:
            t=np.clip((distance-expand)/feather,0,1)
            alpha=(0.5+0.5*np.cos(np.pi*t)).astype(np.float32)
            alpha[distance>=expand+feather]=0.
        else:alpha=(distance<=expand).astype(np.float32)
        alpha[solid]=1.
        outputs.append(alpha)
    return torch.from_numpy(np.stack(outputs)).to(mask.device)

class WBMaskExpandFeather:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'mask':('MASK',),'expand_pixels':('INT',{'default':32,'min':0,'max':4096}),
           'feather_pixels':('INT',{'default':128,'min':0,'max':4096}), 'invert':('BOOLEAN',{'default':False})}}
    RETURN_TYPES=('MASK',);RETURN_NAMES=('expanded_feathered_mask',);FUNCTION='process';CATEGORY=CATEGORY
    DESCRIPTION='Expand the white selection, then add an outward cosine fade. Fade is exactly zero outside its finite radius. Dimensions stay unchanged.'
    def process(self,mask,expand_pixels=32,feather_pixels=128,invert=False):return (feather_mask(mask,expand_pixels,feather_pixels,invert),)

class WBComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'original_a':('IMAGE',),'candidate_b':('IMAGE',),'mask':('MASK',)},'optional':{'protect_mask':('MASK',)}}
    RETURN_TYPES=('IMAGE','STRING');RETURN_NAMES=('composited_image','preservation_report');FUNCTION='composite';CATEGORY=CATEGORY
    DESCRIPTION='White reveals image B. Black preserves original A exactly. Both images must already have the same geometry. Protection is applied after feathering.'
    def composite(self,original_a,candidate_b,mask,protect_mask=None):
        if original_a.ndim!=4 or original_a.shape[-1] not in (3,4) or original_a.shape!=candidate_b.shape:
            raise ValueError('Original A and candidate B must be aligned and have exactly matching canvas size, batch size and channels. This node does not align viewpoints.')
        if not torch.isfinite(original_a).all() or not torch.isfinite(candidate_b).all():raise ValueError('Image contains non-finite pixels.')
        validate_mask(mask)
        if mask.shape!=original_a.shape[:3]:raise ValueError('Selection mask must be made on original image A at its exact canvas size.')
        alpha=mask.to(original_a.device,dtype=original_a.dtype)
        if protect_mask is not None:
            validate_mask(protect_mask)
            if protect_mask.shape!=alpha.shape:raise ValueError('Protection mask must match original A.')
            alpha=torch.where(protect_mask.to(alpha.device)>0,torch.zeros_like(alpha),alpha)
        b=candidate_b.to(original_a.device,dtype=original_a.dtype)
        result=original_a*(1-alpha[...,None])+b*alpha[...,None]
        result=torch.where((alpha==0)[...,None],original_a,result)
        result=torch.where((alpha==1)[...,None],b,result)
        protected=(alpha==0)
        delta=(result-original_a).abs().amax(-1)
        error=float(delta[protected].max()) if protected.any() else 0.
        if error!=0:raise ValueError('Unselected pixels changed; composite was rejected.')
        return result,json.dumps({'outside_selection_max_error':error,'protected_pixels':int(protected.sum()),'width':original_a.shape[2],'height':original_a.shape[1]})

class WBMaskPreview:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'original_a':('IMAGE',),'mask':('MASK',)},'optional':{'protect_mask':('MASK',)}}
    RETURN_TYPES=('IMAGE',);FUNCTION='preview';CATEGORY=CATEGORY
    def preview(self,original_a,mask,protect_mask=None):
        validate_mask(mask)
        if mask.shape!=original_a.shape[:3]:raise ValueError('Mask dimensions must match original A.')
        color=torch.tensor([0.28,0.85,0.7],device=original_a.device,dtype=original_a.dtype)
        alpha=mask.to(original_a.device)[...,None]*.42
        out=original_a[...,:3]*(1-alpha)+color*alpha
        if protect_mask is not None:
            validate_mask(protect_mask)
            if protect_mask.shape!=mask.shape:raise ValueError('Protection dimensions do not match.')
            p=(protect_mask.to(original_a.device)>0)[...,None]*.5
            out=out*(1-p)+torch.tensor([1.,.35,.35],device=out.device)*p
        return (out,)

class WBSAMMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'original_a':('IMAGE',),'object_name':('STRING',{'default':'shirt','multiline':True}),
           'threshold':('FLOAT',{'default':.5,'min':.05,'max':.95,'step':.05}), 'instance_index':('INT',{'default':-1,'min':-1,'max':63})}}
    RETURN_TYPES=('MASK','STRING');FUNCTION='select';CATEGORY=CATEGORY
    DESCRIPTION='Select a named object from original A, never candidate B. Uses the separate SAM model and does not load Qwen generation weights.'
    def select(self,original_a,object_name='shirt',threshold=.5,instance_index=-1):
        import nodes
        cls=nodes.NODE_CLASS_MAPPINGS.get('Q21PhotoMask')
        if cls is None:raise ValueError('Text-SAM utilities are supported in the Qwen workspace. Use its Toolbox or a manual mask here.')
        return cls().make_mask(source=original_a,auto_mask=True,sam_prompt=object_name,threshold=threshold,
          refine_iterations=2,instance_index=instance_index,search_region='full photo',manual_correction='ignore',expand_pixels=0,manual_mask=None)

for cls in [WBCheckpointLoader,WBOptionalLoRA,WBMaskExpandFeather,WBComposite,WBMaskPreview,WBSAMMask]:NODE_CLASS_MAPPINGS[cls.__name__]=cls
NODE_DISPLAY_NAME_MAPPINGS={
 'WBComposite':'Workbench | Original A + Selected Parts of B',
 'WBMaskExpandFeather':'Workbench | Expand + Outward Feather + Invert',
 'WBMaskPreview':'Workbench | Preview Selection / Protection',
 'WBSAMMask':'Workbench | Select Object in Original A',
 'WBOptionalLoRA':'Workbench | Optional Family-Matched LoRA',
 'WBCheckpointLoader':'Workbench | Registered All-in-One Checkpoint',
}
