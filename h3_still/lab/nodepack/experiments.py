"""Explicit, independently switchable experiments. Nothing is downloaded during generation."""
import inspect
import json
import math
from pathlib import Path
import time
from .common import (CATEGORY, ROOT, LAB, NONE, registry, outputs, invoke, provider,
                     provider_node, required_defaults, short_name, dimensions, text_ui, cuda_sync)


class H3LabExperiment:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model':('MODEL',),'plan':('H3_STILL_PLAN',),
            'character_swap_adapter':('BOOLEAN',{'default':False}),
            'character_swap_strength':('FLOAT',{'default':1.,'min':0.,'max':2.,'step':.05}),
            'accelerator':(['None','Spectrum','Spark SM86'],{'default':'None'}),
            'allow_combined_experiment':('BOOLEAN',{'default':False})}}
    RETURN_TYPES=('MODEL','STRING')
    RETURN_NAMES=('model','experiment_report')
    FUNCTION='apply'
    CATEGORY=CATEGORY
    DESCRIPTION='Baseline is unchanged with all controls OFF. Test one modification at a time.'

    def apply(self,model,plan,character_swap_adapter,character_swap_strength,accelerator,allow_combined_experiment):
        enabled_count=int(character_swap_adapter and character_swap_strength!=0)+int(accelerator!='None')+int(bool(plan.get('adapter')))
        if enabled_count>1 and not allow_combined_experiment:
            raise ValueError('One experiment at a time: disable other adapters/Turbo/acceleration, or explicitly enable allow_combined_experiment.')
        report={'accelerator':accelerator,'swap_adapter':bool(character_swap_adapter),
                'swap_strength':character_swap_strength,'combined_test':allow_combined_experiment,
                'warning':'Approximate/adapted still-image experiments, not verified quality improvements.'}
        if character_swap_adapter and character_swap_strength:
            if 'ref2va' not in plan.get('model',''):
                raise ValueError('Character Swap V1.5 requires the REF2VA route.')
            import folder_paths
            name='h3_character_swap_v1.5.safetensors'
            if not folder_paths.get_full_path('loras',name):
                raise RuntimeError('Download once in Jupyter Terminal: python3 /opt/h3-still/lab/manage.py download swap')
            before=sum(len(v) for v in getattr(model,'patches',{}).values())
            model=registry('LoraLoaderModelOnly')().load_lora_model_only(model,name,character_swap_strength)[0]
            after=sum(len(v) for v in getattr(model,'patches',{}).values())
            report['adapter_patch_delta']=after-before
            if hasattr(model,'patches') and after<=before:
                raise RuntimeError('Character adapter did not add any model patches. This test did not apply the adapter.')
        if accelerator=='Spectrum':
            cls=provider_node('spectrum','SpectrumApplyMiniMaxH3')
            values=required_defaults(cls,{'model':model,'enabled':True,'debug':True,
                'model_aware_mode':'full','offline_smoothing_replay':False,
                'model_aware_trust_shrinkage':False,'history_storage':'system_ram',
                'audio_blend_weight':0.,'degree':1,'warmup_steps':1,'tail_actual_steps':1})
            model=outputs(getattr(cls(),cls.FUNCTION)(**values))[0]
            report['spectrum']={k:v for k,v in values.items() if k!='model'}
        elif accelerator=='Spark SM86':
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(8,6):
                raise ValueError('The packaged Spark test requires an SM86 GPU such as A40. It is not enabled on A100/SM80.')
            mapping=provider('spark').NODE_CLASS_MAPPINGS
            candidates=[(name,cls) for name,cls in mapping.items()
                        if 'SparkAttention' in name and ('SM86' in name or name.endswith('Auto'))]
            if not candidates:
                raise RuntimeError('Spark SM86 node not found in the pinned provider.')
            preferred=sorted(candidates,key=lambda pair:('SM86' not in pair[0],pair[0]))[0]
            name,cls=preferred
            # Require the reviewed controls rather than assuming an unrelated method has the same inputs.
            fields={**cls.INPUT_TYPES().get('required',{}),**cls.INPUT_TYPES().get('optional',{})}
            values={'model':model, 'steps':int(plan['steps']), 'strict':True}
            # Actual field names are verified by the GPU setup script before this node is used.
            for key,value in [('mode','fidelity'),('dense_backend','sdpa'),('sparse_scope','joint-video')]:
                if key not in fields:raise RuntimeError('Spark contract changed: missing '+key)
                if isinstance(fields[key][0],(list,tuple)) and value not in fields[key][0]:
                    raise RuntimeError('Spark value is not supported: %s=%s'%(key,value))
                values[key]=value
            values=required_defaults(cls,values)
            model=outputs(getattr(cls(),cls.FUNCTION)(**values))[0]
            report['spark_node']=name
            report['spark']={k:v for k,v in values.items() if k!='model'}
        elif accelerator!='None':raise ValueError('Unknown accelerator.')
        return model,json.dumps(report,indent=2)


class H3LabRefModSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'reference':('H3_STILL_REFERENCE',),'vae':('VAE',),
            'name':('STRING',{'default':'person_face_v1'}),
            'mode':(['Full reference','Compressed experimental'],{'default':'Full reference'}),
            'short_edge':('INT',{'default':1024,'min':128,'max':2048,'step':32})}}
    RETURN_TYPES=('STRING',)
    RETURN_NAMES=('details',)
    OUTPUT_NODE=True
    FUNCTION='save'
    CATEGORY=CATEGORY
    def save(self,reference,vae,name,mode,short_edge):
        short_name(name)
        image=reference.get('image')
        if image is None:raise ValueError('Load one reference to save.')
        root=ROOT/'models/refmods/studio_lab';root.mkdir(parents=True,exist_ok=True)
        if (root/(name+'.safetensors')).exists():
            raise FileExistsError('That saved reference exists. Use a new version name; existing references are not overwritten.')
        import folder_paths
        folder_paths.add_model_folder_path('refmods',str(ROOT/'models/refmods'),is_default=True)
        node=provider_node('refmods','MiniMaxH3RefModExtract')
        result=invoke(node,name=name,mode='encode' if mode=='Full reference' else 'training',
            refs_image={'ref_image_1':image},vae=vae,ref_resolution=short_edge,
            pool_h=32,pool_w=32,latent_frames=1,identity=0,multiplier=1,max_tokens=65536,
            description=reference.get('lab_role','user reference'),save=True,
            concept_type='identity',subfolder='studio_lab',merge=False,motion_only=False,
            extraction_preset='manual',budget_policy='error')
        details=str(result[1])
        return text_ui(details,(details,))


class H3LabRefModLoad:
    @classmethod
    def INPUT_TYPES(cls):
        root=ROOT/'models/refmods'
        names=sorted(p.relative_to(root).with_suffix('').as_posix() for p in root.rglob('*.safetensors')) if root.is_dir() else []
        return {'required':{'face_refmod':([NONE]+names,), 'body_refmod':([NONE]+names,),
                            'face_strength':('FLOAT',{'default':1.,'min':0.,'max':1.,'step':.05}),
                            'body_strength':('FLOAT',{'default':1.,'min':0.,'max':1.,'step':.05})}}
    RETURN_TYPES=('H3_REF_MODS','STRING')
    RETURN_NAMES=('mods','mapping')
    FUNCTION='load'
    CATEGORY=CATEGORY
    @classmethod
    def IS_CHANGED(cls,**values):
        # A saved file can be externally replaced. Invalidate only this inexpensive loader.
        return float('nan')
    def load(self,face_refmod,body_refmod,face_strength,body_strength):
        import folder_paths
        folder_paths.add_model_folder_path('refmods',str(ROOT/'models/refmods'),is_default=True)
        cls=provider_node('refmods','MiniMaxH3RefModsLoader')
        result=cls().load(show_info=True,max_total_tokens=65536,
            mod_1=face_refmod,strength_1=face_strength,copies_1=1,components_1='Visual',
            mod_2=body_refmod,strength_2=body_strength,copies_2=1,components_2='Visual')
        # The provider's text encoder skips zero-strength entries. Keep tag counts identical.
        mods=[(mod,strength) for mod,strength in result[0] if strength > 0]
        if not mods:raise ValueError('Select at least one saved single-photo RefMod.')
        if any(mod.kind!='image' for mod,_ in mods):
            raise ValueError('This controlled test accepts single-photo RefMods only. Save face and body separately.')
        mapping='\n'.join('<Picture %s> = %s'%(i+1,mod.name) for i,(mod,_) in enumerate(mods))
        return mods,mapping


class H3LabRefModPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'mods':('H3_REF_MODS',),
            'aspect_ratio':(['1:1','2:3','3:2','4:5','9:16','16:9'],{'default':'2:3'}),
            'output_megapixels':('FLOAT',{'default':2.5,'min':.25,'max':8.,'step':.25})}}
    RETURN_TYPES=('H3_STILL_PLAN','INT','INT')
    FUNCTION='prepare'
    CATEGORY=CATEGORY
    def prepare(self,mods,aspect_ratio,output_megapixels):
        cls=registry('H3StudioMediaPlan')
        module=__import__(cls.__module__,fromlist=['media_plan'])
        plan=module.media_plan('References',len(mods),0,aspect_ratio,output_megapixels,output_megapixels,
                               'Quality - 50 steps, no Turbo')
        plan.update(conditioning_length=5,images=[],videos=[],original=None,original_mask=None)
        plan['summary']={k:v for k,v in plan.items() if k not in ('images','videos','original','original_mask')}
        plan['summary']['refmods']=[{'name':mod.name,'strength':s,'tokens':mod.token_count} for mod,s in mods]
        return plan,plan['width'],plan['height']


class H3LabRefModConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'mods':('H3_REF_MODS',),'plan':('H3_STILL_PLAN',),'clip':('CLIP',),'vae':('VAE',),
            'prompt':('STRING',{'default':'Create one natural studio photograph of the person. Use <Picture 1> for facial identity and <Picture 2> for physique.','multiline':True})}}
    RETURN_TYPES=('CONDITIONING','STRING')
    FUNCTION='encode'
    CATEGORY=CATEGORY
    def encode(self,mods,plan,clip,vae,prompt):
        import re
        for tag in re.findall(r'<Picture\s+(\d+)>',prompt):
            if not 1<=int(tag)<=len(mods):raise ValueError('Prompt references a missing saved photograph.')
        cls=provider_node('refmods','MiniMaxH3RefModTextEncode')
        cuda_sync();start=time.perf_counter()
        conditioning,mapping=cls().encode(clip=clip,mods=mods,prompt=prompt,reference_fps=24.,max_total_tokens=65536,vae=vae)
        cuda_sync()
        report=dict(plan['summary'],conditioning_seconds=time.perf_counter()-start,reference_mapping=mapping,
                    note='RefMods are decoded for the vision encoder. Native equivalence or speedup is not assumed. Do not apply the mods twice.')
        return conditioning,json.dumps(report,indent=2)


PH3_SIGMAS=[1.0,0.984753,0.967905,0.951688,0.925372,0.881923,0.810083,0.528718,0.0]
PH3_FILES=('p-h3-image_int8_convrot.safetensors','p-h3-image_text_encoder_int8_convrot.safetensors','p-h3-image_vae_int8_convrot.safetensors')

class H3LabPH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'prompt':('STRING',{'default':'A natural photograph of a ceramic cup beside a window in soft light.','multiline':True}),
            'aspect_ratio':(['1:1','2:3','3:2','4:5','9:16','16:9'],{'default':'2:3'}),
            'output_megapixels':('FLOAT',{'default':4.,'min':1.,'max':8.,'step':.5}),
            'follow_image_1_aspect':('BOOLEAN',{'default':True})},
            'optional':{'reference_%s'%i:('H3_STILL_REFERENCE',) for i in range(1,9)}}
    RETURN_TYPES=('MODEL','VAE','CONDITIONING','LATENT','SAMPLER','SIGMAS','STRING')
    RETURN_NAMES=('model','vae','conditioning','latent','sampler','sigmas','report')
    FUNCTION='prepare'
    CATEGORY=CATEGORY
    def __init__(self):self.cached=None
    def prepare(self,prompt,aspect_ratio,output_megapixels,follow_image_1_aspect,**refs):
        import nodes,folder_paths,torch,comfy.samplers
        if 'PH3ImageConditioning' not in nodes.NODE_CLASS_MAPPINGS:
            raise RuntimeError('P-H3 needs the ph3 build profile. The stable Lab profile deliberately keeps the original ComfyUI runtime.')
        for folder,name in zip(('diffusion_models','text_encoders','vae'),PH3_FILES):
            if not folder_paths.get_full_path(folder,name):
                raise RuntimeError('Download the INT8 set once: python3 /opt/h3-still/lab/manage.py download ph3')
        images=[refs['reference_%s'%i]['image'] for i in range(1,9)
                if refs.get('reference_%s'%i) and refs['reference_%s'%i].get('image') is not None]
        if not prompt.strip():raise ValueError('Enter a prompt; no prompt enhancer is installed.')
        if images and follow_image_1_aspect:ih,iw=images[0].shape[1:3]
        else:iw,ih=map(int,aspect_ratio.split(':'))
        w,h=dimensions(iw,ih,output_megapixels)
        if self.cached is None:
            model=nodes.UNETLoader().load_unet(PH3_FILES[0],'default')[0]
            clip=nodes.CLIPLoader().load_clip(PH3_FILES[1],'minimax','default')[0]
            vae=nodes.VAELoader().load_vae(PH3_FILES[2])[0]
            self.cached=model,clip,vae
        model,clip,vae=self.cached
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift,PH3ImageConditioning
        model=outputs(MiniMaxH3SigmaShift.execute(model,1.,1.))[0]
        cuda_sync();start=time.perf_counter()
        conditioning,latent=outputs(PH3ImageConditioning.execute(clip,prompt,w,h,vae=vae,
                                      images={'image_%s'%(i+1):im for i,im in enumerate(images)}))
        cuda_sync()
        report={'mode':'P-H3 INT8','width':w,'height':h,'models':PH3_FILES,
                'photos':len(images),'conditioning_seconds':time.perf_counter()-start,
                'steps':8,'sampler':'euler','sigmas':PH3_SIGMAS,
                'prompt_enhancer':False,'core_edit_prefix_note':'Native P-H3 may prefix short image prompts with Edit <Picture 1>; no separate LLM is used.'}
        return model,vae,conditioning,latent,comfy.samplers.sampler_object('euler'),torch.tensor(PH3_SIGMAS),json.dumps(report,indent=2)


EXPERIMENT_NODES={c.__name__:c for c in (H3LabExperiment,H3LabRefModSave,H3LabRefModLoad,
                        H3LabRefModPlan,H3LabRefModConditioning,H3LabPH3)}
