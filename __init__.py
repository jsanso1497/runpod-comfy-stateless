"""H3 Portrait: references + free-form direction -> local Ollama -> native H3.
Installed Python nodes, not executable code embedded in a workflow.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import uuid

from . import logic
from . import ollama_client as oc

WEB_DIRECTORY='./web'
_PROMPT_CACHE={}


def input_paths(value):
    import folder_paths
    try: names=json.loads(value)
    except (TypeError,ValueError) as e:raise ValueError('Use Upload reference images to select your photos.') from e
    if not isinstance(names,list) or not 1<=len(names)<=logic.MAX_REFERENCES:
        raise ValueError('Upload 1 through 9 reference images. Native H3 accepts at most 9; none will be silently discarded.')
    root=Path(folder_paths.get_input_directory()).resolve();paths=[]
    for name in names:
        if not isinstance(name,str) or '\\' in name or '\x00' in name:
            raise ValueError('Invalid reference filename.')
        p=(root/name).resolve()
        if not p.is_relative_to(root) or p==root or not p.is_file():
            raise ValueError(f'Reference is missing from ComfyUI input: {Path(name).name}')
        if p.suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):
            raise ValueError('References must be PNG, JPEG or WebP still images.')
        paths.append((name,p))
    return paths


class H3PortraitReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'reference_files':('STRING',{'default':'[]','multiline':False})}}
    RETURN_TYPES=('H3_PORTRAIT_REFS',)
    RETURN_NAMES=('references',)
    FUNCTION='load'
    CATEGORY='H3 Portrait'
    DESCRIPTION='Upload 1-9 images. The numbered order matches image 1, image 2, etc. in your normal-language brief.'

    @classmethod
    def IS_CHANGED(cls,reference_files):
        try:return tuple((n,p.stat().st_mtime_ns,p.stat().st_size) for n,p in input_paths(reference_files))
        except Exception:return float('nan')

    def load(self,reference_files):
        import numpy as np
        from PIL import Image,ImageOps
        import torch
        refs=[]
        for n,p in input_paths(reference_files):
            with Image.open(p) as im:
                if im.width*im.height>50_000_000:
                    raise ValueError(f'{p.name} exceeds 50 megapixels. Resize that source before upload.')
                im=ImageOps.exif_transpose(im)
                if 'A' in im.getbands():
                    rgba=im.convert('RGBA');rgb=Image.new('RGB',rgba.size,(255,255,255));rgb.paste(rgba,mask=rgba.getchannel('A'));im=rgb
                else:im=im.convert('RGB')
                array=np.asarray(im,dtype=np.float32)/255.
                image=torch.from_numpy(array.copy())[None]
            sha=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):sha.update(block)
            refs.append({'filename':n,'sha256':sha.hexdigest(),'image':image})
        return (refs,)


class H3PortraitDirector:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        lora_names=['(none)']+folder_paths.get_filename_list('loras')
        return {'required':{
            'references':('H3_PORTRAIT_REFS',),
            'instruction':('STRING',{'multiline':True,'default':'These images show the same person. Use the clearest headshot for the face and the full-body photo for the outfit. Show the person walking slowly through a modern hotel lobby, then looking toward the camera. Preserve their identity and proportions. One continuous portrait shot. No extra people, dialogue, music or on-screen text.'}),
            'aspect':(list(logic.ASPECTS),{'default':'9:16'}),
            'quality':(list(logic.PRESETS),{'default':'Standard'}),
            'seconds':('INT',{'default':5,'min':3,'max':15,'step':1}),
            'seed':('INT',{'default':42,'min':0,'max':2**32-1}),
            'use_loras':('BOOLEAN',{'default':True}),
            'mode':(['Generate video','Draft only'],{'default':'Generate video'}),
            'prompt_variation':('INT',{'default':0,'min':0,'max':2**31-1})},
            'optional':{
            'lora_1':(lora_names,{'default':'(none)'}),
            'strength_1':('FLOAT',{'default':1.0,'min':-2.0,'max':2.0,'step':0.05}),
            'lora_2':(lora_names,{'default':'(none)'}),
            'strength_2':('FLOAT',{'default':1.0,'min':-2.0,'max':2.0,'step':0.05}),
            'extra_trigger_words':('STRING',{'default':'','multiline':False})}}
    RETURN_TYPES=('H3_PORTRAIT_JOB','STRING')
    RETURN_NAMES=('ready_job','prompt_and_reference_map')
    FUNCTION='direct'
    CATEGORY='H3 Portrait'
    DESCRIPTION='Write naturally. Thinking pass 1 analyzes references; pass 2 writes the MiniMax guide prompt. H3 starts only after Ollama unloads.'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Custom prompt files are read on each execution, not frozen at import.
        # ComfyUI still tracks the node's ordinary input changes itself.
        try:
            value = {'settings': logic.settings(), 'policy': logic.prompt_policy()}
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        except (ValueError, OSError):
            return float('nan')

    def direct(self,references,instruction,aspect,quality,seconds,seed,use_loras,mode,prompt_variation,
               lora_1='(none)',strength_1=1.0,lora_2='(none)',strength_2=1.0,extra_trigger_words=''):
        import folder_paths
        import comfy.model_management as mm
        from comfy_execution.graph_utils import ExecutionBlocker
        if mode not in ('Generate video','Draft only'):
            raise ValueError('Unsupported run mode.')
        if not isinstance(instruction,str) or not instruction.strip():
            raise ValueError('Describe the shot in the instruction field. Any ordinary wording is accepted.')
        if not 1<=len(references)<=9:raise ValueError('Supply 1-9 reference images.')
        recipe=logic.geometry(aspect,quality,seconds);cfg=logic.settings()
        # Resolve the intended LoRA before paying for prompt generation.
        loras=logic.active_loras(use_loras,[(lora_1,strength_1),(lora_2,strength_2)],Path(folder_paths.models_dir)/'loras')
        for lora in loras:
            if not folder_paths.get_full_path('loras',lora['lora_name']):
                raise ValueError('The configured LoRA has not downloaded. Read the startup download log.')
        triggers='; '.join(x.get('trigger_words','') for x in loras if x.get('trigger_words','').strip())
        if extra_trigger_words.strip() and loras:
            triggers='; '.join(x for x in (triggers,extra_trigger_words.strip()) if x)
        print('H3 PORTRAIT: '+str(len(loras))+' selected LoRA(s). Downloaded library entries are not auto-applied.',flush=True)
        with oc.session() as session:
            info=oc.model_info(session,cfg['ollama_model'])
            key=logic.cache_key([r['filename']+'|'+r['sha256'] for r in references],instruction,recipe,info['digest'],cfg,triggers,prompt_variation)
            if key in _PROMPT_CACHE:
                obj=_PROMPT_CACHE[key]
                print('H3 PORTRAIT: using the same cached prompt. Change prompt_variation for another draft.',flush=True)
                # No fresh Ollama load. Ensure a previous failed/external request did not leave it resident.
                if oc.api(session,'/api/ps').get('models'):oc.unload(session,info['model'],cfg['unload_timeout_seconds'])
            else:
                mm.unload_all_models();mm.soft_empty_cache()
                try:
                    obj=oc.generate(session,cfg,info,logic.prompt_policy()['system_prompt.txt'],instruction,references,recipe,triggers,prompt_variation,mm.throw_exception_if_processing_interrupted)
                except logic.ClarificationNeeded as e:
                    raise ValueError('Clarify this in your instruction, then run again: '+str(e)) from e
                if len(_PROMPT_CACHE)>=32:_PROMPT_CACHE.pop(next(iter(_PROMPT_CACHE)))
                _PROMPT_CACHE[key]=obj
        prompt=logic.format_prompt(obj,instruction,recipe,triggers)
        job={'recipe':recipe,'seed':int(seed),'references':references,'prompt':prompt,'loras':loras,'files':cfg['model_files']}
        report={'version':'h3-portrait-1.2.0','ollama':info,'recipe':recipe,'seed':int(seed),
                'user_direction':instruction,'mapping':obj['references'],'prompt':prompt,
                'reference_analysis':obj.get('_analysis'), 'prompt_stages':obj.get('_stages',[]),
                'prompt_policy_sha256':hashlib.sha256(json.dumps(logic.prompt_policy(),sort_keys=True).encode()).hexdigest(),
                'prompt_settings':{k:cfg.get(k) for k in ('think','temperature','top_p','top_k','context_length','image_max_edge','analysis_max_output_tokens','max_output_tokens')},
                'sources':[{'file':r['filename'],'sha256':r['sha256']} for r in references],
                'loras':loras}
        root=Path(folder_paths.get_output_directory())/'H3_Portrait'/'prompts';root.mkdir(parents=True,exist_ok=True)
        name=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        (root/(name+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        (root/(name+'.txt')).write_text(prompt+'\n')
        print(f'H3 PORTRAIT PROMPT READY: {len(references)} images, {recipe["aspect"]}, {recipe["quality"]}, {recipe["length"]} frames, {recipe["steps"]} steps, {len(loras)} LoRA(s).',flush=True)
        if mode=='Draft only':return (ExecutionBlocker(None),prompt)
        return (job,prompt)


class _LoraWarnings(logging.Handler):
    def __init__(self):super().__init__(logging.WARNING);self.messages=[]
    def emit(self,record):
        msg=record.getMessage()
        if any(x in msg.lower() for x in ('lora key not loaded','not loaded','shape mismatch','weight not merged','calculate weight failed')):
            self.messages.append(msg)


def load_loras(model,entries):
    import folder_paths
    import comfy.utils
    import comfy.lora
    import comfy.lora_convert
    for entry in entries:
        p=folder_paths.get_full_path_or_raise('loras',entry['lora_name'])
        weights=comfy.utils.load_torch_file(p,safe_load=True)
        weights=comfy.lora_convert.convert_lora(weights)
        keys=comfy.lora.model_lora_keys_unet(model.model,{})
        guard=_LoraWarnings();logging.getLogger().addHandler(guard)
        try:patches=comfy.lora.load_lora(weights,keys,log_missing=True)
        finally:logging.getLogger().removeHandler(guard)
        if not patches or guard.messages:
            raise ValueError(f'LoRA {entry["name"]} does not map completely to the full H3 Ref2VA model. '+('; '.join(guard.messages[:3]) or 'No matching model weights.'))
        updated=model.clone();applied=updated.add_patches(patches,float(entry['strength']))
        if set(applied)!=set(patches):
            raise ValueError(f'LoRA {entry["name"]}: some patches were rejected. Confirm the full Ref2VA target and ComfyUI format.')
        print(f'H3 PORTRAIT LORA: {entry["lora_name"]}; strength={entry["strength"]}; {len(applied)} patch targets.',flush=True)
        model=updated
    return model


class H3PortraitModels:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'ready_job':('H3_PORTRAIT_JOB',)}}
    RETURN_TYPES=('MODEL','CLIP','VAE','VAE')
    RETURN_NAMES=('model_with_loras','h3_encoder','video_vae','audio_vae')
    FUNCTION='load'
    CATEGORY='H3 Portrait/backend'
    def load(self,ready_job):
        import nodes
        files=ready_job['files']
        signature=(tuple(sorted(files.items())),tuple((x['lora_name'],x.get('sha256',''),float(x['strength'])) for x in ready_job['loras']))
        if getattr(self,'_signature',None)==signature and getattr(self,'_loaded',None) is not None:
            return self._loaded
        # Reuse unchanged model objects across seed/prompt changes. ComfyUI still
        # owns GPU loading/offloading; this is not a keep-everything-in-VRAM mode.
        self._loaded=None
        model=nodes.UNETLoader().load_unet(files['diffusion'],'default')[0]
        model=load_loras(model,ready_job['loras'])
        clip=nodes.CLIPLoader().load_clip(files['encoder'],'minimax','default')[0]
        vae=nodes.VAELoader().load_vae(files['video_vae'])[0]
        audio=nodes.VAELoader().load_vae(files['audio_vae'])[0]
        self._loaded=(model,clip,vae,audio)
        self._signature=signature
        return self._loaded


class H3PortraitConditioning:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'ready_job':('H3_PORTRAIT_JOB',),'clip':('CLIP',),'vae':('VAE',),'audio_vae':('VAE',)}}
    RETURN_TYPES=('CONDITIONING','LATENT')
    FUNCTION='encode'
    CATEGORY='H3 Portrait/backend'
    def encode(self,ready_job,clip,vae,audio_vae):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        job=ready_job;r=job['recipe']
        out=MiniMaxH3ReferenceToVideo.execute(clip=clip,vae=vae,audio_vae=audio_vae,
              prompt=job['prompt'],width=r['width'],height=r['height'],length=r['length'],
              ref_image_size=r['ref_image_size'],
              ref_images={f'ref_image_{i}':ref['image'] for i,ref in enumerate(job['references'])})
        return out[0],out[1]


class H3PortraitSampler:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'ready_job':('H3_PORTRAIT_JOB',),'model':('MODEL',),'positive':('CONDITIONING',),'latent':('LATENT',)}}
    RETURN_TYPES=('LATENT',)
    FUNCTION='sample'
    CATEGORY='H3 Portrait/backend'
    def sample(self,ready_job,model,positive,latent):
        from comfy_extras import nodes_custom_sampler as ns
        r=ready_job['recipe']
        noise=ns.RandomNoise.execute(ready_job['seed'])[0]
        guider=ns.BasicGuider.execute(model,positive)[0]
        sampler=ns.KSamplerSelect.execute(r['sampler'])[0]
        sigmas=ns.BasicScheduler.execute(model,r['scheduler'],r['steps'],1.0)[0]
        guard=_LoraWarnings();logging.getLogger().addHandler(guard)
        try:output=ns.SamplerCustomAdvanced.execute(noise,guider,sampler,sigmas,latent)[0]
        finally:logging.getLogger().removeHandler(guard)
        if guard.messages:
            raise ValueError('Model reported rejected LoRA weights during sampling. No video exported: '+'; '.join(guard.messages[:3]))
        return (output,)


class H3PortraitExactAspect:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'images':('IMAGE',),'ready_job':('H3_PORTRAIT_JOB',)}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='crop'
    CATEGORY='H3 Portrait/backend'
    def crop(self,images,ready_job):
        r=ready_job['recipe']
        if images.ndim==5 and images.shape[0]==1:images=images[0]
        if images.ndim!=4:raise ValueError('Expected a video frame tensor [frames,height,width,channels].')
        h,w=images.shape[1:3]
        if (w,h)!=(r['width'],r['height']):raise ValueError('Decoded video does not match the requested generation canvas.')
        ow,oh=r['output_width'],r['output_height']
        x=(w-ow)//2;y=(h-oh)//2
        return (images[:,y:y+oh,x:x+ow,:],)


NODE_CLASS_MAPPINGS={c.__name__:c for c in (H3PortraitReferences,H3PortraitDirector,H3PortraitModels,H3PortraitConditioning,H3PortraitSampler,H3PortraitExactAspect)}
NODE_DISPLAY_NAME_MAPPINGS={'H3PortraitReferences':'1. Upload references',
                          'H3PortraitDirector':'2. Describe the video',
                          'H3PortraitModels':'Load H3 + selected LoRA',
                          'H3PortraitConditioning':'Native H3 Ref2VA references',
                          'H3PortraitSampler':'H3 sampler',
                          'H3PortraitExactAspect':'Exact portrait aspect (no stretching)'}
