"""Orchestration around upstream ComfyUI/Fizgig; no modified diffusion implementation."""
from __future__ import annotations
import gc
import hashlib
import json
import logging
from pathlib import Path
import time
from .rules import (VERSION, ENGINES, PRECISIONS, MODES, ASPECTS, NONE, dimensions,
                    use_frames, resolve_mode, reference_budget, translate_prompt, recipe, model_names)
from .media import EXTENSIONS, checked_path, extract

CATEGORY = 'H3 Video Still Studio'


def result(value):
    if hasattr(value, 'result'):
        value = value.result
    return value['result'] if isinstance(value, dict) else value


def node_call(name, **kwargs):
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError('Required node is unavailable: '+name+'. Use the new H3 Video Still Studio image.')
    if hasattr(cls, 'execute'):
        return result(cls.execute(**kwargs))
    return result(getattr(cls(), cls.FUNCTION)(**kwargs))


def resize(image, width, height):
    if list(image.shape[1:3]) == [height, width]:
        return image
    import comfy.utils
    return comfy.utils.common_upscale(image.movedim(-1,1), width, height, 'lanczos', 'disabled').movedim(1,-1)


def contact_sheet(images, labels, cell=384):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    if not images:
        return torch.zeros(1,32,32,3)
    columns = min(3,len(images)); rows = (len(images)+columns-1)//columns
    canvas = Image.new('RGB',(columns*cell,rows*(cell+32)),(230,230,230))
    draw = ImageDraw.Draw(canvas)
    for index,(image,label) in enumerate(zip(images,labels)):
        array=(image[0].detach().cpu().clamp(0,1).numpy()*255).round().astype('uint8')
        pic=Image.fromarray(array[...,:3]); pic=ImageOps.contain(pic,(cell,cell))
        x=(index%columns)*cell; y=(index//columns)*(cell+32)
        canvas.paste(pic,(x+(cell-pic.width)//2,y+(cell-pic.height)//2))
        draw.text((x+8,y+cell+6),label[:58],fill=(20,20,20))
    return torch.from_numpy(np.asarray(canvas).copy()).float().div_(255).unsqueeze(0)


class H3VSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'engine': (ENGINES,), 'precision': (PRECISIONS,), 'mode': (MODES,),
            'aspect_ratio': (ASPECTS, {'default':'2:3'}),
            'output_megapixels': ('FLOAT',{'default':2.5,'min':.25,'max':8.,'step':.25}),
            'photo_cap_megapixels': ('FLOAT',{'default':4.,'min':.25,'max':8.,'step':.25}),
            'reference_size': (['max','match'],),
            'video_strategy': (['Auto','Keyframes for all engines'],),
            'standard_steps': ('INT',{'default':50,'min':20,'max':100}),
            'eros_steps': ('INT',{'default':8,'min':6,'max':12}),
            'protect_background': ('BOOLEAN',{'default':False})}}
    RETURN_TYPES=('H3V_CONFIG',)
    FUNCTION='configure'; CATEGORY=CATEGORY
    DESCRIPTION='One model per run. Auto video: native H3/Eros, original-file keyframes for image-only P-H3. Output stays one PNG.'
    def configure(self, **settings):
        recipe(settings['engine'],settings['standard_steps'],settings['eros_steps'])
        model_names(settings['engine'],settings['precision'],'Generate')
        return (dict(settings,version=VERSION),)


class H3VVideo:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        root=Path(folder_paths.get_input_directory())
        names=sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                     if p.is_file() and p.suffix.lower() in EXTENSIONS)
        return {'required': {'config':('H3V_CONFIG',), 'file':([NONE]+names,),
            'enabled':('BOOLEAN',{'default':True}),
            'start_seconds':('FLOAT',{'default':0.,'min':0.,'max':86400.,'step':.1}),
            'duration_seconds':('FLOAT',{'default':3.1,'min':.25,'max':15.,'step':.1}),
            'native_video_megapixels':('FLOAT',{'default':1.,'min':.1,'max':1.,'step':.1}),
            'keyframe_megapixels':('FLOAT',{'default':2.5,'min':.25,'max':8.,'step':.25}),
            'keyframe_count':('INT',{'default':4,'min':1,'max':8}),
            'keyframe_times':('STRING',{'default':'','tooltip':'Optional seconds relative to start, e.g. 0.2, 1.0, 2.4. Overrides count for keyframe routes only.'}),
            'crop_left_percent':('FLOAT',{'default':0.,'min':0.,'max':90.,'step':1.}),
            'crop_top_percent':('FLOAT',{'default':0.,'min':0.,'max':90.,'step':1.}),
            'crop_right_percent':('FLOAT',{'default':0.,'min':0.,'max':90.,'step':1.}),
            'crop_bottom_percent':('FLOAT',{'default':0.,'min':0.,'max':90.,'step':1.})}}
    RETURN_TYPES=('H3V_VIDEO',)
    FUNCTION='load'; CATEGORY=CATEGORY
    @classmethod
    def VALIDATE_INPUTS(cls,file,enabled=True,**kwargs):
        if not enabled or file==NONE:return True
        import folder_paths
        try:checked_path(file,folder_paths.get_input_directory());return True
        except (ValueError,OSError) as error:return str(error)
    @classmethod
    def IS_CHANGED(cls,file,enabled=True,**kwargs):
        if not enabled or file==NONE:return 'empty'
        import folder_paths
        path=checked_path(file,folder_paths.get_input_directory());stat=path.stat()
        return '%s:%s:%s'%(path,stat.st_size,stat.st_mtime_ns)
    def load(self,config,file,enabled,start_seconds,duration_seconds,native_video_megapixels,
             keyframe_megapixels,keyframe_count,keyframe_times,crop_left_percent,crop_top_percent,
             crop_right_percent,crop_bottom_percent):
        if not enabled or file==NONE:return ({'frames':None,'keyframes':[],'summary':None},)
        import folder_paths
        path=checked_path(file,folder_paths.get_input_directory())
        prepared=extract(path,start_seconds,duration_seconds,
            use_frames(config['engine'],config['video_strategy']),native_video_megapixels,
            keyframe_megapixels,keyframe_count,keyframe_times,
            (crop_left_percent,crop_top_percent,crop_right_percent,crop_bottom_percent))
        return (prepared,)


class H3VPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'config':('H3V_CONFIG',),'prompt':('STRING',{'multiline':True})},
                'optional':dict({'reference_%d'%i:('H3_STILL_REFERENCE',) for i in range(1,10)},video=('H3V_VIDEO',))}
    RETURN_TYPES=('H3V_PLAN','IMAGE','STRING')
    RETURN_NAMES=('plan','actual_references','reference_report')
    FUNCTION='prepare'; CATEGORY=CATEGORY
    def prepare(self,config,prompt,video=None,**refs):
        selected=[(i,refs['reference_%d'%i]) for i in range(1,10)
                  if refs.get('reference_%d'%i) and refs['reference_%d'%i].get('image') is not None]
        video=video or {}; frames=video.get('frames'); keyframes=video.get('keyframes',[])
        has_video=frames is not None or bool(keyframes)
        mode=resolve_mode(config['mode'],[i for i,_ in selected],has_video)
        reference_budget(config['engine'],len(selected),len(keyframes))
        if config['engine']=='P-H3 Image' and frames is not None:
            raise ValueError('P-H3 cannot condition on native video. Reconnect the video node to the same settings node.')
        original=selected[0][1]['image'] if selected else None
        original_mask=selected[0][1].get('mask') if selected else None
        if mode in ('Edit','Replace person'):
            ih,iw=original.shape[1:3]
        else:
            iw,ih=map(int,config['aspect_ratio'].split(':'))
        width,height=dimensions(iw,ih,float(config['output_megapixels']))
        if config['protect_background']:
            if mode!='Replace person':raise ValueError('Background protection is for Replace person only.')
            import torch
            if original_mask is None or tuple(original_mask.shape)!=(1,*original.shape[1:3]) or not torch.isfinite(original_mask).all() or original_mask.max()<=0:
                raise ValueError('Use the replacement preset and paint a mask on the target Load Image. White replaces; black preserves. Or disable protect_background.')
        images=[];mapping=[]
        for number,(slot,ref) in enumerate(selected,1):
            image=ref['image']
            if image.shape[0]!=1:raise ValueError('Use one still photo per reference slot.')
            w,h=dimensions(image.shape[2],image.shape[1],float(config['photo_cap_megapixels']),down_only=True)
            images.append(resize(image[...,:3],w,h))
            mapping.append(dict(tag='<Picture %d>'%number,slot=slot,file=ref.get('name','photo'),size=[w,h]))
        for index,image in enumerate(keyframes):
            images.append(image)
            mapping.append(dict(tag='<Picture %d>'%len(images),source='video keyframe',
                time_seconds=video['summary']['timestamps_seconds'][index],size=[image.shape[2],image.shape[1]]))
        literal=translate_prompt(prompt,len(selected),len(keyframes),frames is not None,config['engine']=='P-H3 Image')
        sampling=recipe(config['engine'],config['standard_steps'],config['eros_steps'])
        names=model_names(config['engine'],config['precision'],mode)
        summary=dict(config,mode=mode,width=width,height=height,models=list(names),sampling=sampling,
            reference_mapping=mapping,video=video.get('summary'),user_prompt=prompt,effective_prompt=literal,
            prompt_enhancer=False,output='one PNG',
            output_size=[original.shape[2],original.shape[1]] if config['protect_background'] else [width,height],
            mask_usage='final composite only, not diffusion inpainting' if config['protect_background'] else 'none',
            validation='Source-level routing only. This plan is not proof of GPU inference or likeness quality.')
        plan=dict(config,mode=mode,width=width,height=height,models=names,recipe=sampling,images=images,
            videos=[] if frames is None else [frames],conditioning_length=int(frames.shape[0]) if frames is not None else 5,
            prompt=literal,original=original,original_mask=original_mask,summary=summary)
        previews=list(images);labels=[m['tag']+(' video %.2fs'%m['time_seconds'] if 'time_seconds'in m else ' photo slot %s'%m['slot']) for m in mapping]
        if frames is not None:
            for index in sorted(set([0,len(frames)//3,2*len(frames)//3,len(frames)-1])):
                previews.append(frames[index:index+1]);labels.append('Video 1 preview frame %d'%index)
        return plan,contact_sheet(previews,labels),json.dumps(summary,indent=2)


def finish_image(plan, image):
    """Final composite only. Zero-mask pixels preserve the original tensor exactly."""
    if not plan['protect_background']:
        return image
    import torch
    original=plan['original'].to(image)
    mask=plan['original_mask'].to(image).clamp(0,1).unsqueeze(-1)
    candidate=resize(image,original.shape[2],original.shape[1])
    return torch.where(mask==0,original,candidate*mask+original*(1-mask))


class LogCapture(logging.Handler):
    def __init__(self):super().__init__(logging.WARNING);self.lines=[]
    def emit(self,record):self.lines=(self.lines+[record.getMessage()[:2000]])[-100:]


class H3VRender:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        names=[NONE]+[n for n in folder_paths.get_filename_list('loras') if n.lower().endswith('.safetensors')]
        required={'plan':('H3V_PLAN',),'seed':('INT',{'default':18071042,'min':0,'max':0xffffffffffffffff,'control_after_generate':True})}
        for i in (1,2):
            required.update({f'lora_{i}_enabled':('BOOLEAN',{'default':False}),f'lora_{i}_name':(names,),
                f'lora_{i}_strength':('FLOAT',{'default':.6,'min':-2.,'max':2.,'step':.05}),
                f'lora_{i}_family':(['H3 video / Eros','P-H3 image'],)})
        return {'required':required}
    RETURN_TYPES=('IMAGE','IMAGE','STRING')
    RETURN_NAMES=('final_image','reference_left_result_right','run_report')
    FUNCTION='render'; CATEGORY=CATEGORY
    DESCRIPTION='Only the selected family is loaded. Two optional model-only LoRAs. Family-specific sampling and decoding; no accelerator or enhancer.'
    def __init__(self):self.cache_key=None;self.cached=None
    def load_models(self,names):
        import folder_paths
        import nodes
        import comfy.model_management as mm
        keys=[]
        for folder,name in zip(('diffusion_models','text_encoders','vae'),names):
            path=folder_paths.get_full_path(folder,name)
            if not path:
                raise FileNotFoundError('Missing %s. For full precision, run in Jupyter Terminal: python3 /opt/h3-still/video_studio/manage.py download bf16 . Then refresh ComfyUI.'%name)
            stat=Path(path).stat();keys.append((path,stat.st_size,stat.st_mtime_ns))
        key=tuple(keys)
        if key!=self.cache_key:
            # Keep one selected base stack, not one stack per dropdown option.
            self.cached=None;self.cache_key=None
            mm.unload_all_models();gc.collect();mm.soft_empty_cache()
            model=nodes.UNETLoader().load_unet(names[0],'default')[0]
            clip=nodes.CLIPLoader().load_clip(names[1],'minimax','default')[0]
            vae=nodes.VAELoader().load_vae(names[2])[0]
            self.cached=model,clip,vae;self.cache_key=key
        return self.cached
    def render(self,plan,seed,**loras):
        import torch
        import nodes
        import comfy.samplers
        import comfy.model_management as mm
        from comfy_extras.nodes_minimax_h3 import (MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo,
            MiniMaxH3SigmaShift, PH3ImageConditioning)
        from comfy_extras.nodes_custom_sampler import RandomNoise, BasicGuider, BasicScheduler, SamplerCustomAdvanced
        capture=LogCapture();logging.getLogger().addHandler(capture)
        began=time.perf_counter();lora_report=[]
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        try:
            model,clip,vae=self.load_models(plan['models'])
            for i in (1,2):
                if not loras.get(f'lora_{i}_enabled') or loras.get(f'lora_{i}_name')==NONE or loras.get(f'lora_{i}_strength',0)==0:continue
                name=loras[f'lora_{i}_name'];strength=loras[f'lora_{i}_strength'];family=loras[f'lora_{i}_family']
                if 'turbo' in name.lower() or 'p-h3-image_lora' in name.lower():
                    raise ValueError('These slots are for identity/style LoRAs, not Turbo or base-conversion adapters. Eros and P-H3 already contain their acceleration recipe.')
                expected='P-H3 image' if plan['engine']=='P-H3 Image' else 'H3 video / Eros'
                if family!=expected:raise ValueError('LoRA %d is labeled for %s, not %s. Use a compatible adapter and select its actual family.'%(i,family,expected))
                import folder_paths
                import comfy.lora
                import comfy.sd
                import comfy.utils
                path=folder_paths.get_full_path_or_raise('loras',name)
                state=comfy.utils.load_torch_file(path,safe_load=True)
                key_map=comfy.lora.model_lora_keys_unet(model.model,{})
                matching=comfy.lora.load_lora(state,key_map)
                if not matching:raise ValueError('No diffusion keys matched '+name+'. This is not a usable LoRA for the selected family.')
                # Native bypass adds the low-rank forward path without rewriting quantized base weights.
                model=comfy.sd.load_bypass_lora_for_models(model,None,state,strength,0.)[0]
                lora_report.append(dict(file=name,strength=strength,family=family,matched_keys=len(matching),
                    loader='Native ComfyUI model-only bypass',
                    warning='Matched keys are not proof of training-family or visual compatibility. Inspect warnings and compare with LoRA off.'))
                del state,matching,key_map
            ready=time.perf_counter()
            if plan['engine']=='P-H3 Image':
                cond,latent=result(PH3ImageConditioning.execute(clip=clip,prompt=plan['prompt'],width=plan['width'],height=plan['height'],
                    vae=vae,images={'image_%d'%(i+1):im for i,im in enumerate(plan['images'])}))
                decoder='Native P-H3 image VAE'
            else:
                common=dict(clip=clip,vae=vae,prompt=plan['prompt'],width=plan['width'],height=plan['height'],length=plan['conditioning_length'])
                if plan['mode']=='Generate':cond=result(MiniMaxH3ImageToVideo.execute(**common))[0]
                else:cond=result(MiniMaxH3ReferenceToVideo.execute(**common,ref_image_size=plan['reference_size'],
                    ref_images={'ref_image_%d'%i:im for i,im in enumerate(plan['images'])},
                    ref_videos={'ref_video_%d'%i:frames for i,frames in enumerate(plan['videos'])}))[0]
                latent=node_call('FizgigH3StillLatent',width=plan['width'],height=plan['height'],batch_size=1)[0]
                decoder='Fizgig H3 single-frame video VAE'
            if torch.cuda.is_available():torch.cuda.synchronize()
            conditioned=time.perf_counter();sampling=plan['recipe']
            if sampling['shift'] is not None:model=result(MiniMaxH3SigmaShift.execute(model,*sampling['shift']))[0]
            sigmas=(torch.tensor(sampling['sigmas'],dtype=torch.float32) if sampling['sigmas'] is not None else
                    result(BasicScheduler.execute(model,sampling['scheduler'],sampling['steps'],1.))[0])
            noise=result(RandomNoise.execute(seed))[0]
            guider=result(BasicGuider.execute(model,cond))[0]
            sampler=comfy.samplers.sampler_object(sampling['sampler'])
            samples=result(SamplerCustomAdvanced.execute(noise,guider,sampler,sigmas,latent))[0]
            if torch.cuda.is_available():torch.cuda.synchronize()
            sampled=time.perf_counter()
            if plan['engine']=='P-H3 Image':image=nodes.VAEDecode().decode(vae,samples)[0]
            else:image=node_call('FizgigH3StillDecode',samples=samples,vae=vae)[0]
            image=image[:1,...,:3]
            image=finish_image(plan,image)
            if torch.cuda.is_available():torch.cuda.synchronize()
            completed=time.perf_counter()
            original=plan['original']
            if original is None and plan['images']:original=plan['images'][0]
            if original is None and plan['videos']:original=plan['videos'][0][:1]
            comparison=contact_sheet([original,image],['Reference / original','Generated result'],512) if original is not None else image
            report=dict(plan['summary'],seed=seed,loras=lora_report,decoder=decoder,warnings=capture.lines,
                timings_seconds=dict(load_and_loras=ready-began,conditioning=conditioned-ready,sampling=sampled-conditioned,
                                     decode_and_finish=completed-sampled,total=completed-began),
                output_actual_size=[image.shape[2],image.shape[1]],
                prompt_sha256=hashlib.sha256(plan['prompt'].encode()).hexdigest(),
                precision_note='Precision refers to stored components; activations and memory offloading are managed by native ComfyUI.',
                validation='One completed local generation, not release-wide GPU qualification.')
            if torch.cuda.is_available():
                report['gpu']=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                    total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            return image,comparison,json.dumps(report,indent=2)
        except torch.OutOfMemoryError as error:
            self.cached=None;self.cache_key=None;mm.unload_all_models();gc.collect();mm.soft_empty_cache()
            raise RuntimeError('GPU memory exhausted. Reduce output MP, reference size/count or video duration; or switch precision to INT8 HQ. No lower-quality fallback was applied silently.') from error
        finally:logging.getLogger().removeHandler(capture)


class H3VAudit:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'report':('STRING',{'multiline':True}),'save_report':('BOOLEAN',{'default':True})}}
    RETURN_TYPES=();OUTPUT_NODE=True;FUNCTION='show';CATEGORY=CATEGORY
    def show(self,report,save_report):
        if save_report:
            import folder_paths
            import uuid
            folder=Path(folder_paths.get_output_directory())/'H3_Video_Studio'/'reports';folder.mkdir(parents=True,exist_ok=True)
            path=folder/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.json')
            with path.open('x',encoding='utf-8') as stream:stream.write(report+'\n')
        return {'ui':{'text':[report]},'result':()}


NODE_CLASS_MAPPINGS={cls.__name__:cls for cls in (H3VSettings,H3VVideo,H3VPlan,H3VRender,H3VAudit)}
