"""Reference controls, literal prompt assembly, reports and timing around upstream nodes."""
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import time
from .common import (CATEGORY, VERSION, NONE, ROOT, registry, invoke, outputs, crop_bounds,
                     dimensions, resize, cuda_sync, text_ui, run_filename, short_name)


class H3LabReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'reference': ('H3_STILL_REFERENCE',),
            'role': ('STRING', {'default': 'supporting reference'}),
            'cap_megapixels': ('FLOAT', {'default': 2.5, 'min': .25, 'max': 8., 'step': .25}),
            'crop_left_percent': ('FLOAT', {'default': 0., 'min': 0., 'max': 99., 'step': 1.}),
            'crop_top_percent': ('FLOAT', {'default': 0., 'min': 0., 'max': 99., 'step': 1.}),
            'crop_right_percent': ('FLOAT', {'default': 0., 'min': 0., 'max': 99., 'step': 1.}),
            'crop_bottom_percent': ('FLOAT', {'default': 0., 'min': 0., 'max': 99., 'step': 1.})}}
    RETURN_TYPES = ('H3_STILL_REFERENCE', 'IMAGE', 'STRING')
    RETURN_NAMES = ('prepared_reference', 'preview', 'details')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY

    def prepare(self, reference, role, cap_megapixels, crop_left_percent, crop_top_percent,
                crop_right_percent, crop_bottom_percent):
        import torch
        start = time.perf_counter()
        source = reference.get('image')
        if source is None:
            # Preview placeholder is NOT passed to H3. The reference remains genuinely absent.
            return reference, torch.zeros(1, 32, 32, 3), 'Inactive reference; preview placeholder is not used as conditioning.'
        if source.shape[0] != 1:
            raise ValueError('Use one still photograph per reference input.')
        h, w = source.shape[1:3]
        x0,y0,x1,y1 = crop_bounds(w,h,crop_left_percent,crop_top_percent,crop_right_percent,crop_bottom_percent)
        cropped = source[:, y0:y1, x0:x1, :3]
        nw,nh = dimensions(x1-x0,y1-y0,cap_megapixels,True)
        prepared = resize(cropped,nw,nh)
        out = dict(reference, image=prepared, lab_role=role)
        out.pop('mask', None)  # Supporting references do not inherit a target-compositing mask.
        out['lab_preprocess'] = {'role': role, 'source_size':[w,h], 'crop':[x0,y0,x1,y1],
                                 'prepared_size':[nw,nh], 'seconds':time.perf_counter()-start}
        return out, prepared, json.dumps(out['lab_preprocess'], indent=2)


class H3LabPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'mode': (['Auto','Generate','Edit','References','Replace person'], {'default':'Auto'}),
            'aspect_ratio': (['1:1','16:9','9:16','2:3','3:2','4:5','5:4','4:3','3:4'], {'default':'2:3'}),
            'output_megapixels': ('FLOAT', {'default':2.5,'min':.25,'max':8.,'step':.25}),
            'preset': (['Quality - 50 steps, no Turbo','REF2VA Turbo - 8 steps','Developer FL2VA fast - 20 steps'],),
            'manual_quality_steps': ('INT', {'default':0,'min':0,'max':100,'tooltip':'0 keeps the baseline 50. Only affects Quality; 8-step Turbo is never silently changed.'}),
            'reference_size_mode': (['match','max'], {'default':'match'}),
            'protect_background': ('BOOLEAN', {'default':False})},
            'optional': dict({'reference_%s'%i:('H3_STILL_REFERENCE',) for i in range(1,10)}, video=('H3_STILL_VIDEO',))}
    RETURN_TYPES = ('H3_STILL_PLAN','INT','INT')
    RETURN_NAMES = ('plan','width','height')
    FUNCTION = 'prepare'
    CATEGORY = CATEGORY

    def prepare(self, mode, aspect_ratio, output_megapixels, preset, manual_quality_steps,
                reference_size_mode, protect_background, **refs):
        if manual_quality_steps and preset != 'Quality - 50 steps, no Turbo':
            raise ValueError('Manual steps are only allowed with Quality. Set manual_quality_steps to 0 for Turbo.')
        if manual_quality_steps and manual_quality_steps < 4:
            raise ValueError('Use at least four steps for a manual quality test.')
        legacy = registry('H3StudioMediaPlan')()
        plan,w,h = legacy.prepare(mode,aspect_ratio,output_megapixels,output_megapixels,preset,
                                 8.,protect_background,**refs)
        # Do not mutate cached upstream data.
        plan = dict(plan)
        if manual_quality_steps:
            plan['steps'] = manual_quality_steps
        plan['lab_reference_size'] = reference_size_mode
        pre = [dict(v.get('lab_preprocess', {}), slot=k) for k,v in refs.items()
               if k.startswith('reference_') and v and v.get('image') is not None]
        summary = dict(plan['summary'], steps=plan['steps'], lab_version=VERSION,
                       reference_size_mode=reference_size_mode, per_reference=pre)
        # Native 'match' may scale below the independent per-reference cap.
        encoded = []
        for image in plan['images']:
            ih,iw=image.shape[1:3]
            scale=min(1.,math.sqrt(w*h/(iw*ih))) if reference_size_mode=='match' else min(1.,2048/min(iw,ih))
            encoded.append([max(32,round(iw*scale/32)*32),max(32,round(ih*scale/32)*32)])
        summary['effective_native_photo_sizes']=encoded
        summary['output_size_control']='output_megapixels (applies to the selected mode)'
        plan['summary']=summary
        return plan,w,h


def compose_roles(plan, target_description, face_slot, body_slot, preserve_other_people, match_lighting, request):
    mapping = {row['slot']:row['tag'] for row in plan['summary'].get('reference_mapping',[])}
    for slot in (face_slot, body_slot):
        if slot and slot not in mapping:
            raise ValueError('Requested role uses empty/disabled photo slot %s.'%slot)
    bits = [request.strip()] if request.strip() else []
    if plan['mode']=='Replace person':
        if not target_description.strip():
            raise ValueError('Describe which person in Picture 1 should be replaced, for example person on the left.')
        bits.append('Replace only the %s in <Picture 1>.'%target_description.strip().rstrip('.'))
        bits.append('Use <Picture 1> for the scene, composition, camera and pose, not the replacement identity.')
    if face_slot:
        bits.append('Use %s as the exclusive head, facial identity and hair reference.'%mapping[face_slot])
    if body_slot:
        bits.append('Use %s for the replacement physique and natural body proportions.'%mapping[body_slot])
    if preserve_other_people:
        bits.append('Preserve all other people and unrelated elements of the target photograph.')
    if match_lighting:
        bits.append('Match the replacement to the illumination, shadows, exposure and color of <Picture 1>.')
    result=' '.join(bits).strip()
    if not result:
        raise ValueError('Enter a scene description or edit request.')
    return result


class H3LabPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan':('H3_STILL_PLAN',),
            'use_role_builder':('BOOLEAN',{'default':True}),
            'target_description':('STRING',{'default':'person on the left'}),
            'face_slot':('INT',{'default':2,'min':0,'max':9}),
            'body_slot':('INT',{'default':3,'min':0,'max':9}),
            'preserve_other_people':('BOOLEAN',{'default':True}),
            'match_lighting':('BOOLEAN',{'default':False}),
            'request':('STRING',{'default':'Create one natural, photorealistic still image.','multiline':True})}}
    RETURN_TYPES=('STRING',)
    RETURN_NAMES=('literal_prompt',)
    FUNCTION='build'
    CATEGORY=CATEGORY
    def build(self, plan, use_role_builder, target_description,face_slot,body_slot,preserve_other_people,match_lighting,request):
        if use_role_builder:
            result=compose_roles(plan,target_description,face_slot,body_slot,preserve_other_people,match_lighting,request)
        else:
            result=request.strip()
            if not result:raise ValueError('Enter a prompt.')
        return (result,)


class H3LabConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'plan':('H3_STILL_PLAN',),'clip':('CLIP',),'vae':('VAE',),'prompt':('STRING',{'multiline':True})}}
    RETURN_TYPES=('CONDITIONING','STRING')
    RETURN_NAMES=('conditioning','preparation_report')
    FUNCTION='encode'
    CATEGORY=CATEGORY
    def encode(self,plan,clip,vae,prompt):
        # Keep native reference conditioning; neither fill/mask nor hidden prompt rewrites are added.
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
        # Reuse Studio's input-tag validation.
        module=__import__(registry('H3StudioMediaConditioning').__module__,fromlist=['validate_media_prompt'])
        module.validate_media_prompt(prompt,plan['count'],plan['video_count'])
        cuda_sync();start=time.perf_counter()
        common=dict(clip=clip,vae=vae,prompt=prompt,width=plan['width'],height=plan['height'],length=plan['conditioning_length'])
        if plan['mode']=='Generate':value=MiniMaxH3ImageToVideo.execute(**common)
        else:value=MiniMaxH3ReferenceToVideo.execute(**common,ref_image_size=plan.get('lab_reference_size','match'),
            ref_images={'ref_image_%s'%i:im for i,im in enumerate(plan['images'])},
            ref_videos={'ref_video_%s'%i:im for i,im in enumerate(plan['videos'])})
        cuda_sync()
        report=dict(plan['summary'],conditioning_seconds=time.perf_counter()-start,
                    timing_note='This time belongs to the last execution of this conditioning node. ComfyUI may reuse it from cache.',
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        return outputs(value)[0],json.dumps(report,indent=2)


class LogCapture(logging.Handler):
    def __init__(self):super().__init__(logging.INFO);self.lines=[]
    def emit(self,record):
        text=record.getMessage()
        if record.levelno>=logging.WARNING or any(x in text.lower() for x in ('spectrum','spark','forecast','actual','not loaded','lora key')):
            self.lines.append(text[:4000]);self.lines=self.lines[-200:]


class H3LabTimedSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'noise':('NOISE',),'guider':('GUIDER',),'sampler':('SAMPLER',),'sigmas':('SIGMAS',),
                             'latent_image':('LATENT',),'preparation_report':('STRING',{'multiline':True}),
                             'experiment_report':('STRING',{'multiline':True}),
                             'run_label':('STRING',{'default':'baseline'})}}
    RETURN_TYPES=('LATENT','STRING')
    RETURN_NAMES=('samples','sampling_report')
    FUNCTION='sample'
    CATEGORY=CATEGORY
    def sample(self,noise,guider,sampler,sigmas,latent_image,preparation_report,experiment_report,run_label):
        import torch
        from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
        label=short_name(run_label)
        capture=LogCapture();root=logging.getLogger();root.addHandler(capture)
        cuda_sync()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        try:
            result=SamplerCustomAdvanced.execute(noise,guider,sampler,sigmas,latent_image)
            cuda_sync()
        finally:root.removeHandler(capture)
        report={'label':label,'id':run_filename(label),'lab_version':VERSION,
                'sampling_seconds':time.perf_counter()-started,'outer_steps':len(sigmas)-1,
                'preparation':json.loads(preparation_report or '{}'),
                'experiment':json.loads(experiment_report or '{}'),
                'sampling_log':capture.lines,
                'noise_seed':getattr(noise,'seed',None),
                'executed_on_cuda':bool(torch.cuda.is_available()),
                'note':'This is one completed local run, not release-wide qualification. Log lines determine accelerator fallback status.'}
        if torch.cuda.is_available():
            report.update(device=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        exp=report['experiment'].get('accelerator','None')
        report['accelerator_execution']='not requested' if exp=='None' else 'requested; inspect captured provider logs and compare sampling time; not inferred from node selection'
        return outputs(result)[0],json.dumps(report,indent=2)


class H3LabTimedDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'samples':('LATENT',),'vae':('VAE',),'report':('STRING',{'multiline':True}),
                             'decoder':(['Fizgig','Native P-H3'],)}}
    RETURN_TYPES=('IMAGE','STRING')
    RETURN_NAMES=('image','complete_report')
    FUNCTION='decode'
    CATEGORY=CATEGORY
    def decode(self,samples,vae,report,decoder):
        cuda_sync();start=time.perf_counter()
        node=registry('FizgigH3StillDecode' if decoder=='Fizgig' else 'VAEDecode')
        if hasattr(node,'execute'): image=invoke(node,samples=samples,vae=vae)[0]
        else:image=outputs(getattr(node(),node.FUNCTION)(samples=samples,vae=vae))[0]
        cuda_sync();data=json.loads(report);data.update(decode_seconds=time.perf_counter()-start,decoded_size=[image.shape[2],image.shape[1]],decoder=decoder)
        return image,json.dumps(data,indent=2)


class H3LabReport:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{'text':('STRING',{'multiline':True}),'save_report':('BOOLEAN',{'default':False})}}
    RETURN_TYPES=()
    FUNCTION='show'
    OUTPUT_NODE=True
    CATEGORY=CATEGORY
    def show(self,text,save_report):
        if save_report:
            import folder_paths
            folder=Path(folder_paths.get_output_directory())/'H3_Lab'/'reports';folder.mkdir(parents=True,exist_ok=True)
            path=folder/(run_filename('report')+'.json')
            try:content=json.dumps(json.loads(text),indent=2)
            except ValueError:content=json.dumps({'text':text},indent=2)
            with path.open('x') as f:f.write(content+'\n')
            text += '\nSaved: '+str(path)
        return text_ui(text)


class H3LabContactSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'columns':('INT',{'default':3,'min':1,'max':3}),
                            'cell_size':('INT',{'default':512,'min':128,'max':1024,'step':32})},
                'optional':{'reference_%s'%i:('H3_STILL_REFERENCE',) for i in range(1,10)}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='build'
    CATEGORY=CATEGORY
    def build(self,columns,cell_size,**kwargs):
        import torch
        import numpy as np
        from PIL import Image,ImageDraw
        active=[(i,kwargs.get('reference_%s'%i)) for i in range(1,10)]
        active=[(i,r) for i,r in active if r and r.get('image') is not None]
        if not active:raise ValueError('Upload at least one original reference photograph.')
        cols=min(columns,len(active));rows=math.ceil(len(active)/cols)
        canvas=Image.new('RGB',(cols*cell_size,rows*(cell_size+30)),(235,235,235));draw=ImageDraw.Draw(canvas)
        for n,(slot,ref) in enumerate(active):
            image=Image.fromarray((ref['image'][0].detach().cpu().clamp(0,1).numpy()*255).round().astype('uint8'))
            image.thumbnail((cell_size,cell_size),Image.Resampling.LANCZOS)
            x=(n%cols)*cell_size;y=(n//cols)*(cell_size+30)
            canvas.paste(image,(x+(cell_size-image.width)//2,y+(cell_size-image.height)//2))
            draw.text((x+8,y+cell_size+7),'Source %s: %s'%(slot,str(ref.get('lab_role','original'))[:40]),fill=(25,25,25))
        return (torch.from_numpy(np.asarray(canvas).copy()).float().unsqueeze(0)/255.,)


NODE_CLASS_MAPPINGS={c.__name__:c for c in (H3LabReference,H3LabPlan,H3LabPrompt,H3LabConditioning,
    H3LabTimedSampler,H3LabTimedDecode,H3LabReport,H3LabContactSheet)}
