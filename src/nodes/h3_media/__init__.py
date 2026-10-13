"""Reference input adapter for native MiniMax H3, not a model implementation."""
from __future__ import annotations
import inspect
import json
import math
import re
import torch
from . import media_io
from .export_helpers import H3PortraitExportVideo, H3PortraitSaveLastFrame

VERSION='1.2.0'
VISUAL_ROLES=['Ignore video (voice only)','Identity - selected frame','Action / motion','Identity + action']
AUDIO_ROLES=['Voice timbre - NEW dialogue','Ambience / sound texture','Ignore audio']
ASPECTS={'16:9 landscape':(1344,768),'9:16 portrait':(768,1344),'3:2 landscape':(1152,768),
         '2:3 portrait':(768,1152),'1:1 square':(1024,1024),'4:5 portrait':(896,1120)}


def text(value,name):
    if not isinstance(value,str) or not value.strip(): raise ValueError(name+' must contain text.')
    return value.strip()


def image(value):
    if not isinstance(value,torch.Tensor) or value.ndim!=4 or value.shape[0]!=1 or value.shape[-1] not in (3,4):
        raise ValueError('Each still reference must be one ComfyUI RGB/RGBA image.')
    if min(value.shape[1:3])<16 or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError('Invalid reference-image pixels.')
    return value[...,:3]


def bank(previous):
    if previous is None:return []
    if not isinstance(previous,list) or any(not isinstance(r,dict) or r.get('kind') not in ('image','video','audio') for r in previous):
        raise ValueError('Connect an H3 Media reference-list output.')
    return list(previous)


def check_limits(refs):
    counts={'image':0,'video':0,'audio':0,'clips':0}
    for item in refs:
        if item['kind']=='image':counts['image']+=1
        else:
            if item.get('identity') is not None:counts['image']+=1
            if item.get('motion') is not None:counts['video']+=1
            if item.get('audio') is not None:counts['audio']+=1
            if item['kind']=='video':counts['clips']+=1
    if counts['image']>9 or counts['video']>3 or counts['audio']>3 or counts['clips']>3:
        raise ValueError('Native H3 limit: 9 still references, 3 motion clips, 3 standalone audio references; this workflow allows 3 source videos. Disable unused roles or remove references.')
    if not sum(counts[k] for k in ('image','video','audio')):
        raise ValueError('At least one active reference is needed.')
    return counts


class H3MediaImageReference:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{
        'image':('IMAGE',),'label':('STRING',{'default':'Primary face'}),
        'use_for':('STRING',{'default':'Facial identity and distinctive facial proportions. Do not copy the background or pose.','multiline':True}),
        'applies_to':('STRING',{'default':'the main subject'})},
        'optional':{'references':('H3_MEDIA_REFS',)}}
    RETURN_TYPES=('H3_MEDIA_REFS',)
    RETURN_NAMES=('references',)
    FUNCTION='append'
    CATEGORY='H3 Media HQ'
    def append(self,image,label,use_for,applies_to,references=None):
        rows=bank(references)+[{'kind':'image','image':globals()['image'](image),
            'label':text(label,'Label'),'use_for':text(use_for,'Reference purpose'),'target':text(applies_to,'Applies to')}]
        check_limits(rows)
        return (rows,)


class H3MediaVideoReference:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{
        'video':('VIDEO',),'label':('STRING',{'default':'Voice clip'}),
        'applies_to':('STRING',{'default':'the main subject'}),
        'visual_role':(VISUAL_ROLES,{'default':VISUAL_ROLES[0]}),
        'audio_role':(AUDIO_ROLES,{'default':AUDIO_ROLES[0]}),
        'start_seconds':('FLOAT',{'default':0.,'min':0.,'max':2592000.,'step':.1}),
        'duration_seconds':('FLOAT',{'default':5.,'min':.25,'max':15.,'step':.1}),
        'identity_frame_position':('FLOAT',{'default':.5,'min':0.,'max':1.,'step':.01}),
        'audio_track':('INT',{'default':0,'min':0,'max':15}),
        'extra_use':('STRING',{'default':'','multiline':True})},
        'optional':{'references':('H3_MEDIA_REFS',)}}
    RETURN_TYPES=('H3_MEDIA_REFS','STRING')
    RETURN_NAMES=('references','clip_report')
    FUNCTION='append'
    CATEGORY='H3 Media HQ'
    DESCRIPTION='Native LoadVideo -> bounded MP4 audio/frame extraction. Voice-only never sends or decodes reference video frames.'
    def append(self,video,label,applies_to,visual_role,audio_role,start_seconds,duration_seconds,
               identity_frame_position,audio_track,extra_use='',references=None):
        if visual_role not in VISUAL_ROLES or audio_role not in AUDIO_ROLES:raise ValueError('Invalid media role.')
        if visual_role==VISUAL_ROLES[0] and audio_role=='Ignore audio':raise ValueError('Both streams are ignored. Remove this reference node instead.')
        rows=bank(references)
        if sum(r['kind']=='video' for r in rows)>=3:raise ValueError('Use at most three source video clips.')
        label=text(label,'Clip label');target=text(applies_to,'Applies to')
        identity,motion,audio,meta=media_io.decode_reference(video,start_seconds,duration_seconds,
            visual_role,audio_role,identity_frame_position,audio_track)
        rows.append({'kind':'video','identity':identity,'motion':motion,'audio':audio,
            'visual_role':visual_role,'audio_role':audio_role,'target':target,'label':label,
            'use_for':str(extra_use).strip(),'metadata':meta})
        check_limits(rows)
        return rows,json.dumps(meta,indent=2)


class H3MediaAudioReference:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{
        'audio':('AUDIO',),'label':('STRING',{'default':'Voice reference'}),
        'applies_to':('STRING',{'default':'the main subject'}),
        'audio_role':(AUDIO_ROLES[:2],{'default':AUDIO_ROLES[0]}),
        'start_seconds':('FLOAT',{'default':0.,'min':0.,'max':2592000.,'step':.1}),
        'duration_seconds':('FLOAT',{'default':5.,'min':.25,'max':15.,'step':.1})},
        'optional':{'references':('H3_MEDIA_REFS',)}}
    RETURN_TYPES=('H3_MEDIA_REFS',)
    RETURN_NAMES=('references',)
    FUNCTION='append'
    CATEGORY='H3 Media HQ'
    def append(self,audio,label,applies_to,audio_role,start_seconds,duration_seconds,references=None):
        if audio_role not in AUDIO_ROLES[:2]:raise ValueError('Invalid audio role.')
        start=media_io.number(start_seconds,'Start seconds',0,2592000)
        duration=media_io.number(duration_seconds,'Duration seconds',.25,15)
        rate=audio.get('sample_rate') if isinstance(audio,dict) else None
        wave=audio.get('waveform') if isinstance(audio,dict) else None
        if type(rate) is not int or not 8000<=rate<=192000 or not isinstance(wave,torch.Tensor) or wave.ndim!=3 or wave.shape[0]!=1 or wave.shape[1] not in (1,2) or not wave.is_floating_point():
            raise ValueError('Use a mono or stereo ComfyUI AUDIO input.')
        clip=wave[...,round(start*rate):round((start+duration)*rate)]
        if clip.shape[-1]<rate//4 or not torch.isfinite(clip).all() or clip.abs().max()<1e-6:
            raise ValueError('Selected audio is empty, invalid, silent or shorter than 0.25 seconds.')
        rows=bank(references)+[{'kind':'audio','audio':{'waveform':clip,'sample_rate':rate},
            'identity':None,'motion':None,'audio_role':audio_role,'label':text(label,'Label'),
            'target':text(applies_to,'Applies to'),'use_for':'','metadata':{'audio_seconds':clip.shape[-1]/rate}}]
        check_limits(rows)
        return (rows,)


def compile_references(refs):
    check_limits(refs)
    images={};videos={};audios={};manifest=[]
    for item in refs:
        pic=item['image'] if item['kind']=='image' else item.get('identity')
        if pic is not None:
            tag=f'<Picture {len(images)+1}>';images[f'ref_image_{len(images)}']=pic
            purpose=item['use_for'] if item['kind']=='image' else 'Visual identity, facial geometry, hair and body appearance from the selected original video frame. '+item.get('use_for','')
            manifest.append({'tag':tag,'label':item['label'],'target':item['target'],'role':'identity / appearance','purpose':purpose})
    for item in refs:
        frames=item.get('motion')
        if frames is not None:
            tag=f'<Video {len(videos)+1}>';videos[f'ref_video_{len(videos)}']=frames
            purpose='Action, motion, gesture and performance rhythm. '
            if item['visual_role']=='Action / motion':purpose+='Do not transfer the performer identity, wardrobe, background or camera framing. '
            else:purpose+='Also reference the visible person identity, with the selected still providing facial detail. '
            manifest.append({'tag':tag,'label':item['label'],'target':item['target'],'role':item['visual_role'],'purpose':purpose+item.get('use_for','')})
    # Deliberately all standalone: adding a video never renumbers paired audio
    # ahead of an existing voice. This is conditioning, NOT source-track replay.
    for item in refs:
        audio=item.get('audio')
        if audio is not None:
            tag=f'<Audio {len(audios)+1}>';audios[f'ref_audio_{len(audios)}']=audio
            purpose=('Voice timbre, accent and delivery only. Generate the NEW scripted words; do not repeat the source words, music or background sounds.'
                     if item['audio_role']==AUDIO_ROLES[0] else 'Ambience and sound texture only, without replaying the original waveform.')
            manifest.append({'tag':tag,'label':item['label'],'target':item['target'],'role':item['audio_role'],'purpose':purpose+' '+item.get('use_for','')})
    return {'ref_images':images,'ref_videos':videos,'ref_audios':audios,'ref_video_audios':{}},manifest


def build_prompt(manifest,scene,dialogue,language,speaker,extra,override):
    scene=text(scene,'Scene description');speaker=text(speaker,'Speaker target');language=text(language,'Language')
    if not re.fullmatch(r'[\w -]{1,48}',language):raise ValueError('Use a language name such as English, not dialogue markup.')
    if not all(isinstance(v,str) for v in (dialogue,extra,override)):raise ValueError('Dialogue and instructions must be text.')
    targets=list(dict.fromkeys(r['target'] for r in manifest))
    subjects={target:f'<Subject {i+1}>' for i,target in enumerate(targets)}
    definitions=[];retention=[]
    for target in targets:
        visual=[r for r in manifest if r['target']==target and not r['tag'].startswith('<Audio')]
        details=' '.join(f"From {r['tag']} ({r['label']}): {r['purpose']}" for r in visual)
        definitions.append(f"{subjects[target]} is {target}. "+details)
        if visual:
            retention.append(f"{subjects[target]} (appears in [Shot 1]): fully_preserved - retain the requested identity / appearance / action attributes only; the text defines the new scene, wardrobe and camera.")
    for row in manifest:
        if not row['tag'].startswith('<Audio'):continue
        subject=subjects[row['target']]
        speaker_id=' (S1)' if row['role']==AUDIO_ROLES[0] and row['target']==speaker and dialogue.strip() else ''
        definitions.append(f"{row['tag']} ({row['label']}) is the {row['role']} reference for {subject}{speaker_id}: {row['purpose']}")
        retention.append(f"{row['tag']} (audible characteristics in [Shot 1]): reference - {row['purpose']}")
    action=scene
    if dialogue.strip():
        if '<d>' in dialogue or '</d>' in dialogue:raise ValueError('Enter plain new dialogue; the workflow adds the <d> markup. Use Advanced prompt for hand-written markup.')
        voices=[r['tag'] for r in manifest if r['role']==AUDIO_ROLES[0] and r['target']==speaker]
        if not override.strip() and not voices and any(r['role']==AUDIO_ROLES[0] for r in manifest):
            raise ValueError('Speaker target must match Applies to on at least one voice reference. Use Advanced prompt for multiple scripted speakers.')
        voice=' matching the voice timbre of '+', '.join(voices) if voices else ''
        action+=f'\n{subjects.get(speaker,speaker)} ({speaker}, S1) speaks{voice}, with natural mouth movements and expression: <d>[{language}] {dialogue.strip()}</d>'
    else:action+='\nNo spoken dialogue unless explicitly described above.'
    if extra.strip():action+='\n'+extra.strip()
    prompt=('subject_definitions:\n'+'\n'.join(definitions)+'\n\nsummary:\n'
        '[reference generation'+(' + audio reference' if any(r['tag'].startswith('<Audio') for r in manifest) else '')+'] '
        'Create one new realistic video from the specified reference roles. References are guidance, not a request to replay their recordings.\n\n'
        'retention_analysis:\n'+'\n'.join(retention)+'\n\ndetailed_description:\n'
        'Photorealistic rendering with consistent anatomy, natural skin texture, coherent lighting and stable identities.\n[Shot 1] '+action+
        '\n\noverall_soundscape:\nNatural scene ambience at a low level. Foreground dialogue, when present, is clear. '
        'Do not copy unrelated noise or original dialogue from voice references.\n\nnon_diegetic_music:\nNone unless explicitly requested in the scene description.')
    if override.strip():prompt=override.strip()
    valid={r['tag'] for r in manifest}
    for tag in re.findall(r'<(?:Picture|Video|Audio)\s+\d+>',prompt):
        if tag not in valid:raise ValueError('Prompt refers to unavailable '+tag+'. Check the reference map after changing roles.')
    return prompt


class H3MediaPlan:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{
        'references':('H3_MEDIA_REFS',),
        'scene_description':('STRING',{'default':'A photorealistic medium shot of the main subject in a softly lit room, looking at the camera. Subtle natural movement, relaxed posture, neutral clothing and a locked eye-level camera.','multiline':True}),
        'new_dialogue':('STRING',{'default':'This is a new line, spoken in the reference voice.','multiline':True}),
        'speaker_target':('STRING',{'default':'the main subject'}),
        'language':('STRING',{'default':'English'}),
        'aspect':(list(ASPECTS),{'default':'16:9 landscape'}),
        'seconds':('INT',{'default':8,'min':3,'max':15}),
        'ref_image_size':(['max','match'],{'default':'max'}),
        'extra_instruction':('STRING',{'default':'','multiline':True}),
        'advanced_prompt':('STRING',{'default':'','multiline':True})}}
    RETURN_TYPES=('H3_MEDIA_PLAN','STRING')
    RETURN_NAMES=('ready_plan','prompt_and_reference_map')
    FUNCTION='prepare'
    CATEGORY='H3 Media HQ'
    def prepare(self,references,scene_description,new_dialogue,speaker_target,language,aspect,seconds,
                ref_image_size,extra_instruction='',advanced_prompt=''):
        if aspect not in ASPECTS or ref_image_size not in ('max','match'):raise ValueError('Invalid output/reference size setting.')
        if type(seconds) is not int or not 3<=seconds<=15:raise ValueError('Duration must be an integer from 3 to 15 seconds.')
        kwargs,manifest=compile_references(references)
        prompt=build_prompt(manifest,scene_description,new_dialogue,language,speaker_target,extra_instruction,advanced_prompt)
        length=5+math.ceil((seconds*24-5)/17)*17
        w,h=ASPECTS[aspect]
        plan={**kwargs,'prompt':prompt,'width':w,'height':h,'length':length,'ref_image_size':ref_image_size}
        report={'reference_map':manifest,'output':{'width':w,'height':h,'frames':length,'fps':24,'seconds':length/24},
            'source_clip_details':[r.get('metadata',{}) for r in references if r['kind']!='image'],
            'audio_mode':'Generated H3 audio, conditioned by references. No input-audio passthrough.',
            'warning':'Voice resemblance and motion following are model-dependent, not exact cloning or pose locking.'}
        return plan,json.dumps(report,indent=2)+'\n\n'+prompt


class H3MediaConditioning:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{
        'ready_plan':('H3_MEDIA_PLAN',),'clip':('CLIP',),'vae':('VAE',),'audio_vae':('VAE',)}}
    RETURN_TYPES=('CONDITIONING','LATENT')
    RETURN_NAMES=('positive','latent')
    FUNCTION='encode'
    CATEGORY='H3 Media HQ'
    def encode(self,ready_plan,clip,vae,audio_vae):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        required={'clip','vae','audio_vae','prompt','width','height','length','ref_image_size',
                  'ref_images','ref_videos','ref_video_audios','ref_audios'}
        if not required.issubset(inspect.signature(MiniMaxH3ReferenceToVideo.execute).parameters):
            raise RuntimeError('Installed native H3 API is incompatible with the media adapter; do not silently drop audio/video inputs.')
        if set(ready_plan) != required-{'clip','vae','audio_vae'}:
            raise ValueError('The ready plan is not from the H3 Media Plan node.')
        out=MiniMaxH3ReferenceToVideo.execute(clip=clip,vae=vae,audio_vae=audio_vae,**ready_plan)
        return out[0],out[1]


NODE_CLASS_MAPPINGS={c.__name__:c for c in (H3MediaImageReference,H3MediaVideoReference,H3MediaAudioReference,H3MediaPlan,H3MediaConditioning)}
NODE_CLASS_MAPPINGS.update(H3MediaExportVideo=H3PortraitExportVideo,H3MediaSaveLastFrame=H3PortraitSaveLastFrame)
NODE_DISPLAY_NAME_MAPPINGS={
    'H3MediaImageReference':'H3 HQ - Labeled Image Reference',
    'H3MediaVideoReference':'H3 HQ - MP4 Voice / Identity / Action',
    'H3MediaAudioReference':'H3 HQ - Standalone Audio Reference',
    'H3MediaPlan':'H3 HQ - Scene + NEW Dialogue + Reference Map',
    'H3MediaConditioning':'H3 HQ - Native Reference Conditioning',
    'H3MediaExportVideo':'H3 HQ - Save Generated Video + Audio',
    'H3MediaSaveLastFrame':'H3 HQ - Save Final Frame for Chaining',
}
