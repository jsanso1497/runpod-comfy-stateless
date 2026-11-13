"""Generate deterministic GUI workflows and API prompts without requiring a GPU."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import uuid
P=Path(__file__).resolve().parent
VERSION='2026-11-13.a40.1'


class Graph:
    def __init__(self,name):
        self.nodes={};self.api={}
        self.data=dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL,'h3-a40-identity/'+VERSION+'/'+name)),
            revision=0,last_node_id=0,last_link_id=0,nodes=[],links=[],groups=[],config={},version=.4,
            extra={'ds':{'scale':.43,'offset':[40,80]},'h3_video_studio':{'version':VERSION,'gpu_tested':False,'prompt_enhancer':False}})
    def add(self,id,kind,title,x,y,outputs=(),widgets=None,size=(330,180),collapsed=False):
        widgets=widgets or {}
        node=dict(id=id,type=kind,title=title,pos=[x,y],size=list(size),flags={'collapsed':True} if collapsed else {},
            order=len(self.nodes),mode=0,inputs=[],outputs=[dict(name=n,type=t,links=[]) for n,t in outputs],
            properties={'Node name for S&R':kind},widgets_values=list(widgets.values()))
        self.nodes[id]=node;self.data['nodes'].append(node)
        self.api[str(id)]=dict(class_type=kind,inputs={k:v for k,v in widgets.items() if k not in ('control_after_generate','upload')},_meta={'title':title})
        return node
    def link(self,source,slot,target,name,kind):
        edge=len(self.data['links'])+1;node=self.nodes[target];index=len(node['inputs'])
        entry=dict(name=name,type=kind,link=edge)
        if name in self.api[str(target)]['inputs']:
            entry['widget']={'name':name}
        node['inputs'].append(entry)
        self.nodes[source]['outputs'][slot]['links'].append(edge)
        self.data['links'].append([edge,source,slot,target,index,kind])
        self.api[str(target)]['inputs'][name]=[str(source),slot]
    def finish(self):
        self.data['last_node_id']=max(self.nodes);self.data['last_link_id']=len(self.data['links'])
        for node in self.nodes.values():
            for out in node['outputs']:
                if not out['links']:out['links']=None
        return self.data


VIDEO_PROMPT=('subject_definitions:\nThe person in <Video 1> is the sole identity reference. Preserve their facial features, '
 'natural asymmetries, hair, physique and body proportions.\n\n'
 'Create one realistic studio photograph of that same person, front-facing with a relaxed natural expression, '
 'arms relaxed, soft neutral lighting and a plain background. Keep plausible natural skin and hair detail; '
 'do not invent defining features. Remove all text, numbers or watermarks.')
COMBINED_PROMPT=('subject_definitions:\n<Picture 1> supplies the facial identity and hair. <Picture 2> supplies the physique '
 'and body proportions. <Video 1> shows the same person from additional angles and supports their identity.\n\n'
 'Create one realistic, front-facing full-body studio photograph of this person, with relaxed arms, natural proportions, '
 'a simple fitted T-shirt and jeans, soft neutral lighting and a plain background. Do not shorten or stretch the body '
 'to fill the canvas. Remove all text, numbers or watermarks.')
REPLACE_PROMPT=('Edit <Picture 1>. Replace only the woman in <Picture 1>. Use <Picture 2> for her facial identity and hair '
 'and <Picture 3> for her physique and body proportions. Any additional video reference supplies supporting identity evidence only. '
 'Preserve the woman\'s position, pose, action, facial expression, head angle, face turn and gaze from <Picture 1>. '
 'Match the original scene lighting, shadows, exposure and color. Preserve the man, background, camera framing and all '
 'unrelated objects. Do not change other people. Remove all text, numbers or watermarks.')
TEXT_PROMPT='A natural editorial photograph of a ceramic mug beside a window, soft morning light, neutral color and realistic material detail. Remove all text, numbers or watermarks.'
EDIT_PROMPT='Edit <Picture 1>. Improve photographic clarity while preserving the composition, pose, expression, camera angle, lighting and natural proportions. Avoid invented skin detail. Remove all text, numbers or watermarks.'
PHOTO_PROMPT='subject_definitions:\n<Picture 1> supplies facial identity and hair. <Picture 2> supplies physique and body proportions.\n\nCreate one natural front-facing studio photograph of the same person, with relaxed arms and soft neutral lighting. Remove all text, numbers or watermarks.'


def build(profile):
    name=profile['file'];g=Graph(name);replace=profile.get('replace',False)
    settings=dict(engine=profile.get('engine','MiniMax H3'),precision='INT8 HQ',mode=profile.get('mode','References'),
        aspect_ratio='2:3',output_megapixels=profile.get('mp',2.5),photo_cap_megapixels=2.5,reference_size='match',
        video_strategy=profile.get('strategy','Auto'),standard_steps=50,eros_steps=8,protect_background=replace)
    g.add(10,'H3A40Settings','1. H3 / EROS - A40 QUALITY SETTINGS',1080,0,[('config','H3A40_CONFIG')],settings,(510,430))
    for i in range(1,10):
        x=((i-1)%3)*350;y=((i-1)//3)*330
        if replace and i==1:
            g.add(1,'LoadImage','PHOTO 1: target scene + Mask Editor',x,y,[('IMAGE','IMAGE'),('MASK','MASK')],
                  dict(image='UPLOAD_TARGET_PHOTO.png',upload='image'),(325,280))
            g.add(31,'H3StudioTargetPhoto','Target and replacement mask',0,1020,[('target_image_1','H3_STILL_REFERENCE')],size=(325,100))
            g.link(1,0,31,'image','IMAGE');g.link(1,1,31,'mask','MASK')
        else:
            g.add(i,'H3StudioOptionalImage','PHOTO SLOT %d (optional)'%i,x,y,[('reference','H3_STILL_REFERENCE')],
                dict(image='(none)',enabled=True,upload='image'),(325,280),collapsed=i>3)
    video_widgets=dict(file='(none)',enabled=profile.get('video',True),start_seconds=0.,duration_seconds=3.1,
        native_video_megapixels=profile.get('video_mp',1.),keyframe_megapixels=2.5,keyframe_count=4,keyframe_times='',
        crop_left_percent=0.,crop_top_percent=0.,crop_right_percent=0.,crop_bottom_percent=0.)
    g.add(11,'H3A40Video','2. NATIVE VIDEO - START WITH 3.1 SECONDS',1080,480,[('video','H3A40_VIDEO')],video_widgets,(510,510))
    g.link(10,0,11,'config','H3A40_CONFIG')
    g.add(12,'H3A40Plan','3. LITERAL PROMPT + REFERENCE MAPPING',1660,0,
        [('plan','H3A40_PLAN'),('actual_references','IMAGE'),('reference_report','STRING')],
        dict(prompt=profile['prompt']),(620,390))
    g.link(10,0,12,'config','H3A40_CONFIG');g.link(11,0,12,'video','H3A40_VIDEO')
    for i in range(1,10):g.link(31 if replace and i==1 else i,0,12,'reference_%d'%i,'H3_STILL_REFERENCE')
    g.add(13,'PreviewImage','Actual image inputs / native-video preview frames',1660,460,size=(620,500))
    g.link(12,1,13,'images','IMAGE')
    g.add(14,'H3A40Audit','Inspect reference tags and exact prompt',1660,1000,widgets=dict(report='',save_report=False),size=(620,300))
    g.link(12,2,14,'report','STRING')
    if not profile.get('preflight'):
        widgets=dict(encoding_report='',seed=18071042,control_after_generate='fixed')
        for i in (1,2):widgets.update({f'lora_{i}_enabled':False,f'lora_{i}_name':'(none)',f'lora_{i}_strength':.6,f'lora_{i}_family':'H3 video / Eros'})
        g.add(15,'H3A40Render','5. GENERATE STILL / OPTIONAL LORAS',2980,0,
            [('final_image','IMAGE'),('reference_left_result_right','IMAGE'),('run_report','STRING')],widgets,(530,440))
        g.link(12,0,15,'plan','H3A40_PLAN')
        g.add(19,'H3A40Encode','4. ENCODE REFERENCES ONCE PER INPUT CHANGE',2360,0,
            [('conditioning','CONDITIONING'),('single_frame_latent','LATENT'),('vae','VAE'),('encoding_report','STRING')],size=(530,240))
        g.link(12,0,19,'plan','H3A40_PLAN')
        for slot,name,kind in [(0,'conditioning','CONDITIONING'),(1,'latent','LATENT'),(2,'vae','VAE'),(3,'encoding_report','STRING')]:
            g.link(19,slot,15,name,kind)
        g.add(20,'H3A40Audit','Encoding time and actual native canvas',2360,360,
            widgets=dict(report='',save_report=False),size=(530,420))
        g.link(19,3,20,'report','STRING')
        g.add(16,'SaveImage','FINAL LOSSLESS PNG',3600,0,widgets=dict(filename_prefix='H3_A40_Identity/'+profile.get('prefix','Final')),size=(620,590))
        g.link(15,0,16,'images','IMAGE')
        g.add(17,'PreviewImage','Reference left / result right (not pixel registration)',3600,650,size=(620,470))
        g.link(15,1,17,'images','IMAGE')
        g.add(18,'H3A40Audit','Sampling timing / VRAM / LoRA warnings',2980,510,widgets=dict(report='',save_report=True),size=(530,480))
        g.link(15,2,18,'report','STRING')
    notes=('H3 A40 IDENTITY STUDIO '+VERSION+'\n\n'
        'Native H3 VIDEO model to ONE still PNG. MiniMax H3 and Eros are both available in the same model dropdown.\n'
        'INT8 diffusion + INT8 32B reference encoder + FP16 video VAE. No BF16 full stack or four-bit encoder.\n'
        'Start: 3.1 seconds, 1.0 MP native video setting, 2.5 MP output, up to two selected face/body photos. '
        'The stock conditioning node may resize native video further to its 768-short-edge canvas; exact size is logged.\n'
        'The four native preview thumbnails do NOT mean only four frames are used. Default clip retains 73 frames when source permits.\n'
        'Seed-only reruns reuse the separate encoding node through ordinary ComfyUI execution caching. '
        'Changing prompt/references/settings can rerun encoding. The cache is not persistent across restart.\n'
        'Standard H3: 50 steps, ER-SDE/simple, no Turbo. Eros Fix1: 8 steps, Euler/simple, 12/3 shift, baked Turbo. '
        'No extra Turbo adapters, Spectrum, VSA, Sage or approximate denoising caches.\n'
        'No 2 MP native override. Duration UI is 0.25-6 seconds; decoded-pixel guard restored. '
        '4 MP OUTPUT is an optional final test, not a proven A40 memory limit or a guaranteed quality increase.\n'
        'Photo tags follow enabled slot order. Video is <Video 1> in Auto mode. Preview references before rendering. '
        'With Keyframes for all engines, use 1-8 extracted frames and watch the nine-image total.\n'
        'The focused image downloads four assets only. Older workflow files and existing models are never deleted.\n'
        'Mask protection is final compositing, NOT diffusion inpainting. Pose drift inside the mask can still happen.\n'
        'Save an edited workflow under your own name. No GPU benchmark or identity-quality test was run by the author.\n'
        'Powered by MiniMax H3. Use only with applicable model authorization.')
    g.add(30,'Note','READ FIRST',0,1190,widgets={'text':notes},size=(1500,490))
    g.data['groups']=[dict(title='PHOTO REFERENCES',bounding=[-20,-60,1060,1200],color='#3f789e',font_size=24,flags={}),
        dict(title='MODEL AND VIDEO CONTROLS',bounding=[1060,-60,550,1100],color='#497e61',font_size=24,flags={}),
        dict(title='PROMPT AND INPUT AUDIT',bounding=[1640,-60,660,1400],color='#766393',font_size=24,flags={})]
    gui=g.finish();api={k:v for k,v in g.api.items() if v['class_type']!='Note'}
    return gui,api

PROFILES=[
 dict(file='00_H3_A40_Identity_Quality.json',prompt=VIDEO_PROMPT,prefix='Quality'),
 dict(file='01_H3_A40_Video_Plus_Photos.json',prompt=COMBINED_PROMPT,prefix='Video_Photos'),
 dict(file='02_H3_A40_Eros_Identity.json',prompt=VIDEO_PROMPT,engine='Eros Beta 7 Fix1',prefix='Eros'),
 dict(file='03_H3_A40_4MP_Final_Test.json',prompt=VIDEO_PROMPT,mp=4.,prefix='4MP_Test'),
 dict(file='04_H3_A40_Reference_Preflight.json',prompt=VIDEO_PROMPT,preflight=True,prefix='Preflight'),
 dict(file='05_H3_A40_Photo_References.json',prompt=PHOTO_PROMPT,video=False,prefix='Photos'),
 dict(file='06_H3_A40_Replace_Person.json',prompt=REPLACE_PROMPT,replace=True,mode='Replace person',prefix='Replacement'),
]


def main():
    for directory in ('workflows','api_workflows'):(P/directory).mkdir(exist_ok=True)
    for profile in PROFILES:
        gui,api=build(profile)
        for directory,data in [('workflows',gui),('api_workflows',api)]:
            (P/directory/profile['file']).write_text(json.dumps(data,indent=2)+'\n')
    print('Created %d unified presets; every generation preset retains the same engine selector.'%len(PROFILES))
if __name__=='__main__':main()
