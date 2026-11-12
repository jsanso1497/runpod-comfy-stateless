"""Generate deterministic GUI workflows and API prompts without requiring a GPU."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import uuid
P=Path(__file__).resolve().parent
VERSION='2026-11-11.1'


class Graph:
    def __init__(self,name):
        self.nodes={};self.api={}
        self.data=dict(id=str(uuid.uuid5(uuid.NAMESPACE_URL,'h3-video-studio/'+VERSION+'/'+name)),
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
        aspect_ratio='2:3',output_megapixels=profile.get('mp',2.5),photo_cap_megapixels=4.,reference_size='max',
        video_strategy=profile.get('strategy','Auto'),standard_steps=50,eros_steps=8,protect_background=replace)
    g.add(10,'H3VSettings','1. MODEL / PRECISION / MODE / SIZE',1080,0,[('config','H3V_CONFIG')],settings,(510,430))
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
        native_video_megapixels=1.,keyframe_megapixels=2.5,keyframe_count=4,keyframe_times='',
        crop_left_percent=0.,crop_top_percent=0.,crop_right_percent=0.,crop_bottom_percent=0.)
    g.add(11,'H3VVideo','2. OPTIONAL VIDEO / CROP / KEYFRAMES',1080,480,[('video','H3V_VIDEO')],video_widgets,(510,510))
    g.link(10,0,11,'config','H3V_CONFIG')
    g.add(12,'H3VPlan','3. LITERAL PROMPT + REFERENCE MAPPING',1660,0,
        [('plan','H3V_PLAN'),('actual_references','IMAGE'),('reference_report','STRING')],
        dict(prompt=profile['prompt']),(620,390))
    g.link(10,0,12,'config','H3V_CONFIG');g.link(11,0,12,'video','H3V_VIDEO')
    for i in range(1,10):g.link(31 if replace and i==1 else i,0,12,'reference_%d'%i,'H3_STILL_REFERENCE')
    g.add(13,'PreviewImage','Actual image inputs / native-video preview frames',1660,460,size=(620,500))
    g.link(12,1,13,'images','IMAGE')
    g.add(14,'H3VAudit','Inspect reference tags and exact prompt',1660,1000,widgets=dict(report='',save_report=False),size=(620,300))
    g.link(12,2,14,'report','STRING')
    if not profile.get('preflight'):
        widgets=dict(seed=18071042,control_after_generate='fixed')
        for i in (1,2):widgets.update({f'lora_{i}_enabled':False,f'lora_{i}_name':'(none)',f'lora_{i}_strength':.6,f'lora_{i}_family':'H3 video / Eros'})
        g.add(15,'H3VRender','4. GENERATE ONE STILL / OPTIONAL LORAS',2360,0,
            [('final_image','IMAGE'),('reference_left_result_right','IMAGE'),('run_report','STRING')],widgets,(530,440))
        g.link(12,0,15,'plan','H3V_PLAN')
        g.add(16,'SaveImage','FINAL LOSSLESS PNG',2980,0,widgets=dict(filename_prefix='H3_Video_Studio/'+profile.get('prefix','Final')),size=(620,590))
        g.link(15,0,16,'images','IMAGE')
        g.add(17,'PreviewImage','Reference left / result right (not pixel registration)',2980,650,size=(620,470))
        g.link(15,1,17,'images','IMAGE')
        g.add(18,'H3VAudit','Run timing / VRAM / models / LoRA warnings',2360,510,widgets=dict(report='',save_report=True),size=(530,480))
        g.link(15,2,18,'report','STRING')
    notes=('H3 VIDEO STILL STUDIO '+VERSION+'\n\n'
        'The MODEL dropdown changes only one selected backend, not three simultaneous runs.\n'
        'Auto video: MiniMax/Eros receive native reference video; P-H3 receives source-file keyframes. '
        'Keyframes for all engines is useful for a controlled comparison.\n'
        'Active photo slots are numbered consecutively. Video keyframes are appended after the active photos. '
        '<Video 1> is expanded to their Picture tags on keyframe routes. Read the reference report.\n'
        'Standard H3: 50 steps, no Turbo. Eros Fix1: 8 steps, baked Turbo, 12/3 shift. P-H3: fixed native 8-step schedule. '
        'No prompt enhancer, Spark, Spectrum, VSA or automatic upscaler.\n'
        'Photo/image cap is independent of output MP. Video crop is applied before downscaling. '
        'P-H3 has 8 TOTAL image slots, including video keyframes; H3/Eros have 9. No inputs are silently discarded.\n'
        'Start at 2.5 MP for video H3/Eros, then compare 4 MP. Try 8 MP only after checking VRAM. P-H3 uses 4 MP here; 8 MP is optional.\n'
        'BF16 files are optional: python3 /opt/h3-still/video_studio/manage.py download bf16 . '
        'Use only with applicable model authorization. No model weights are in this JSON.\n'
        'Preserved Lab workflows provide SAM, reference crops and RefMods. Accelerator experiments remain separate.\n'
        'Background protection requires a painted target mask and composites ORIGINAL pixels outside it. It does not guide diffusion. '
        'It can leave remnants if the replacement pose leaves the mask.\n'
        'Factory files never overwrite saved user copies. Save your edited workflow under a personal name.\n'
        'No GPU inference has been run by the package author in this environment. Validate one engine at a time.\n\n'
        'Powered by MiniMax H3. Check LICENSE_NOTES.md before use.')
    g.add(30,'Note','READ FIRST',0,1190,widgets={'text':notes},size=(1500,490))
    g.data['groups']=[dict(title='PHOTO REFERENCES',bounding=[-20,-60,1060,1200],color='#3f789e',font_size=24,flags={}),
        dict(title='MODEL AND VIDEO CONTROLS',bounding=[1060,-60,550,1100],color='#497e61',font_size=24,flags={}),
        dict(title='PROMPT AND INPUT AUDIT',bounding=[1640,-60,660,1400],color='#766393',font_size=24,flags={})]
    gui=g.finish();api={k:v for k,v in g.api.items() if v['class_type']!='Note'}
    return gui,api

PROFILES=[
 dict(file='00_H3_Unified_Video_To_Still.json',prompt=VIDEO_PROMPT,prefix='Video_Still'),
 dict(file='01_H3_Unified_Video_And_Photos.json',prompt=COMBINED_PROMPT,prefix='Video_Photos'),
 dict(file='02_H3_Unified_Replace_Person.json',prompt=REPLACE_PROMPT,replace=True,mode='Replace person',prefix='Replacement'),
 dict(file='03_H3_Unified_Text_To_Image.json',prompt=TEXT_PROMPT,mode='Generate',video=False,mp=4.,prefix='Text_Image'),
 dict(file='04_H3_Unified_Single_Photo_Edit.json',prompt=EDIT_PROMPT,mode='Edit',video=False,prefix='Photo_Edit'),
 dict(file='05_H3_Unified_Photo_References.json',prompt=PHOTO_PROMPT,mode='References',video=False,prefix='Photo_References'),
 dict(file='06_H3_Unified_Keyframe_Comparison.json',prompt=VIDEO_PROMPT,strategy='Keyframes for all engines',prefix='Keyframe_Compare'),
 dict(file='07_H3_Reference_Preflight_No_Generation.json',prompt=VIDEO_PROMPT,preflight=True),
 dict(file='08_P_H3_Video_Keyframes_To_Still.json',prompt=VIDEO_PROMPT,engine='P-H3 Image',mp=4.,prefix='PH3_Video_Still'),
 dict(file='09_Eros_Fix1_Video_To_Still.json',prompt=VIDEO_PROMPT,engine='Eros Beta 7 Fix1',prefix='Eros_Video_Still')]


def main():
    for directory in ('workflows','api_workflows'):(P/directory).mkdir(exist_ok=True)
    for profile in PROFILES:
        gui,api=build(profile)
        for directory,data in [('workflows',gui),('api_workflows',api)]:
            (P/directory/profile['file']).write_text(json.dumps(data,indent=2)+'\n')
    print('Created %d unified presets; every generation preset retains the same engine selector.'%len(PROFILES))
if __name__=='__main__':main()
