#!/usr/bin/env python3
"""Generate named Lab tests. The user-confirmed diagnostic graph is retained byte-for-byte."""
from pathlib import Path
import copy
import json
import uuid

P=Path(__file__).resolve().parent
QUALITY='Quality - 50 steps, no Turbo'
ENC='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
VAE='minimax_h3_video_vae_int8_convrot.safetensors'

class Graph:
    def __init__(self,name):
        self.nodes={};self.links=[];self.name=name
    def add(self,id,kind,title,pos=(0,0),out=(),widgets=None,size=(360,180),collapsed=False):
        if id in self.nodes:raise ValueError('Duplicate node')
        self.nodes[id]={'id':id,'type':kind,'title':title,'pos':list(pos),'size':list(size),
            'flags':{'collapsed':True} if collapsed else {},'order':len(self.nodes),'mode':0,
            'inputs':[],'outputs':[{'name':name,'type':typ,'links':[]} for name,typ in out],
            'properties':{'Node name for S&R':kind},'widgets_values':list((widgets or {}).values()),
            'widgets_values_named':widgets or {}}
        if not kind.startswith(('H3','Fizgig')) and kind!='Note':self.nodes[id]['properties']['cnr_id']='comfy-core'
        return id
    def link(self,s,slot,t,name,typ,widget=False):
        n=len(self.links)+1;node=self.nodes[t];index=len(node['inputs'])
        if any(p['name']==name for p in node['inputs']):raise ValueError('Duplicate destination input')
        entry={'name':name,'type':typ,'link':n}
        if widget:entry['widget']={'name':name}
        node['inputs'].append(entry);self.nodes[s]['outputs'][slot]['links'].append(n)
        self.links.append([n,s,slot,t,index,typ])
    def finish(self):
        # True topological execution ordering; does not rely on drawing order.
        remaining=set(self.nodes);done=set();order=[]
        while remaining:
            ready=sorted(n for n in remaining if all(e[1] in done for e in self.links if e[3]==n))
            if not ready:raise ValueError('Graph contains a cycle')
            order.extend(ready);remaining.difference_update(ready);done.update(ready)
        for i,n in enumerate(order):self.nodes[n]['order']=i
        for node in self.nodes.values():
            for port in node['outputs']:
                if not port['links']:port['links']=None
        return {'id':str(uuid.uuid5(uuid.NAMESPACE_URL,'h3-studio-lab-2026-11-08.1/'+self.name)),
            'revision':0,'last_node_id':max(self.nodes),'last_link_id':len(self.links),
            'nodes':list(self.nodes.values()),'links':self.links,'groups':[], 'config':{},'version':.4,
            'extra':{'ds':{'scale':.35,'offset':[30,60]},'h3_lab':{'release':'2026-11-08.1','gpu_tested':False,'name':self.name}}}


def references(g,replace=False,count=9):
    for i in range(1,count+1):
        x=((i-1)%3)*420;y=((i-1)//3)*900
        if i==1 and replace:
            g.add(1,'LoadImage','PICTURE 1 - TARGET SCENE',(x,y),[('IMAGE','IMAGE'),('MASK','MASK')],
                  {'image':'UPLOAD_TARGET_PHOTO.png','upload':'image'},(380,330))
            g.add(31,'H3StudioTargetPhoto','Original target + selected mask',(x,y+370),[('reference','H3_STILL_REFERENCE')],size=(380,110))
            g.link(1,0,31,'image','IMAGE');g.link(1,1,31,'mask','MASK')
        else:
            role='face and hair' if replace and i==2 else 'body proportions' if replace and i==3 else 'supporting reference'
            g.add(i,'H3StudioOptionalImage','PICTURE %s - %s'%(i,role),(x,y),[('reference','H3_STILL_REFERENCE')],
                  {'image':'(none)','enabled':True,'upload':'image'},(380,300),i>3)
            g.add(100+i,'H3LabReference','Crop and size - Picture %s'%i,(x,y+330),
                  [('reference','H3_STILL_REFERENCE'),('preview','IMAGE'),('details','STRING')],
                  {'role':role,'cap_megapixels':2.5,'crop_left_percent':0.,'crop_top_percent':0.,'crop_right_percent':0.,'crop_bottom_percent':0.},(380,230),i>3)
            g.link(i,0,100+i,'reference','H3_STILL_REFERENCE')
            if i<=3:
                g.add(200+i,'PreviewImage','CHECK ACTUAL CROP - Picture %s'%i,(x,y+580),size=(380,220))
                g.link(100+i,1,200+i,'images','IMAGE')


def base_models(g,plan=10):
    g.add(11,'H3StudioMediaModel','Existing INT8 model selection',(2200,0),[('MODEL','MODEL')],size=(450,90));g.link(plan,0,11,'plan','H3_STILL_PLAN')
    for i,y,title in ((12,170,'Personal identity LoRA'),(13,380,'Personal style LoRA')):
        g.add(i,'H3StudioMediaLora',title+' - OFF',(2200,y),[('MODEL','MODEL')],{'enabled':False,'lora_name':'(none)','strength':1.},(450,150));g.link(i-1,0,i,'model','MODEL')
    g.add(14,'CLIPLoader','Existing required H3 encoder',(2200,600),[('CLIP','CLIP')],{'clip_name':ENC,'type':'minimax','device':'default'},(450,130))
    g.add(15,'VAELoader','Existing H3 video VAE',(2200,800),[('VAE','VAE')],{'vae_name':VAE},(450,90))


def render_chain(g,plan=10,conditioning=16,prep_slot=1,experiment='None',swap=False,label='baseline'):
    g.add(17,'FizgigH3StillLatent','True one-frame output',(2800,750),[('LATENT','LATENT')],{'width':1344,'height':768,'batch_size':1},(350,130))
    g.link(plan,1,17,'width','INT',True);g.link(plan,2,17,'height','INT',True)
    g.add(18,'RandomNoise','Fixed seed for A/B tests',(2800,0),[('NOISE','NOISE')],{'noise_seed':18071042,'control_after_generate':'fixed'},(350,110))
    g.add(19,'H3StudioMediaSampling','Baseline sampling / correct Turbo',(2800,180),[('MODEL','MODEL'),('SAMPLER','SAMPLER'),('SIGMAS','SIGMAS')],size=(350,130))
    g.link(13,0,19,'model','MODEL');g.link(plan,0,19,'plan','H3_STILL_PLAN')
    g.add(20,'H3LabExperiment','EXPERIMENT - test one change at a time',(3200,180),[('MODEL','MODEL'),('report','STRING')],
        {'character_swap_adapter':swap,'character_swap_strength':1.,'accelerator':experiment,'allow_combined_experiment':False},(480,230))
    g.link(19,0,20,'model','MODEL');g.link(plan,0,20,'plan','H3_STILL_PLAN')
    g.add(21,'BasicGuider','Conditioning + experimental model',(3200,490),[('GUIDER','GUIDER')],size=(450,100))
    g.link(20,0,21,'model','MODEL');g.link(conditioning,0,21,'conditioning','CONDITIONING')
    g.add(22,'H3LabTimedSampler','SAMPLE - change run_label to repeat cached test',(3800,0),[('LATENT','LATENT'),('report','STRING')],
          {'preparation_report':'','experiment_report':'','run_label':label},(510,230))
    for s,slot,name,typ,widget in [(18,0,'noise','NOISE',False),(21,0,'guider','GUIDER',False),(19,1,'sampler','SAMPLER',False),(19,2,'sigmas','SIGMAS',False),(17,0,'latent_image','LATENT',False),(conditioning,prep_slot,'preparation_report','STRING',True),(20,1,'experiment_report','STRING',True)]:g.link(s,slot,22,name,typ,widget)
    g.add(23,'H3LabTimedDecode','Fizgig decode + timing',(3800,330),[('IMAGE','IMAGE'),('report','STRING')],{'report':'','decoder':'Fizgig'},(510,130))
    g.link(22,0,23,'samples','LATENT');g.link(15,0,23,'vae','VAE');g.link(22,1,23,'report','STRING',True)
    g.add(40,'SaveImage','A. RAW H3 - inspect before compositing',(4420,0),widgets={'filename_prefix':'H3_Lab/'+label+'/Raw'},size=(540,470));g.link(23,0,40,'images','IMAGE')
    g.add(41,'H3LabReport','PERFORMANCE / ERRORS / CONFIGURATION',(4420,1100),widgets={'text':'','save_report':True},size=(620,420));g.link(23,1,41,'text','STRING',True)


def studio(name,replace=False,sam=False,experiment='None',swap=False,sheet=False):
    g=Graph(name);references(g,replace)
    g.add(30,'H3StudioOptionalVideo','Optional short video - OFF initially',(1400,0),[('video','H3_STILL_VIDEO')],
          {'file':'(none)','enabled':False,'start_seconds':0.,'duration_seconds':3.1,'video_cap_megapixels':.5,'upload':'video'},(520,230))
    g.add(10,'H3LabPlan','1. MODE / ONE OUTPUT MP CONTROL',(1400,340),[('plan','H3_STILL_PLAN'),('width','INT'),('height','INT')],
          {'mode':'Replace person' if replace else 'References' if sheet else 'Auto','aspect_ratio':'2:3',
           'output_megapixels':4. if sheet else 2.5,'preset':QUALITY,'manual_quality_steps':0,'reference_size_mode':'match','protect_background':sam},(640,300))
    for i in range(1,10):g.link(31 if replace and i==1 else 100+i,0,10,'reference_%s'%i,'H3_STILL_REFERENCE')
    g.link(30,0,10,'video','H3_STILL_VIDEO')
    g.add(32,'PrimitiveStringMultiline','SELECT ONE PERSON - position + short description',(1400,780),[('STRING','STRING')],{'value':'person on the left'},(600,110))
    request=('Create a four-panel photographic character sheet of the same person in the references, showing front, three-quarter, side and back views. Preserve identity and natural proportions. These are generated interpretations, not source photographs.' if sheet else
             'Create one natural, photorealistic still image.' if replace else 'A natural photograph of a ceramic cup beside a window in soft light.')
    g.add(33,'H3LabPrompt','2. DIRECT PROMPT / OPTIONAL ROLE BUILDER',(1400,990),[('STRING','STRING')],
          {'use_role_builder':replace,'target_description':'person on the left','face_slot':2 if replace else 0,'body_slot':3 if replace else 0,
           'preserve_other_people':replace,'match_lighting':False,'request':request},(640,340))
    g.link(10,0,33,'plan','H3_STILL_PLAN');g.link(32,0,33,'target_description','STRING',True)
    g.add(34,'H3LabReport','READ THE EXACT PROMPT BEFORE RENDERING',(1400,1430),widgets={'text':'','save_report':False},size=(640,260));g.link(33,0,34,'text','STRING',True)
    base_models(g)
    g.add(16,'H3LabConditioning','Native reference conditioning - no enhancer',(2200,1100),[('CONDITIONING','CONDITIONING'),('report','STRING')],{'prompt':''},(450,140))
    g.link(10,0,16,'plan','H3_STILL_PLAN');g.link(14,0,16,'clip','CLIP');g.link(15,0,16,'vae','VAE');g.link(33,0,16,'prompt','STRING',True)
    label='swap' if swap else 'spectrum' if experiment=='Spectrum' else 'spark' if experiment=='Spark SM86' else 'sheet' if sheet else 'sam' if sam else 'replacement' if replace else 'studio'
    render_chain(g,experiment=experiment,swap=swap,label=label)
    g.add(28,'H3StudioMediaFinish','Final original-pixel protection (not inpainting)',(3800,590),[('IMAGE','IMAGE')],size=(510,110))
    g.link(10,0,28,'plan','H3_STILL_PLAN');g.link(23,0,28,'images','IMAGE')
    g.add(24,'SaveImage','B. FINAL - protected only when mask option is ON',(4420,560),widgets={'filename_prefix':'H3_Lab/'+label+'/Final'},size=(540,470));g.link(28,0,24,'images','IMAGE')
    g.add(25,'H3StudioMediaCompare','Target left / result right',(3800,850),widgets={'save_comparison':True},size=(510,240));g.link(10,0,25,'plan','H3_STILL_PLAN');g.link(28,0,25,'images','IMAGE')
    if sam:
        # Replace only the target's existing mask connection. Never alter the original RGB image.
        old=next(p for p in g.nodes[31]['inputs'] if p['name']=='mask');oldlink=old['link']
        g.links=[e for e in g.links if e[0]!=oldlink];g.nodes[1]['outputs'][1]['links'].remove(oldlink)
        # Use fresh monotonically increasing link IDs after deletion.
        # Restore slot then rewire by editing the old link to retain all IDs.
        g.add(50,'CheckpointLoaderSimple','SAM 3.1 - download sam once',(0,2500),[('MODEL','MODEL'),('CLIP','CLIP'),('VAE','VAE')],{'ckpt_name':'sam3.1_multiplex_fp16.safetensors'},(530,110))
        g.add(51,'CLIPTextEncode','SAM text uses the same target description',(620,2500),[('CONDITIONING','CONDITIONING')],{'text':'person on the left'},(500,150))
        # Reinsert preserved edge id before adding any other link.
        g.add(52,'SAM3_Detect','Find ONE selected person',(1190,2500),[('masks','MASK'),('bboxes','BOUNDING_BOX')],{'threshold':.5,'refine_iterations':2,'individual_masks':False},(500,150))
        g.add(53,'GrowMask','MASK MARGIN - 0 protects nearby people',(1770,2500),[('MASK','MASK')],{'expand':0,'tapered_corners':True},(420,110))
        g.links.append([oldlink,53,0,31,1,'MASK']);g.nodes[53]['outputs'][0]['links'].append(oldlink)
        g.link(50,0,52,'model','MODEL');g.link(1,0,52,'image','IMAGE');g.link(50,1,51,'clip','CLIP');g.link(32,0,51,'text','STRING',True);g.link(51,0,52,'conditioning','CONDITIONING');g.link(52,0,53,'mask','MASK')
        g.add(54,'MaskPreview','CHECK MASK: white only on target person',(2300,2500),[('MASK','MASK')],size=(520,400));g.link(53,0,54,'mask','MASK')
    notes='H3 STUDIO LAB. Existing models and login remain unchanged. No model downloads happen when opening this graph.\n'
    notes+='First test the unchanged 00 baseline. Keep the same seed, photographs and prompt for A/B tests. Change run_label to force another timed sample; preparation may be cached.\n'
    notes+='One output_megapixels value controls the selected mode, including Replace person. Supporting photos have their own crop and MP nodes. Previews are before the native match/max reference resize; the report shows final reference dimensions.\n'
    notes+='Role builder is literal text assembly, not an LLM. Turn it OFF to paste your successful prompt unchanged. Face/body roles are instructions, not hard identity locks.\n'
    notes+='All optional personal LoRAs are OFF. Experimental graph presets request exactly one experiment. Spectrum and Spark change the numerical trajectory; do not assume a speedup or likeness improvement.\n'
    if sam:notes+='SAM uses one short target description. Run 12_SAM_Mask_Only first. Inspect overlap boundaries; pixels outside the chosen mask are preserved only by final compositing.\n'
    if experiment=='Spark SM86':notes+='A40/SM86 test only. Run manage.py setup spark first. Do not use on A100/SM80.\n'
    if swap:notes+='Run manage.py download swap once. The adapter is video-trained; single-frame quality is experimental.\n'
    if sheet:notes+='Generated turnaround views may invent unseen anatomy. Use original photographs as the identity authority.\n'
    g.add(90,'Note','START HERE',(0,3300),widgets={'text':notes},size=(2050,360))
    return g.finish()


def contact(name):
    g=Graph(name);references(g)
    g.add(10,'H3LabContactSheet','Arrange originals - no new views invented',(1500,0),[('IMAGE','IMAGE')],{'columns':3,'cell_size':512},(400,120))
    for i in range(1,10):g.link(100+i,0,10,'reference_%s'%i,'H3_STILL_REFERENCE')
    g.add(24,'SaveImage','Original-reference contact sheet',(2000,0),widgets={'filename_prefix':'H3_Lab/Original_Contact_Sheet'},size=(900,700));g.link(10,0,24,'images','IMAGE')
    return g.finish()


def refmod_save(name):
    g=Graph(name);references(g,count=1)
    g.add(15,'VAELoader','Existing H3 VAE only',(600,0),[('VAE','VAE')],{'vae_name':VAE},(500,95))
    g.add(10,'H3LabRefModSave','SAVE ONE SOURCE PHOTO - use a new version name',(1200,0),[('details','STRING')],{'name':'person_face_v1','mode':'Full reference','short_edge':1024},(650,250))
    g.link(101,0,10,'reference','H3_STILL_REFERENCE');g.link(15,0,10,'vae','VAE')
    g.add(90,'Note','Save face and body separately',(600,450),widgets={'text':'Full reference is the default. This encodes a source reference; it is not LoRA training. Save a face photo as person_face_v1, then a body photo as person_body_v1. Existing names are never overwritten. Compressed experimental pools information and can reduce identity fidelity. Use 10_Generate_From_RefMods to load the files.'},size=(1000,280))
    return g.finish()


def refmod_generate(name):
    g=Graph(name)
    g.add(1,'H3LabRefModLoad','Choose saved face / optional body references',(0,0),[('mods','H3_REF_MODS'),('mapping','STRING')],{'face_refmod':'(none)','body_refmod':'(none)','face_strength':1.,'body_strength':1.},(600,200))
    g.add(10,'H3LabRefModPlan','Reference generation from saved photographs',(800,0),[('plan','H3_STILL_PLAN'),('width','INT'),('height','INT')],{'aspect_ratio':'2:3','output_megapixels':2.5},(450,160));g.link(1,0,10,'mods','H3_REF_MODS')
    base_models(g)
    g.add(16,'H3LabRefModConditioning','Direct prompt using saved Picture tags',(800,350),[('CONDITIONING','CONDITIONING'),('report','STRING')],{'prompt':'Create one natural studio photograph of the person in <Picture 1>. Preserve their facial identity and natural proportions.'},(1000,250))
    for s,slot,key,typ in [(1,0,'mods','H3_REF_MODS'),(10,0,'plan','H3_STILL_PLAN'),(14,0,'clip','CLIP'),(15,0,'vae','VAE')]:g.link(s,slot,16,key,typ)
    g.add(34,'H3LabReport','Actual saved-reference ordering',(0,350),widgets={'text':'','save_report':False},size=(650,250));g.link(1,1,34,'text','STRING',True)
    render_chain(g,label='refmods')
    g.add(90,'Note','READ FIRST',(0,900),widgets={'text':'Save originals with 09_Save_RefMod first, then refresh model definitions. Face/body files become Picture 1 / Picture 2 in active order. With only one file, remove Picture 2 references from your prompt. The saved latent is decoded for the vision encoder, so neither identical output nor a speed improvement is guaranteed. Do not append these same mods a second time with an Apply node.'},size=(1700,300))
    return g.finish()


def ph3(name):
    g=Graph(name);references(g,count=8)
    g.add(10,'H3LabPH3','P-H3 INT8 - separate build + optional download',(1450,0),[('MODEL','MODEL'),('VAE','VAE'),('CONDITIONING','CONDITIONING'),('LATENT','LATENT'),('SAMPLER','SAMPLER'),('SIGMAS','SIGMAS'),('report','STRING')],
          {'prompt':'Create one natural photographic portrait of the person in the reference images. Preserve their identity and natural proportions.','aspect_ratio':'2:3','output_megapixels':4.,'follow_image_1_aspect':True},(740,450))
    for i in range(1,9):g.link(100+i,0,10,'reference_%s'%i,'H3_STILL_REFERENCE')
    g.add(18,'RandomNoise','Fixed seed',(2340,0),[('NOISE','NOISE')],{'noise_seed':18071042,'control_after_generate':'fixed'},(360,110))
    g.add(21,'BasicGuider','Native P-H3 conditioning',(2340,200),[('GUIDER','GUIDER')],size=(360,110));g.link(10,0,21,'model','MODEL');g.link(10,2,21,'conditioning','CONDITIONING')
    g.add(22,'H3LabTimedSampler','8-step Euler, exact P-H3 sigmas',(2800,0),[('LATENT','LATENT'),('report','STRING')],{'preparation_report':'','experiment_report':'{"accelerator":"None","profile":"P-H3 native"}','run_label':'ph3'},(540,220))
    for s,slot,key,typ,widget in [(18,0,'noise','NOISE',False),(21,0,'guider','GUIDER',False),(10,4,'sampler','SAMPLER',False),(10,5,'sigmas','SIGMAS',False),(10,3,'latent_image','LATENT',False),(10,6,'preparation_report','STRING',True)]:g.link(s,slot,22,key,typ,widget)
    g.add(23,'H3LabTimedDecode','P-H3 dedicated VAE - NOT Fizgig decode',(2800,400),[('IMAGE','IMAGE'),('report','STRING')],{'report':'','decoder':'Native P-H3'},(540,140));g.link(22,0,23,'samples','LATENT');g.link(10,1,23,'vae','VAE');g.link(22,1,23,'report','STRING',True)
    g.add(24,'SaveImage','P-H3 result',(3450,0),widgets={'filename_prefix':'H3_Lab/PH3_INT8'},size=(600,650));g.link(23,0,24,'images','IMAGE')
    g.add(41,'H3LabReport','P-H3 measured run report',(3450,800),widgets={'text':'','save_report':True},size=(620,420));g.link(23,1,41,'text','STRING',True)
    g.add(90,'Note','P-H3 is a separate native pipeline',(1400,1100),widgets={'text':'Build profile ph3 is required; stable deliberately retains the old ComfyUI core. Run manage.py download ph3 once (~29.5 GB). All three files are INT8. No prompt upsampler is downloaded. P-H3 treats Image 1 as an edit source; for replacement put scene in 1, face in 2, body in 3 and write those roles explicitly. Zero refs supports text generation. Eight image inputs maximum here. This comparison does not apply the old H3 Turbo LoRA or the old video VAE.'},size=(1800,350))
    return g.finish()


def build_all():
    result={}
    result['00_Known_Good_Replacement.json']=json.loads((P/'fixtures/known_good_replacement.json').read_text())
    for name,kwargs in [
        ('01_Studio_Generate_Edit_References.json',{}),
        ('02_Replace_One_Person.json',{'replace':True}),
        ('03_Replace_One_Person_SAM.json',{'replace':True,'sam':True}),
        ('04_Spectrum_Test.json',{'replace':True,'experiment':'Spectrum'}),
        ('05_Character_Swap_Test.json',{'replace':True,'swap':True}),
        ('06_Spark_A40_Test.json',{'replace':True,'experiment':'Spark SM86'}),
        ('08_Generated_Character_Sheet.json',{'sheet':True})]:result[name]=studio(name,**kwargs)
    result['07_Original_Reference_Contact_Sheet.json']=contact('07_Original_Reference_Contact_Sheet.json')
    result['09_Save_RefMod.json']=refmod_save('09_Save_RefMod.json')
    result['10_Generate_From_RefMods.json']=refmod_generate('10_Generate_From_RefMods.json')
    result['11_P_H3_INT8_Comparison.json']=ph3('11_P_H3_INT8_Comparison.json')
    result['12_SAM_Mask_Only.json']=json.loads((P/'fixtures/sam_mask_preview.json').read_text())
    return result

def main():
    folder=P/'workflows';folder.mkdir(exist_ok=True)
    for name,graph in build_all().items():(folder/name).write_text(json.dumps(graph,indent=2)+'\n')
    print('Generated',len(build_all()),'Lab workflows.')
if __name__=='__main__':main()
