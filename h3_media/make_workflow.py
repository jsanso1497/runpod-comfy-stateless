#!/usr/bin/env python3
"""Generate the additive HQ media graph without altering existing H3 graphs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('_h3_media_builder_nodes',HERE/'node/__init__.py',submodule_search_locations=[str(HERE/'node')])
local=importlib.util.module_from_spec(spec);sys.modules[spec.name]=local;spec.loader.exec_module(local)
PROTOS={}
for f in ('h3_portrait/workflows/H3_Ref2VA_Standard.json','h3_portrait/workflows/H3_Portrait_Auto.json','config/workflows/00_SeedVR2_4K_Video_Restore.json'):
    for n in json.loads((ROOT/f).read_text())['nodes']:PROTOS.setdefault(n['type'],n)
NAMES={'LoadImage':['image','upload'],'LoadVideo':['file','upload'],'UNETLoader':['unet_name','weight_dtype'],
       'CLIPLoader':['clip_name','type','device'],'VAELoader':['vae_name'],'RandomNoise':['noise_seed','control_after_generate'],
       'KSamplerSelect':['sampler_name'],'BasicScheduler':['scheduler','steps','denoise'],
       'CreateVideo':['fps','bit_depth','color_space','codec']}
# CreateVideo names must come from the actual source graph when available.

class Graph:
    def __init__(self):self.nodes=[];self.links=[]
    def add(self,kind,title,x,y,values=None):
        if kind in local.NODE_CLASS_MAPPINGS:
            cls=local.NODE_CLASS_MAPPINGS[kind]
            n={'inputs':[],'outputs':[],'widgets_values_named':{}}
            for group in ('required','optional'):
                for name,defn in cls.INPUT_TYPES().get(group,{}).items():
                    typ=defn[0];opts=defn[1] if len(defn)>1 else {}
                    if isinstance(typ,list) or typ in ('STRING','INT','FLOAT','BOOLEAN'):
                        n['widgets_values_named'][name]=opts.get('default',typ[0] if isinstance(typ,list) else '')
                    else:n['inputs'].append({'name':name,'type':typ,'link':None})
            for name,typ in zip(getattr(cls,'RETURN_NAMES',cls.RETURN_TYPES),cls.RETURN_TYPES):
                n['outputs'].append({'name':name,'type':typ,'links':[]})
        else:
            n=deepcopy(PROTOS[kind])
            if 'widgets_values_named' not in n and kind in NAMES:
                n['widgets_values_named']=dict(zip(NAMES[kind],n.get('widgets_values',[])))
        if kind=='LoadVideo':
            n['inputs']=[i for i in n['inputs'] if i['name']!='upload']
        n.update(id=len(self.nodes)+1,type=kind,title=title,pos=[x,y],size=[360,120],flags={},mode=0,properties={'Node name for S&R':kind})
        for i in n.get('inputs',[]):i['link']=None
        for o in n.get('outputs',[]):o['links']=[]
        if values:
            if kind=='Note':n['widgets_values']=[values['text']]
            else:n.setdefault('widgets_values_named',{}).update(values)
        if 'widgets_values_named' in n:
            n['widgets_values']=list(n['widgets_values_named'].values())
            n['size'][1]=max(120,80+38*len(n['widgets_values']))
        self.nodes.append(n);return n
    def link(self,a,out,b,name):
        names=[i['name'] for i in b['inputs']];typ=a['outputs'][out]['type']
        if name not in names:
            b['inputs'].append({'name':name,'type':typ,'link':None,'widget':{'name':name}});names.append(name)
        idx=names.index(name);lid=len(self.links)+1
        if b['inputs'][idx]['link'] is not None:raise ValueError('Duplicate input')
        b['inputs'][idx].update(type=typ,link=lid);a['outputs'][out]['links'].append(lid)
        self.links.append([lid,a['id'],out,b['id'],idx,typ])
    def save(self):
        done=set()
        while len(done)<len(self.nodes):
            avail=[n for n in self.nodes if n['id'] not in done and all(l[1] in done for l in self.links if l[3]==n['id'])]
            if not avail:raise ValueError('Cycle')
            for n in avail:n['order']=len(done);done.add(n['id'])
        data={'last_node_id':len(self.nodes),'last_link_id':len(self.links),'nodes':self.nodes,'links':self.links,'groups':[],
              'config':{},'version':0.4,'extra':{'ds':{'scale':.4,'offset':[1480,850]},
               'h3_media':{'version':'1.2.0','recipe':'hq_mp4_voice_identity_action','comfy_ref':'65787d668397d230bf5839d69a0a7239e2dad378'}}}
        (HERE/'workflows/H3_HQ_MP4_Voice_Identity_Action_v1_2.json').write_text(json.dumps(data,indent=2)+'\n');return data

def make():
    g=Graph()
    face=g.add('LoadImage','REFERENCE A - primary face',-1500,-450);face['size']=[350,400]
    body=g.add('LoadImage','REFERENCE B - body / another view',-1500,20);body['size']=[350,400]
    clip=g.add('LoadVideo','UPLOAD MP4 - choose a short clear voice section',-1500,500);clip['size']=[350,350]
    a=g.add('H3MediaImageReference','A: face identity',-1080,-450)
    b=g.add('H3MediaImageReference','B: body / complementary detail',-1080,30,{'label':'Body / second view','use_for':'Physique, body proportions and hair of the same main subject. Use reference A for facial identity. Do not copy this photograph\'s pose or background.'})
    v=g.add('H3MediaVideoReference','MP4 roles - defaults to VOICE ONLY',-1080,520);v['size']=[440,580]
    g.link(face,0,a,'image');g.link(body,0,b,'image');g.link(a,0,b,'references');g.link(clip,0,v,'video');g.link(b,0,v,'references')
    plan=g.add('H3MediaPlan','START HERE - describe scene and type NEW words',-560,-430);plan['size']=[640,780]
    g.link(v,0,plan,'references')
    preview=g.add('PreviewAny','Inspect actual Picture / Video / Audio mapping',-560,420);preview['size']=[640,650];g.link(plan,1,preview,'source')
    unet=g.add('UNETLoader','FULL native H3 Ref2VA BF16 (not Turbo)',200,-850)
    encoder=g.add('CLIPLoader','Qwen3-VL 32B BF16',200,-580)
    vae=g.add('VAELoader','H3 video VAE FP16',200,-330,{'vae_name':'minimax_h3_video_vae_fp16.safetensors'})
    avae=g.add('VAELoader','H3 audio VAE FP32',200,-80,{'vae_name':'minimax_h3_audio_vae_fp32.safetensors'})
    cond=g.add('H3MediaConditioning','Native H3 visual + audio references',640,-460)
    for source,out,name in [(plan,0,'ready_plan'),(encoder,0,'clip'),(vae,0,'vae'),(avae,0,'audio_vae')]:g.link(source,out,cond,name)
    noise=g.add('RandomNoise','Fixed seed for identity / voice comparisons',640,-160)
    guider=g.add('BasicGuider','Native positive conditioning',640,90);g.link(unet,0,guider,'model');g.link(cond,0,guider,'conditioning')
    sampler=g.add('KSamplerSelect','res_multistep',1060,-740)
    schedule=g.add('BasicScheduler','25 steps / beta / full denoise',1060,-500,{'scheduler':'beta','steps':25,'denoise':1.0});g.link(unet,0,schedule,'model')
    sample=g.add('SamplerCustomAdvanced','Generate ONE new video with NEW audio',1060,-170)
    for source,out,name in [(noise,0,'noise'),(guider,0,'guider'),(sampler,0,'sampler'),(schedule,0,'sigmas'),(cond,1,'latent_image')]:g.link(source,out,sample,name)
    dec=g.add('VAEDecode','Decode generated video',1490,-440);g.link(sample,0,dec,'samples');g.link(vae,0,dec,'vae')
    adec=g.add('VAEDecodeAudio','Decode GENERATED H3 audio (not reference track)',1490,-170);g.link(sample,0,adec,'samples');g.link(avae,0,adec,'vae')
    video=g.add('CreateVideo','24 fps with generated audio',1900,-460);g.link(dec,0,video,'images');g.link(adec,0,video,'audio')
    save=g.add('H3MediaExportVideo','Save HQ MP4 / CRF 18',2320,-460,{'filename_prefix':'H3_HQ_Media/video','crf':18});g.link(video,0,save,'video')
    final=g.add('H3MediaSaveLastFrame','Save matching last frame for chaining',2750,-460);g.link(save,0,final,'completed_export')
    note=g.add('Note','Reference controls / limits',-1500,1200,{'text':'Upload two original identity photos and one MP4. Voice only is the safe default: no reference video pixels are decoded or sent. Trim a clear 3-8 second section, then enter NEW dialogue in START HERE. For visual identity select Identity - selected frame and choose its relative position. For action select Action / motion; Identity + action provides both. Audio role is independent, so Ignore audio can be used for silent motion guides. To add more clips duplicate LoadVideo + MP4 role node and chain references. Maximum 3 source clips, 9 total still refs, 3 motion and 3 standalone audio refs. Native LoadAudio plus H3 HQ Standalone Audio Reference also supports existing audio files. Generated audio, not the source track, is saved. Voice likeness and motion following are not exact cloning or locked choreography. Large full BF16 models and max reference size are memory-intensive. No cloud/API inference. Use voices/images you have permission to use. This is an additive HQ variant; old workflows are not rewritten.'});note['size']=[1560,390]
    return g.save()
if __name__=='__main__':make()
