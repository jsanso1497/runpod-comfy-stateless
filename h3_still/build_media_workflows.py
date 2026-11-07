"""Deterministic v2 workflow files; old Studio and working Turbo JSONs stay valid."""
from pathlib import Path
import json
import uuid

PACKAGE = Path(__file__).resolve().parent
FILES = ['00_H3_Studio_Photos_Video_INT8.json',
         '01_H3_Video_Reference_To_Still_Turbo.json',
         '02_H3_Replace_Person_Image1.json']
QUALITY = 'Quality - 50 steps, no Turbo'
TURBO = 'REF2VA Turbo - 8 steps'


class Graph:
    def __init__(self, filename):
        self.nodes = {}
        self.data = {'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'h3-studio-media-2026-11-07.2/' + filename)),
            'revision': 0, 'last_node_id': 0, 'last_link_id': 0, 'nodes': [], 'links': [],
            'groups': [], 'config': {}, 'extra': {'ds': {'scale': .4, 'offset': [50, 70]},
            'h3_media': {'release': '2026-11-07.2', 'gpu_tested': False, 'requires_rebuild': True,
                         'base_precision': 'INT8', 'prompt_enhancer': False}}, 'version': .4}

    def add(self, id, kind, title, x, y, outputs=(), widgets=None, size=(340,150), collapsed=False):
        node = dict(id=id, type=kind, title=title, pos=[x,y], size=list(size),
                    flags={'collapsed': True} if collapsed else {}, order=len(self.nodes), mode=0,
                    inputs=[], outputs=[{'name':name, 'type':type, 'links':[]} for name,type in outputs],
                    properties={'Node name for S&R':kind}, widgets_values=list((widgets or {}).values()),
                    widgets_values_named=widgets or {})
        if kind.startswith('Fizgig'):
            node['properties']['cnr_id']='comfyui-fizgig-h3-still'
        elif not kind.startswith('H3Studio') and kind!='Note':
            node['properties'].update(cnr_id='comfy-core',ver='0.43.0')
        self.nodes[id]=node
        self.data['nodes'].append(node)
        return node

    def link(self, source, slot, target, name, kind, widget=False):
        edge_id=len(self.data['links'])+1
        node=self.nodes[target]; index=len(node['inputs'])
        port=dict(name=name,type=kind,link=edge_id)
        if widget:
            port['widget']={'name':name}
        node['inputs'].append(port)
        self.nodes[source]['outputs'][slot]['links'].append(edge_id)
        self.data['links'].append([edge_id,source,slot,target,index,kind])

    def finish(self):
        for node in self.nodes.values():
            for port in node['outputs']:
                if not port['links']:
                    port['links']=None
        self.data['last_node_id']=max(self.nodes)
        self.data['last_link_id']=len(self.data['links'])
        return self.data


def build(profile=0):
    replace=profile==2; video_mode=profile==1
    g=Graph(FILES[profile])
    for i in range(1,10):
        x=((i-1)%3)*330; y=((i-1)//3)*370
        if replace and i==1:
            g.add(1,'LoadImage','IMAGE 1: TARGET SCENE - paint person mask',x,y,
                  [('IMAGE','IMAGE'),('MASK','MASK')],
                  {'image':'UPLOAD_TARGET_PHOTO.png','upload':'image'},size=(305,310))
            g.add(31,'H3StudioTargetPhoto','Target photo + person mask',0,1150,[('target_image_1','H3_STILL_REFERENCE')],size=(305,105))
            g.link(1,0,31,'image','IMAGE');g.link(1,1,31,'mask','MASK')
        else:
            title='Picture %s - %s'%(i,'replacement identity' if replace else 'optional photo')
            g.add(i,'H3StudioOptionalImage',title,x,y,[('reference','H3_STILL_REFERENCE')],
                  {'image':'(none)','enabled':True,'upload':'image'},size=(305,310),collapsed=i>3)
    g.add(30,'H3StudioOptionalVideo','OPTIONAL VIDEO - silent identity/body reference',1010,0,
          [('video_reference','H3_STILL_VIDEO')],
          {'file':'(none)','enabled':True,'start_seconds':0.,'duration_seconds':3.1,
           'video_cap_megapixels':.5,'upload':'video'},size=(470,230))
    g.add(10,'H3StudioMediaPlan','1. MODE, OUTPUT SIZE + SAMPLING PRESET',1560,0,
          [('plan','H3_STILL_PLAN'),('width','INT'),('height','INT')],
          {'mode':'Replace person' if replace else 'References' if video_mode else 'Auto',
           'aspect_ratio':'2:3','generation_megapixels':1.0 if video_mode else 2.5,
           'edit_megapixels':2.5,'preset':TURBO if video_mode else QUALITY,
           'reference_cap_megapixels':2.5,'protect_background':replace},size=(530,430))
    for i in range(1,10):g.link(31 if replace and i==1 else i,0,10,'reference_%s'%i,'H3_STILL_REFERENCE')
    g.link(30,0,10,'video','H3_STILL_VIDEO')
    g.add(11,'H3StudioMediaModel','Quantized model routing - unchanged INT8 stack',2190,0,[('MODEL','MODEL')],size=(470,100))
    g.link(10,0,11,'plan','H3_STILL_PLAN')
    for i,y,title in [(12,180,'Optional identity LoRA'),(13,390,'Optional style LoRA')]:
        g.add(i,'H3StudioMediaLora',title+' - OFF by default',2190,y,[('MODEL','MODEL')],
              {'enabled':False,'lora_name':'(none)','strength':1.},size=(470,160))
        g.link(i-1,0,i,'model','MODEL')
    g.add(14,'CLIPLoader','Required H3 encoder - no prompt enhancer',2190,630,[('CLIP','CLIP')],
          {'clip_name':'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','type':'minimax','device':'default'},size=(470,145))
    g.add(15,'VAELoader','Existing INT8 video VAE',2190,840,[('VAE','VAE')],
          {'vae_name':'minimax_h3_video_vae_int8_convrot.safetensors'},size=(470,95))
    if replace:
        prompt=('Replace the person in <Picture 1> with the person shown in the other references. '
                'Use <Picture 1> for the scene, camera angle, pose, clothing and lighting. '
                'Use the other photo and video references for the replacement person\'s face, hair, identity and body proportions. '
                'Make one realistic photograph. Keep the background and other objects unchanged.')
    elif video_mode:
        prompt=('Create one photorealistic studio photograph of the person in <Video 1>. '
                'Preserve their facial features, hair, physique and body proportions. '
                'Front-facing, neutral expression, relaxed arms, soft studio lighting and a neutral background.')
    else:
        prompt='A natural editorial photograph of a ceramic cup beside a window, soft morning light and restrained realistic detail.'
    g.add(16,'H3StudioMediaConditioning','2. PROMPT - <Picture 1> / <Video 1> tags',1530,540,[('CONDITIONING','CONDITIONING')],
          {'prompt':prompt},size=(600,340))
    g.link(10,0,16,'plan','H3_STILL_PLAN');g.link(14,0,16,'clip','CLIP');g.link(15,0,16,'vae','VAE')
    g.add(17,'FizgigH3StillLatent','ONE output frame - not a video render',2780,710,[('LATENT','LATENT')],
          {'width':1344,'height':768,'batch_size':1},size=(360,160))
    g.link(10,1,17,'width','INT',True);g.link(10,2,17,'height','INT',True)
    g.add(18,'RandomNoise','Seed - fixed for comparisons',2780,0,[('NOISE','NOISE')],
          {'noise_seed':18071042,'control_after_generate':'fixed'},size=(360,110))
    g.add(19,'H3StudioMediaSampling','3. Correct Turbo + sampler + sigma schedule',2780,230,
          [('sampling_model','MODEL'),('sampler','SAMPLER'),('sigmas','SIGMAS')],size=(360,150))
    g.link(13,0,19,'model','MODEL');g.link(10,0,19,'plan','H3_STILL_PLAN')
    g.add(21,'BasicGuider','Conditioning + correctly shifted model',2780,500,[('GUIDER','GUIDER')],size=(360,110))
    g.link(19,0,21,'model','MODEL');g.link(16,0,21,'conditioning','CONDITIONING')
    g.add(22,'SamplerCustomAdvanced','4. SAMPLE STILL',3260,0,[('output','LATENT'),('denoised_output','LATENT')],size=(330,300))
    for source,slot,name,kind in [(18,0,'noise','NOISE'),(21,0,'guider','GUIDER'),(19,1,'sampler','SAMPLER'),(19,2,'sigmas','SIGMAS'),(17,0,'latent_image','LATENT')]:
        g.link(source,slot,22,name,kind)
    g.add(23,'FizgigH3StillDecode','Existing Fizgig single-frame decode',3260,400,[('IMAGE','IMAGE')],size=(330,105))
    g.link(22,0,23,'samples','LATENT');g.link(15,0,23,'vae','VAE')
    g.add(28,'H3StudioMediaFinish','5. Optional original-pixel background protection',3260,620,[('IMAGE','IMAGE')],size=(390,130))
    g.link(10,0,28,'plan','H3_STILL_PLAN');g.link(23,0,28,'images','IMAGE')
    g.add(24,'SaveImage','FINAL PNG - separate from the source',3780,0,
          widgets={'filename_prefix':'H3_Studio_v2/'+('Person_Replacement' if replace else 'Video_Still' if video_mode else 'Final')},size=(540,520))
    g.link(28,0,24,'images','IMAGE')
    g.add(25,'H3StudioMediaCompare','Source LEFT / result RIGHT (edit or replacement)',3780,670,
          widgets={'save_comparison':True},size=(540,340))
    g.link(10,0,25,'plan','H3_STILL_PLAN');g.link(28,0,25,'images','IMAGE')
    g.add(26,'H3StudioAudit','Route, photo roles + actual video frames',1530,1010,size=(600,430))
    g.link(10,0,26,'plan','H3_STILL_PLAN')
    notes=('H3 STUDIO v2 - no new base models, no BF16 upgrade, no prompt enhancer.\n\n'
           'Auto: zero media = Generate; one photo = Edit; video or multiple photos = References.\n'
           'Select References explicitly to make a NEW image from one photo. Upload a video or leave (none).\n'
           'Video is trimmed, normalized to 24 fps and aligned to the H3 frame grid. Audio is ignored. Output is ONE PNG.\n'
           'Quality: 50 steps. REF2VA Turbo: 8 steps for References/Replace person. Developer FL2VA fast: Generate/Edit only.\n'
           'The existing REF2VA Turbo adapter is reused; if absent the new image downloads it once (1.96 GB).\n'
           'Picture tags follow active photo-slot order. Video is always <Video 1>. Start with a short clip and 1-2 photos.\n'
           'Legacy v1 workflows remain intact. No AI Lab color node or SAM is required.\n')
    if replace:
        notes += ('\nPERSON REPLACEMENT: Image 1 is the existing target photograph. Slots 2-9 / optional video define the replacement.\n'
                  'Right-click target Load Image > Open in MaskEditor. Paint the WHOLE original person (white = replace), then save.\n'
                  'protect_background ON requires that mask, resizes the generated candidate to the original canvas and composites it.\n'
                  'Unmasked original pixels stay unchanged. The mask does NOT guide H3 sampling; a drifting pose can need a wider mask or another seed.\n'
                  'For whole-image regeneration without painting, switch protect_background OFF. Background preservation is then not guaranteed.\n'
                  'Default prompt uses image 1 as the target and all other active references as replacement identity. Add more specific roles only when needed.')
    else:
        notes += '\nReplace the default text prompt when adding references. Do not refer to an empty Picture/Video slot.'
    g.add(27,'Note','READ FIRST',0,1350,widgets={'text':notes},size=(1460,390))
    g.data['groups']=[{'title':'PHOTOS - Image 1 is the target in Replace person mode','bounding':[-20,-55,1010,1350],'color':'#3f789e','font_size':24,'flags':{}},
                      {'title':'VIDEO + MEDIA CONTROLS','bounding':[990,-55,1160,1545],'color':'#497e61','font_size':24,'flags':{}},
                      {'title':'EXISTING MODELS + OPTIONAL LORAS','bounding':[2170,-55,520,1080],'color':'#766393','font_size':24,'flags':{}}]
    return g.finish()


def main():
    folder=PACKAGE/'media_workflows';folder.mkdir(exist_ok=True)
    for i, filename in enumerate(FILES):
        (folder/filename).write_text(json.dumps(build(i),indent=2)+'\n')
        print('Wrote',filename)

if __name__=='__main__':main()
