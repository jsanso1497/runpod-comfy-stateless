"""Deterministic editable graph for zero, one, or many optional still references."""
from pathlib import Path
import json
import uuid

PACKAGE = Path(__file__).resolve().parent
FILENAME = '00_H3_Studio_Auto_INT8.json'


def build():
    graph = {'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'h3-studio-int8-2026-11-06.1')),
             'revision': 0, 'last_node_id': 0, 'last_link_id': 0, 'nodes': [], 'links': [],
             'groups': [], 'config': {}, 'extra': {'ds': {'scale': .43, 'offset': [30, 60]},
             'h3_studio': {'release': '2026-11-06.1', 'source_tested': True,
                           'gpu_tested': False, 'prompt_enhancer': False,
                           'requires_rebuild': True, 'reference_limit': 9}}, 'version': .4}
    nodes = {}
    def add(id, kind, title, x, y, outputs=(), widgets=None, size=(310, 140), collapsed=False, core=False):
        node = {'id': id, 'type': kind, 'title': title, 'pos': [x,y], 'size': list(size),
                'flags': {'collapsed': True} if collapsed else {}, 'order': len(nodes), 'mode': 0,
                'inputs': [], 'outputs': [{'name': n, 'type': t, 'links': []} for n,t in outputs],
                'properties': {'Node name for S&R': kind},
                'widgets_values': list((widgets or {}).values()), 'widgets_values_named': widgets or {}}
        if core: node['properties'].update({'cnr_id': 'comfy-core', 'ver': '0.43.0'})
        if kind.startswith('Fizgig'): node['properties']['cnr_id'] = 'comfyui-fizgig-h3-still'
        nodes[id] = node; graph['nodes'].append(node)
        return node
    def connect(source, out, target, name, type, widget=False, optional=False):
        a,b = nodes[source], nodes[target]
        idx = len(b['inputs']); lid = len(graph['links'])+1
        port = {'name': name, 'type': type, 'link': lid}
        if widget: port['widget']={'name': name}
        if optional: port['shape']=7
        b['inputs'].append(port); a['outputs'][out]['links'].append(lid)
        graph['links'].append([lid, source, out, target, idx, type])
    for i in range(1,10):
        row=(i-1)//3; col=(i-1)%3
        add(i,'H3StudioOptionalImage','Reference slot %s - optional'%i,col*340, row*390,
            [('reference','H3_STILL_REFERENCE')], {'image':'(none)','enabled':True,'upload':'image'},
            size=(300,320), collapsed=i>3)
    add(10,'H3StudioPlan','1. MODE + SIZE (Auto uses photo count)',1100,0,
        [('plan','H3_STILL_PLAN'),('width','INT'),('height','INT'),('steps','INT')],
        {'mode':'Auto','aspect_ratio':'1:1','generation_megapixels':4.,'edit_megapixels':2.5,
         'preset':'Quality - 50 steps, no Turbo','reference_cap_megapixels':2.5}, size=(500,370))
    for i in range(1,10): connect(i,0,10,'reference_%s'%i,'H3_STILL_REFERENCE',optional=True)
    add(11,'H3StudioModel','INT8 model - selected automatically',1770,0,[('MODEL','MODEL')],size=(420,90))
    connect(10,0,11,'plan','H3_STILL_PLAN')
    for id,label,y in [(12,'2. Optional H3 identity LoRA (OFF)',170),(13,'3. Optional H3 style LoRA (OFF)',380)]:
        add(id,'H3StudioLora',label,1770,y,[('MODEL','MODEL')],
            {'enabled':False,'lora_name':'(none)','strength':1.},size=(420,160))
        connect(id-1,0,id,'model','MODEL')
    add(14,'CLIPLoader','Required H3 encoder - NOT a prompt enhancer',1770,620,[('CLIP','CLIP')],
        {'clip_name':'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','type':'minimax','device':'default'},size=(460,135),core=True)
    add(15,'VAELoader','INT8 video VAE',1770,810,[('VAE','VAE')],
        {'vae_name':'minimax_h3_video_vae_int8_convrot.safetensors'},size=(460,90),core=True)
    add(16,'H3StudioConditioning','4. YOUR PROMPT - no rewriting',1100,460,[('CONDITIONING','CONDITIONING')],
        {'prompt':'An editorial photograph of a ceramic cup beside a window. Soft morning light, natural materials, balanced exposure, restrained detail.'},size=(600,330))
    connect(10,0,16,'plan','H3_STILL_PLAN');connect(14,0,16,'clip','CLIP');connect(15,0,16,'vae','VAE')
    add(17,'FizgigH3StillLatent','True single frame - shared output dimensions',2380,650,[('LATENT','LATENT')],
        {'width':2048,'height':2048,'batch_size':1},size=(340,155))
    connect(10,1,17,'width','INT',widget=True);connect(10,2,17,'height','INT',widget=True)
    add(18,'RandomNoise','Seed - keep fixed while comparing changes',2380,0,[('NOISE','NOISE')],
        {'noise_seed':18071042,'control_after_generate':'fixed'},size=(350,100),core=True)
    add(19,'KSamplerSelect','Sampler - developer baseline',2380,150,[('SAMPLER','SAMPLER')],
        {'sampler_name':'er_sde'},core=True)
    add(20,'BasicScheduler','Schedule - steps from preset',2380,310,[('SIGMAS','SIGMAS')],
        {'scheduler':'simple','steps':50,'denoise':1.0},core=True)
    connect(13,0,20,'model','MODEL');connect(10,3,20,'steps','INT',widget=True)
    add(21,'BasicGuider','No CFG / negative prompt branch',2380,490,[('GUIDER','GUIDER')],size=(350,90),core=True)
    connect(13,0,21,'model','MODEL');connect(16,0,21,'conditioning','CONDITIONING')
    add(22,'SamplerCustomAdvanced','5. SAMPLE STILL',2880,0,[('output','LATENT'),('denoised_output','LATENT')],size=(320,300),core=True)
    for src,port,name,t in [(18,0,'noise','NOISE'),(21,0,'guider','GUIDER'),(19,0,'sampler','SAMPLER'),(20,0,'sigmas','SIGMAS'),(17,0,'latent_image','LATENT')]:
        connect(src,port,22,name,t)
    add(23,'FizgigH3StillDecode','Fizgig still decoder',2880,370,[('IMAGE','IMAGE')],size=(320,110))
    connect(22,0,23,'samples','LATENT');connect(15,0,23,'vae','VAE')
    add(24,'SaveImage','6. FINAL PNG',3320,0,widgets={'filename_prefix':'H3_Studio/Final'},size=(520,520),core=True)
    connect(23,0,24,'images','IMAGE')
    add(25,'H3StudioCompare','Edit comparison - source LEFT, edited RIGHT',3320,640,
        widgets={'save_comparison':True},size=(520,300))
    connect(10,0,25,'plan','H3_STILL_PLAN');connect(23,0,25,'images','IMAGE')
    add(26,'H3StudioAudit','Route + photo-tag report (after running)',1100,870,size=(600,360))
    connect(10,0,26,'plan','H3_STILL_PLAN')
    text = ('POWERED BY MINIMAX H3\n\n'
        'Auto: no active photo = Generate; one = Edit; two to nine = References.\n'
        'For a NEW scene using just one photo, select References instead of Auto.\n'
        'Upload photos in the slots, left to right. Expand slots 4-9 when needed.\n'
        'Leave unused slots at (none), or switch enabled OFF. No placeholder image is generated.\n'
        'Use <Picture 1>, <Picture 2>, etc. Active photos are numbered in slot order; see the report.\n'
        'Edit keeps the first photo aspect ratio at edit_megapixels. Other modes use generation size/aspect.\n'
        'Quality = 50 steps, no Turbo. Developer fast = 20 steps / Turbo 0.38, Generate or Edit only.\n'
        'Two personal H3 LoRA slots are independently OFF by default. Upload to /workspace/h3-still/models/loras.\n'
        'This is whole-image regeneration, not protected-mask inpainting or guaranteed restoration.\n'
        '4 MP generation is a starting point. 8 MP and nine references require GPU validation.\n'
        'Requires the rebuilt H3 Still Studio image. No BF16 base weights, no AI Lab nodes, no prompt enhancer.')
    add(27,'Note','START HERE',0,920,widgets={'text':text},size=(1000,380))
    for node in graph['nodes']:
        for port in node['outputs']:
            if not port['links']:port['links']=None
    graph['groups']=[{'title':'OPTIONAL PHOTOS - blank is valid','bounding':[ -20,-50,1030,930],'color':'#3f789e','font_size':24,'flags':{}},
                     {'title':'CONTROLS + PROMPT','bounding':[1080,-50,660,1320],'color':'#497e61','font_size':24,'flags':{}},
                     {'title':'QUANTIZED MODEL + OPTIONAL LORAS','bounding':[1750,-50,500,1030],'color':'#766393','font_size':24,'flags':{}}]
    graph['last_node_id']=max(nodes);graph['last_link_id']=len(graph['links'])
    return graph


def main():
    out=PACKAGE/'studio_workflows';out.mkdir(exist_ok=True)
    (out/FILENAME).write_text(json.dumps(build(),indent=2)+'\n',encoding='utf-8')
    print('Wrote',out/FILENAME)

if __name__=='__main__':main()
