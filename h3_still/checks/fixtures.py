"""Small synthetic graphs for offline regression tests, not shipped inference presets."""
from collections import OrderedDict


def graph(edit=False, hq=False):
    nodes = []
    edges = []
    def node(id, kind, named=None, inputs=(), outputs=()):
        named = OrderedDict(named or {})
        result = {'id': id, 'type': kind, 'inputs': [{'name': x, 'link': None} for x in inputs],
                  'outputs': [{'name': x, 'type': x, 'links': []} for x in outputs],
                  'widgets_values_named': named, 'widgets_values': list(named.values()),
                  'properties': {}, 'mode': 0, 'pos':[0,0]}
        nodes.append(result)
        return result
    unet = node(1, 'UNETLoader', {'unet_name':'minimax_h3_fl2va_pruned_int8_convrot.safetensors','weight_dtype':'default'}, outputs=['MODEL'])
    clip = node(3, 'CLIPLoader', {'clip_name':'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors','type':'minimax','device':'default'},outputs=['CLIP'])
    vae = node(4, 'VAELoader', {'vae_name':'minimax_h3_video_vae_int8_convrot.safetensors'}, outputs=['VAE'])
    lora = node(25, 'LoraLoaderModelOnly', {'lora_name':'minimax\\minimax_h3_turbo_v4_step600_ema.safetensors','strength_model':0 if hq else .38},inputs=['model'],outputs=['MODEL'])
    noise = node(9, 'RandomNoise', {'noise_seed':123,'control_after_generate':'fixed'},outputs=['NOISE'])
    choice = node(11, 'KSamplerSelect', {'sampler_name':'er_sde'},outputs=['SAMPLER'])
    scheduler = node(12,'BasicScheduler', {'scheduler':'simple','steps':50 if hq else 20,'denoise':1}, inputs=['model'],outputs=['SIGMAS'])
    guider = node(10, 'BasicGuider', inputs=['model','conditioning'],outputs=['GUIDER'])
    cond = node(29 if edit else 8, 'MiniMaxH3ReferenceToVideo' if edit else 'MiniMaxH3ImageToVideo',
                {'prompt':'<Picture 1> Change the dress to red.' if edit else 'A mountain lake.',
                 'width':1344,'height':768,'length':124 if edit else 5},
                inputs=['clip','vae','width','height']+(['ref_images.ref_image_0','ref_images.ref_image_1'] if edit else []),outputs=['CONDITIONING','LATENT'])
    latent = node(19, 'FizgigH3StillLatent', {'width':1344,'height':768,'batch_size':1},inputs=['width','height'],outputs=['LATENT'])
    sampler = node(13,'SamplerCustomAdvanced',inputs=['noise','guider','sampler','sigmas','latent_image'],outputs=['LATENT'])
    decode = node(14,'FizgigH3StillDecode',inputs=['samples','vae'],outputs=['IMAGE'])
    save = node(20,'SaveImage', {'filename_prefix':'Fizgig_H3_Still'}, inputs=['images'])
    ordinary = node(23,'VAEDecode',inputs=['samples','vae'],outputs=['IMAGE'])
    preview = node(27,'PreviewImage',inputs=['images'])
    def link(a, out, b, name):
        id = len(edges)+1
        target = next(i for i,p in enumerate(b['inputs']) if p['name']==name)
        b['inputs'][target]['link']=id
        a['outputs'][out]['links'].append(id)
        edges.append([id,a['id'],out,b['id'],target,a['outputs'][out]['type']])
    if edit:
        image = node(30,'LoadImage',{'image':'user-must-upload.jpg','upload':'image'},outputs=['IMAGE','MASK'])
        resize = node(34,'AILab_ImageResize',{'custom_width':0,'custom_height':0,'megapixels':2,'scale_by':1,'resize_mode':'longest_side','resize_value':0,'upscale_method':'lanczos','device':'cpu','divisible_by':32,'output_mode':'stretch','crop_position':'center','pad_color':'#FFFFFF'},inputs=['image'],outputs=['IMAGE'])
        size = node(31,'GetImageSize',inputs=['image'],outputs=['INT','INT'])
        link(image,0,resize,'image');link(resize,0,size,'image');link(resize,0,cond,'ref_images.ref_image_0')
    else:
        size=node(7,'ResolutionSelector',{'aspect_ratio':'16:9 (Widescreen)','megapixels':8 if hq else 2.5,'multiple':16 if hq else 32},outputs=['INT','INT'])
    link(size,0,latent,'width');link(size,1,latent,'height')
    link(size,0,cond,'width');link(size,1,cond,'height')
    link(unet,0,lora,'model');link(lora,0,scheduler,'model');link(lora,0,guider,'model')
    link(clip,0,cond,'clip');link(vae,0,cond,'vae');link(cond,0,guider,'conditioning')
    link(noise,0,sampler,'noise');link(choice,0,sampler,'sampler');link(scheduler,0,sampler,'sigmas');link(guider,0,sampler,'guider');link(latent,0,sampler,'latent_image')
    link(sampler,0,decode,'samples');link(vae,0,decode,'vae');link(decode,0,save,'images')
    link(sampler,0,ordinary,'samples');link(vae,0,ordinary,'vae');link(ordinary,0,preview,'images')
    return {'id':'developer-fixture','version':0.4,'nodes':nodes,'links':edges,'extra':{}}
