"""Generate the UI and API forms from the same graph definition."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent

def make(precision='BF16 quality'):
    nodes, links, api = [], [], {}
    def add(i, kind, title, pos, size, inputs=(), outputs=(), values=(), settings=None):
        n = {'id':i, 'type':kind, 'title':title, 'pos':list(pos), 'size':list(size),
             'flags':{}, 'order':len(nodes), 'mode':0,
             'inputs':[{'name':name, 'type':typ, 'link':None} for name,typ in inputs],
             'outputs':[{'name':name, 'type':typ, 'slot_index':j,'links':[]} for j,(name,typ) in enumerate(outputs)],
             'properties':{'Node name for S&R':kind}, 'widgets_values':list(values),
             'widgets_values_named':dict(settings or {})}
        nodes.append(n)
        if kind != 'Note': api[str(i)]={'class_type':kind, 'inputs':dict(settings or {}), '_meta':{'title':title}}
        return n
    def wire(source, output, target, input_name):
        a=next(n for n in nodes if n['id']==source);b=next(n for n in nodes if n['id']==target)
        slot=next(k for k,v in enumerate(b['inputs']) if v['name']==input_name)
        typ=a['outputs'][output]['type'];ident=len(links)+1
        links.append([ident,source,output,target,slot,typ]);a['outputs'][output]['links'].append(ident);b['inputs'][slot]['link']=ident
        api[str(target)]['inputs'][input_name]=[str(source), output]
    add(1,'LoadImage','1 | SOURCE PHOTO - paint/save manual mask here',(0,100),(360,420),
        outputs=[('IMAGE','IMAGE'),('MASK','MASK')],values=['example.png','image'],settings={'image':'example.png'})
    mask_settings={'auto_mask':False,'sam_prompt':'shirt sleeves:2','threshold':.5,'refine_iterations':2,'instance_index':-1,
                   'search_region':'full photo','manual_correction':'ignore','expand_pixels':0}
    add(2,'Q21PhotoMask','2 | AUTO MASK OFF = manual; ON = SAM 3.1',(430,100),(410,430),
        inputs=[('source','IMAGE'),('manual_mask','MASK')],outputs=[('effective_edit_mask','MASK'),('mask_report','STRING')],
        values=mask_settings.values(),settings=mask_settings)
    crop_settings={'resolution':2048,'max_long_side':3072,'context_pixels':128,'upscale_small_crops':False}
    add(3,'Q21PhotoCrop','3 | INPAINT CROP - 2K area budget; full source retained',(910,100),(420,250),
        inputs=[('source','IMAGE'),('mask','MASK'),('mask_report','STRING')],outputs=[('crop_data','Q21_PHOTO_CROP'),('working_crop','IMAGE')],
        values=crop_settings.values(),settings=crop_settings)
    model_settings={'precision':precision,'text_encoder_device':'default','cache_device':'auto'}
    add(4,'Q21PhotoModels','4 | MATCHING QWEN 2.1 MODELS + LOSSLESS CACHE',(910,430),(420,210),
        inputs=[('crop_data','Q21_PHOTO_CROP')],outputs=[('MODEL','MODEL'),('CLIP','CLIP'),('VAE','VAE')],
        values=model_settings.values(),settings=model_settings)
    lora_settings={}
    for j in range(1,4): lora_settings.update({f'lora_{j}':'None',f'strength_{j}':.6})
    add(5,'Q21PhotoLoRAs','5 | OPTIONAL QWEN 2.1 LoRAs - ALL OFF',(1410,430),(410,285),
        inputs=[('model','MODEL')],outputs=[('MODEL','MODEL')],values=lora_settings.values(),settings=lora_settings)
    for j in range(1,5):
        settings={'image':'None','role':'','resolution':1536}
        add(5+j,'Q21PhotoReference',f'REFERENCE {j} = <image{j+1}> | None = unused',((j-1)*470,850),(410,330),
            outputs=[('reference','Q21_PHOTO_REF')],values=['None','',1536,'image'],settings=settings)
    prompt='Edit <image1>. Replace this sentence with the exact change you want. Preserve the source pose, framing, camera perspective, lighting, and all untargeted features. Use each reference only for its stated role.'
    add(10,'Q21PhotoEncode','6 | YOUR PROMPT + UP TO 9 ORDERED REFERENCES',(1410,100),(570,280),
        inputs=[('clip','CLIP'),('vae','VAE'),('crop_data','Q21_PHOTO_CROP')]+[(f'reference_{j}','Q21_PHOTO_REF') for j in range(1,10)],
        outputs=[('positive','CONDITIONING'),('negative','CONDITIONING'),('latent','LATENT'),('reference_report','STRING')],
        values=[prompt],settings={'prompt':prompt})
    sampler={'seed':12345,'steps':40,'cfg':1.,'sampler_name':'euler','scheduler':'simple','denoise':1.}
    add(11,'KSampler','7 | NATIVE QWEN - 40 / CFG 1 / EULER / SIMPLE',(2050,100),(370,320),
        inputs=[('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT')],outputs=[('LATENT','LATENT')],
        values=[12345,'fixed',40,1.,'euler','simple',1.],settings=sampler)
    nodes[-1]['widgets_values_named']['control_after_generate']='fixed'
    add(12,'VAEDecode','8 | NATIVE VAE DECODE',(2050,500),(370,120),inputs=[('samples','LATENT'),('vae','VAE')],outputs=[('IMAGE','IMAGE')])
    stitch={'feather_pixels':12,'boundary_color_strength':0.}
    add(13,'Q21PhotoStitch','9 | PROTECTED STITCH - original-size output',(2490,100),(420,240),
        inputs=[('crop_data','Q21_PHOTO_CROP'),('generated_crop','IMAGE'),('reference_report','STRING')],outputs=[('edited_result','Q21_PHOTO_RESULT')],
        values=stitch.values(),settings=stitch)
    save={'run_edit':False,'filename_prefix':'Qwen21_Photo/Edited'}
    add(14,'Q21PhotoReviewSave','10 | RUN EDIT OFF = mask preview; ON = edit/save',(2990,100),(710,700),
        inputs=[('crop_data','Q21_PHOTO_CROP'),('edited_result','Q21_PHOTO_RESULT')],values=save.values(),settings=save)
    text=('START: Load original source. Keep run_edit OFF. Auto mask OFF uses your saved source mask; ON uses SAM 3.1.\n'
          'Review the overlay, then turn run_edit ON. Only that second run loads Qwen and references.\n'
          'Use sRGB 8-bit inputs. This is protected crop/edit/composite, not whole-photo generation.\n'
          '2048 is a 2048x2048 AREA CEILING. Smaller crops stay native; shape is preserved; 32px padding is removed at stitch.\n'
          'Fill refs left to right. Ref 1 = <image2>, ref 2 = <image3>, etc. None disables a ref. Roles are your own text, not rewritten.\n'
          'Four ref loaders are shown; duplicate a loader and connect reference_5 through reference_9 for more.\n'
          'All LoRAs are OFF. Use only Qwen-Image-2.1-compatible adapters. No BFS, face restoration, prompt enhancer or AI upscaler is enabled.\n'
          'The saved result has original dimensions. Every zero-mask pixel is copied. SAM boundaries and generated identity still require visual review.')
    add(15,'Note','READ FIRST',(0,-290),(1980,290),values=[text])
    for a,slot,b,name in [(1,0,2,'source'),(1,1,2,'manual_mask'),(1,0,3,'source'),(2,0,3,'mask'),(2,1,3,'mask_report'),
                          (3,0,4,'crop_data'),(4,0,5,'model'),(4,1,10,'clip'),(4,2,10,'vae'),(3,0,10,'crop_data'),
                          (5,0,11,'model'),(10,0,11,'positive'),(10,1,11,'negative'),(10,2,11,'latent_image'),
                          (11,0,12,'samples'),(4,2,12,'vae'),(3,0,13,'crop_data'),(12,0,13,'generated_crop'),
                          (10,3,13,'reference_report'),(3,0,14,'crop_data'),(13,0,14,'edited_result')]: wire(a,slot,b,name)
    for j in range(1,5):wire(5+j,0,10,f'reference_{j}')
    # Correct topological ordering; lazy result selection still controls actual execution.
    order=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    for n in nodes:n['order']=order.index(n['id'])
    wf={'id':str(uuid.uuid5(uuid.NAMESPACE_URL,'qwen21-photo-edit-v2:'+precision)),'revision':0,'last_node_id':15,'last_link_id':len(links),
        'nodes':nodes,'links':links,'groups':[], 'config':{},
        'extra':{'ds':{'scale':.65,'offset':[50,320]},'qwen21_photo_version':'2.0.0'},'version':.4}
    return wf,api

if __name__ == '__main__':
    (ROOT/'workflows').mkdir(exist_ok=True)
    for precision,name in [('BF16 quality','Qwen21_Photo_2K_SAM3_BF16'),('INT8 lower memory','Qwen21_Photo_2K_SAM3_INT8')]:
        wf,api=make(precision)
        (ROOT/'workflows'/f'{name}.json').write_text(json.dumps(wf,indent=2)+'\n')
        (ROOT/'workflows'/f'{name}.api.json').write_text(json.dumps(api,indent=2)+'\n')
