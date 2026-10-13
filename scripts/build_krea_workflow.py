#!/usr/bin/env python3
"""Rebuild editable Krea graphs and API exports; no ComfyUI/model imports."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = 'workflows/krea/K01/Krea_Edit_Multi_Reference_HQ.json'
API = 'automation/K01/Krea_Edit_Multi_Reference_HQ.api.json'
UI_STANDARD = 'workflows/krea/K01/Krea_Standard_No_Identity_LoRA_HQ.json'
API_STANDARD = 'automation/K01/Krea_Standard_No_Identity_LoRA_HQ.api.json'
UI_MASKED = 'workflows/krea/K01/Krea_Masked_Photo_HQ.json'
API_MASKED = 'automation/K01/Krea_Masked_Photo_HQ.api.json'
BACKEND = 'Local Qwen vision (workspace model)'


def _relink(graph, api):
    byid = {n['id']: n for n in graph['nodes']}
    graph['links'] = []
    for n in graph['nodes']:
        for p in n['inputs']:
            p['link'] = None
        for p in n['outputs']:
            p['links'] = []
    for dest, row in api.items():
        for name, value in row['inputs'].items():
            if not (isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)):
                continue
            src, slot = int(value[0]), value[1]
            ports = byid[int(dest)]['inputs']
            index = next(i for i, p in enumerate(ports) if p['name'] == name)
            origin = byid[src]['outputs'][slot]
            lid = len(graph['links']) + 1
            graph['links'].append([lid, src, slot, int(dest), index, origin['type']])
            origin['links'].append(lid)
            ports[index]['link'] = lid
    # Topological ordering, including lazy branches; no cycles are permitted.
    remaining = set(byid)
    ordered = []
    while remaining:
        ready = sorted(i for i in remaining if all(link[1] in ordered for link in graph['links'] if link[3] == i))
        if not ready:
            raise ValueError('Workflow graph has a cycle.')
        ordered.extend(ready)
        remaining.difference_update(ready)
    for i, node_id in enumerate(ordered):
        byid[node_id]['order'] = i
    graph['last_node_id'] = max(byid)
    graph['last_link_id'] = len(graph['links'])
    return graph, api


def _node(graph, api, id, kind, title, position, size, ports, outputs, widgets, values):
    node = dict(id=id, type=kind, title=title, pos=position, size=size, flags={}, order=id-1, mode=0,
                inputs=[{'name': n, 'type': t, 'link': None} for n, t in ports],
                outputs=[{'name': n, 'type': t, 'links': []} for n, t in outputs],
                properties={'Node name for S&R': kind}, widgets_values=widgets)
    graph['nodes'].append(node)
    if kind != 'Note':
        api[str(id)] = {'class_type': kind, 'inputs': values, '_meta': {'title': title}}
    return node


def build():
    graph = {'last_node_id': 15, 'last_link_id': 0, 'nodes': [], 'links': [], 'groups': [],
             'config': {}, 'extra': {'ds': {'scale': .6, 'offset': [100, 100]}}, 'version': .4}
    api = {}
    def add(*args):
        return _node(graph, api, *args)
    add(1, 'WBKreaLoadImage', '1. SOURCE photo / sRGB', [30,80], [350,350], [],
        [('IMAGE','IMAGE'),('MASK','MASK')], ['', 'image'], {'image': ''})
    roles = ['Face and identity of the replacement subject', 'Body proportions of the same subject',
             'Additional view of the same subject', 'Clothing or object details only']
    for i, role in enumerate(roles):
        add(i+2, 'WBKreaReference', f'Optional reference {i+1}: enable after upload',
            [440,80+i*240], [340,210], [], [('reference','WB_KREA_REF')],
            [False,'none',role], {'enabled':False,'image':'none','role':role})
    add(6, 'WBKreaPrepare', '2. Source + reference roles', [840,80], [340,250],
        [('source','IMAGE')]+[(f'reference_{i}','WB_KREA_REF') for i in range(1,5)],
        [('editing_inputs','WB_KREA_PACKAGE'),('reference_sheet_preview','IMAGE')], [1.9],
        {'source':['1',0], 'megapixels':1.9, **{f'reference_{i}':[str(i+1),0] for i in range(1,5)}})
    instruction = 'Edit the source photo using the enabled references for their assigned roles. Describe the specific change here.'
    add(7, 'WBKreaPrompt', '3. Edit prompt + verified Qwen vision', [1250,80], [470,740],
        [('editing_inputs','WB_KREA_PACKAGE')], [('final_edit_instruction','STRING')],
        [instruction,True,'auto','',1536,BACKEND], {'editing_inputs':['6',0], 'instruction':instruction,
        'enhance':True,'ollama_model':'auto','trigger_words':'','max_tokens':1536,'backend':BACKEND})
    add(8, 'WBKreaModels', '4. Krea checkpoint + correct encoder', [1790,80], [400,330],
        [('ready_prompt','STRING')], [('model','MODEL'),('krea_encoder','CLIP'),('vae','VAE'),('steps','INT'),('cfg','FLOAT')],
        ['krea2_raw_bf16.safetensors','qwen3vl_4b_bf16.safetensors','qwen_image_vae.safetensors','Auto',20,3.0],
        {'ready_prompt':['7',0],'checkpoint':'krea2_raw_bf16.safetensors','text_encoder':'qwen3vl_4b_bf16.safetensors',
         'vae_name':'qwen_image_vae.safetensors','sampling_preset':'Auto','custom_steps':20,'custom_cfg':3.0})
    add(9, 'WBKreaLoRAs', '5. Edit adapter + optional LoRAs', [1790,470], [400,300], [('model','MODEL')], [('MODEL','MODEL')],
        ['krea2_identity_edit_v1_2.safetensors',1.0,'none',1.0,'none',1.0], {'model':['8',0],
         'edit_adapter':'krea2_identity_edit_v1_2.safetensors','edit_strength':1.0,
         'lora_1':'none','strength_1':1.0,'lora_2':'none','strength_2':1.0})
    add(10, 'WBKreaCondition', '6. Image-grounded FIT conditioning', [2260,80], [390,300],
        [('model','MODEL'),('clip','CLIP'),('vae','VAE'),('editing_inputs','WB_KREA_PACKAGE'),('prompt','STRING')],
        [('edit_model','MODEL'),('positive','CONDITIONING'),('grounded_negative','CONDITIONING'),('target_latent','LATENT')],
        [1.0,4.0,768], {'model':['9',0],'clip':['8',1],'vae':['8',2],'editing_inputs':['6',0],'prompt':['7',0],
         'scene_strength':1.0,'reference_strength':4.0,'grounding_px':768})
    sampler = add(11, 'KSampler', '7. Generate', [2720,80], [330,360],
        [('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT'),('steps','INT'),('cfg','FLOAT')],
        [('LATENT','LATENT')], [12345,'randomize',20,3.0,'euler','simple',1.0],
        {'model':['10',0],'positive':['10',1],'negative':['10',2],'latent_image':['10',3],
         'steps':['8',3],'cfg':['8',4],'seed':12345,'sampler_name':'euler','scheduler':'simple','denoise':1.0})
    sampler['inputs'][4]['widget'] = {'name':'steps'}
    sampler['inputs'][5]['widget'] = {'name':'cfg'}
    add(12, 'VAEDecode', '8. Decode', [2720,510], [330,120], [('samples','LATENT'),('vae','VAE')], [('IMAGE','IMAGE')],
        [], {'samples':['11',0],'vae':['8',2]})
    add(13, 'WBKreaSaveImage', '9. Save sRGB PNG', [3130,80], [620,720], [('images','IMAGE')], [],
        ['Krea_Edit'], {'images':['12',0],'filename_prefix':'Krea_Edit'})
    add(14, 'PreviewImage', 'Reference sheet sent to Krea', [840,400], [340,430], [('images','IMAGE')], [], [], {'images':['6',1]})
    note = ('KREA 2 IDENTITY EDIT\n\nSource + 0-4 optional references. One reference goes directly into the second input; '
        '2-4 references form a labeled sheet. This is not native five-image conditioning.\n\n'
        'Enhancer auto uses WB_QWEN_PROMPT_MODEL, the same full-precision vision helper as Qwen. '
        'Invalid rewrites retry once, then visibly retain your instruction. Service or memory failures stop explicitly. '
        'The helper unloads before Krea. Trigger words remain literal.\n\n'
        'Raw: 20 steps / CFG 3. Turbo: 8 / CFG 1. Keep Identity Edit adapter ON. '
        'Two extra Krea-compatible LoRAs are optional.\n\n'
        'Full-frame editing, not a pixel-preserving mask workflow. Maximum 2 MP, default 1.9, no automatic upscale. '
        'Embedded input ICC profiles are converted to sRGB; untagged inputs are assumed sRGB.\n\n'
        'Optional official PE-I2I requires its Toolbox assets. GPU validation pending.')
    add(15, 'Note', 'READ FIRST', [30,500], [350,650], [], [], [note], {})
    return _relink(graph, api)


def build_standard():
    graph, api = build()
    byid = {n['id']:n for n in graph['nodes']}
    changes = [(7,'WBKreaStandardPrompt','3. Standalone target-photo prompt'),
               (9,'WBKreaOptionalLoRAs','5. Optional LoRAs; no edit adapter'),
               (10,'WBKreaStandardCondition','6. Native text-only conditioning')]
    for id, kind, title in changes:
        byid[id].update(type=kind,title=title,properties={'Node name for S&R':kind})
        api[str(id)].update(class_type=kind,_meta={'title':title})
    instruction = 'Describe the target photograph using enabled references only for their assigned appearance details.'
    byid[7]['widgets_values'][0] = instruction
    api['7']['inputs']['instruction'] = instruction
    byid[9]['widgets_values'] = ['none',1.0,'none',1.0]
    api['9']['inputs'] = {'model':['8',0],'lora_1':'none','strength_1':1.0,'lora_2':'none','strength_2':1.0}
    byid[10]['inputs'] = [p for p in byid[10]['inputs'] if p['name'] != 'vae']
    byid[10]['widgets_values'] = []
    byid[10]['outputs'][0]['name'] = 'model'
    byid[10]['outputs'][2]['name'] = 'negative'
    api['10']['inputs'] = {k:v for k,v in api['10']['inputs'].items() if k in ('model','clip','editing_inputs','prompt')}
    byid[13]['widgets_values'] = ['Krea_Standard_No_Identity']
    api['13']['inputs']['filename_prefix'] = 'Krea_Standard_No_Identity'
    byid[14]['title'] = 'Prompt-analysis references only; NOT sent to Krea'
    byid[15]['widgets_values'] = ['KREA STANDARD: TEXT-ONLY GENERATION\n\nNo Identity Edit adapter or native image conditioning. '
        'The local vision helper sees your source/references and describes the target; Krea sees only text. '
        'Source dimensions set output aspect ratio. This does not preserve source pixels or guarantee likeness.\n\n'
        'With enhancer OFF, provide a complete standalone generation prompt. Official PE-I2I is not supported here. '
        'Raw/Turbo and two optional compatible LoRAs remain available.\n\nGPU validation pending.']
    return _relink(graph, api)


def build_masked():
    graph, api = build()
    graph['nodes'] = [n for n in graph['nodes'] if n['id'] != 14]
    del api['14']  # No independent reference-preview output may bypass the lazy review gate.
    byid = {n['id']:n for n in graph['nodes']}
    def add(*args):
        return _node(graph, api, *args)
    add(16, 'WBKreaMask', 'A. SOURCE mask: manual or SAM 3.1', [30,500], [360,400],
        [('source','IMAGE'),('manual_mask','MASK')], [('effective_edit_mask','MASK'),('mask_report','STRING')],
        [False,'person',.5,2,-1,'full photo','ignore',0], {'source':['1',0],'manual_mask':['1',1],
        'auto_mask':False,'sam_prompt':'person','threshold':.5,'refine_iterations':2,'instance_index':-1,
        'search_region':'full photo','manual_correction':'ignore','expand_pixels':0})
    add(17, 'WBMaskExpandFeather', 'B. Expand + outward fade', [30,960], [360,210], [('mask','MASK')],
        [('expanded_feathered_mask','MASK')], [32,128,False],
        {'mask':['16',0],'expand_pixels':32,'feather_pixels':128,'invert':False})
    add(18, 'WBKreaCrop', 'C. Crop context / Krea 2 MP cap', [840,430], [340,360],
        [('source','IMAGE'),('mask','MASK'),('mask_report','STRING')], [('crop_data','WB_KREA_CROP'),('working_crop','IMAGE')],
        [1.9,2048,128,False], {'source':['1',0],'mask':['17',0],'mask_report':['16',1],
        'megapixels':1.9,'max_long_side':2048,'context_pixels':128,'upscale_small_crops':False})
    byid[6]['widgets_values'] = [2.0]
    api['6']['inputs'].update(source=['18',1],megapixels=2.0)
    add(19, 'WBKreaStitch', 'D. Protected stitch + optional boundary match', [2720,720], [360,190],
        [('crop_data','WB_KREA_CROP'),('generated_crop','IMAGE')], [('WB_KREA_RESULT','WB_KREA_RESULT')], [0.],
        {'crop_data':['18',0],'generated_crop':['12',0],'boundary_color_strength':0.})
    graph['nodes'] = [n for n in graph['nodes'] if n['id'] != 13]
    del api['13']
    add(13, 'WBKreaReviewSave', 'E. Preview mask first / run_edit ON to generate', [3130,80], [620,720],
        [('crop_data','WB_KREA_CROP'),('edited_result','WB_KREA_RESULT')], [], [False,'Krea_Masked'],
        {'crop_data':['18',0],'edited_result':['19',0],'run_edit':False,'filename_prefix':'Krea_Masked'})
    byid[15]['pos'] = [440,1130]
    byid[15]['size'] = [740,410]
    byid[15]['widgets_values'] = ['KREA MASKED PHOTO\n\nPaint/save a mask on SOURCE, or turn auto_mask ON and name the source object. '
        'SAM requires the optional sam Toolbox download. Empty or mismatched masks are rejected.\n\n'
        'run_edit OFF previews the expanded/feathered mask only. No Krea, LoRA, reference or prompt-model inference. '
        'Review this boundary, then turn run_edit ON.\n\n'
        'Generate a <=2 MP working crop, restore its original coordinates, and paste only through the reviewed mask. '
        'Pixels outside it remain exact. Expansion/fade enlarge the editable region. Optional protect_mask on the crop node '
        'is applied after expansion. Boundary color correction is OFF by default.\n\n'
        'Full-size sRGB PNG plus verification JSON. This does not guarantee generated identity or detail. GPU validation pending.']
    return _relink(graph, api)


def main():
    for paths, builder in [((UI,API),build), ((UI_STANDARD,API_STANDARD),build_standard), ((UI_MASKED,API_MASKED),build_masked)]:
        for path, value in zip(paths, builder()):
            output = ROOT/path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(value, separators=(',',':'))+'\n', encoding='utf-8')
            print(path)


if __name__ == '__main__':
    main()
