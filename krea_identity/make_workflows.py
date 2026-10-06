#!/usr/bin/env python3
"""Regenerate the four graphs from the repository's already-versioned schemas."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEIGHTS = '1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0'
# Bundle the exact source schemas so Docker's /opt/krea-identity tests do not
# depend on a repository-only /opt/config directory.
SCHEMAS = json.loads((HERE/'config/graph_prototypes.json').read_text())
BASE, SEED = SCHEMAS['base'], SCHEMAS['seed']
PROTOTYPES = {n['type']: n for n in BASE['nodes']}
PROTOTYPES.update({n['type']: n for n in SEED['nodes'] if 'SeedVR2' in n['type']})
spec = importlib.util.spec_from_file_location('krea_identity_nodes_for_graphs', HERE/'node/__init__.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


class Graph:
    def __init__(self, recipe):
        self.recipe, self.nodes, self.links = recipe, [], []

    def add(self, kind, title, pos, values=None):
        if kind in PROTOTYPES:
            n = deepcopy(PROTOTYPES[kind])
            # Get explicit named values even for the inherited SeedVR2 graphs.
            if 'widgets_values_named' not in n and kind.startswith('SeedVR2'):
                names = [i['name'] for i in n['inputs'] if 'widget' in i]
                if 'seed' in names:
                    names.insert(names.index('seed')+1, 'control_after_generate')
                n['widgets_values_named'] = dict(zip(names, n['widgets_values']))
        elif kind in local.NODE_CLASS_MAPPINGS:
            cls = local.NODE_CLASS_MAPPINGS[kind]
            n = {'inputs': [], 'outputs': [], 'widgets_values_named': {}, 'widgets_values': []}
            for group in ('required', 'optional'):
                for name, definition in cls.INPUT_TYPES().get(group, {}).items():
                    typ = definition[0]
                    if isinstance(typ, list) or typ in ('STRING','INT','FLOAT','BOOLEAN'):
                        val = typ[0] if isinstance(typ,list) else definition[1].get('default', '')
                        n['widgets_values_named'][name] = val
                    else:
                        n['inputs'].append({'name': name, 'type': typ, 'link': None})
            for name, typ in zip(getattr(cls,'RETURN_NAMES',cls.RETURN_TYPES),cls.RETURN_TYPES):
                n['outputs'].append({'name': name, 'type': typ, 'links': []})
        elif kind == 'PreviewImage':
            n = {'inputs':[{'name':'images','type':'IMAGE','link':None}], 'outputs':[], 'widgets_values':[]}
        else:
            raise ValueError(kind)
        n.update(id=len(self.nodes)+1, type=kind, title=title, pos=list(pos),
                 size=[340, max(110, 90+34*len(n.get('widgets_values_named',{})))], flags={}, mode=0,
                 properties={'Node name for S&R':kind})
        for i in n.get('inputs',[]): i['link'] = None
        for i in n.get('outputs',[]): i['links'] = []
        if values:
            if kind=='Note': n['widgets_values']=[values['text']]
            else: n.setdefault('widgets_values_named',{}).update(values)
        if 'widgets_values_named' in n:
            n['widgets_values']=list(n['widgets_values_named'].values())
        self.nodes.append(n)
        return n

    def link(self, a, slot, b, field):
        names = [i['name'] for i in b['inputs']]
        typ = a['outputs'][slot]['type']
        if field not in names:
            # A frontend widget converted to a linked input keeps its serialized
            # widget value (as in the source graphs).
            b['inputs'].append({'name':field,'type':typ,'link':None,'widget':{'name':field}})
            names.append(field)
        index=names.index(field)
        if b['inputs'][index].get('link') is not None: raise ValueError('Double connection')
        lid=len(self.links)+1
        self.links.append([lid,a['id'],slot,b['id'],index,typ])
        a['outputs'][slot]['links'].append(lid)
        b['inputs'][index].update(type=typ,link=lid)

    def save(self, filename):
        # Topological orders are calculated, never guessed from canvas location.
        done=set(); remaining=list(self.nodes)
        while remaining:
            available=[n for n in remaining if all(l[1] in done for l in self.links if l[3]==n['id'])]
            if not available: raise ValueError('Cycle')
            for n in available:
                n['order']=len(done); done.add(n['id']); remaining.remove(n)
        data={'last_node_id':len(self.nodes),'last_link_id':len(self.links),
              'nodes':self.nodes,'links':self.links,'groups':[], 'config':{},
              'extra':{'ds':{'scale':0.42,'offset':[1220,650]},
                       'krea_identity':{'version':'1.0.0','recipe':self.recipe,'comfy_ref':'65787d668397d230bf5839d69a0a7239e2dad378'}},'version':0.4}
        (HERE/'workflows'/filename).write_text(json.dumps(data,indent=2)+'\n')
        return data


def edit_graph(recipe, filename):
    g=Graph(recipe)
    scene=g.add('LoadImage','3. SCENE / POSE (base photograph)',(-1200,560))
    if recipe=='three_reference':
        face=g.add('LoadImage','1. FACE reference (original photograph)',(-1200,-500))
        body=g.add('LoadImage','2. BODY reference (same person)',(-1200,20))
        prepare=g.add('KreaIdentityPrepare','START HERE - reference roles + clothing',(-750,-470))
        g.link(face,0,prepare,'face');g.link(body,0,prepare,'body');g.link(scene,0,prepare,'scene')
    else:
        face=g.add('LoadImage','REFERENCE person / face',(-1200,-400))
        prepare=g.add('KreaIdentityPair','START HERE - two-image edit',(-750,-300))
        g.link(face,0,prepare,'subject')
        if recipe=='face_refine':
            crop=g.add('KreaIdentityCrop','Select head + neck with surrounding context',(-1200,100))
            g.link(scene,0,crop,'image');g.link(crop,0,prepare,'scene')
            preview_crop=g.add('PreviewImage','Check crop before judging the refinement',(-300,870))
            g.link(crop,0,preview_crop,'images')
            prepare['widgets_values_named']['megapixels']=1.0
            prepare['widgets_values_named']['instruction']='Replace only the face in image 1 with the identity in image 2. Preserve image 1\'s head angle, expression, gaze, hair, ears, neck, lighting and surrounding pixels. Match the reference facial geometry without beautification. One natural photographic face.'
            prepare['widgets_values']=list(prepare['widgets_values_named'].values())
        else: g.link(scene,0,prepare,'scene')
    preview=g.add('PreviewImage','Exact subject reference sent to Krea',(-750,600))
    g.link(prepare,1,preview,'images')
    unet=g.add('UNETLoader','Krea 2 Turbo BF16 (quality baseline)',(-300,-580),{'unet_name':'krea2_turbo_bf16.safetensors','weight_dtype':'default'})
    clip=g.add('CLIPLoader','Qwen3-VL 4B BF16 / type=krea2',(-300,-370))
    vae=g.add('VAELoader','Qwen Image VAE',(-300,-140))
    lora=g.add('LoraLoaderModelOnly','Required GENERAL editor, not a trained subject LoRA',(100,-580),{'lora_name':'krea2_identity_edit_v1_2.safetensors','strength_model':1.0})
    g.link(unet,0,lora,'model')
    a=g.add('VAEEncode','Scene appearance',(100,490));b=g.add('VAEEncode','Subject appearance',(100,660))
    g.link(prepare,0,a,'pixels');g.link(prepare,1,b,'pixels')
    g.link(vae,0,a,'vae');g.link(vae,0,b,'vae')
    empty=g.add('EmptySD3LatentImage','Single image / aspect follows SCENE',(100,290),{'batch_size':1})
    g.link(prepare,3,empty,'width');g.link(prepare,4,empty,'height')
    enc=g.add('Krea2EditGroundedEncode','Positive: SAME scene + subject references',(100,-350),{'prompt':'','grounding_px':1024,'system_prompt':''})
    neg=g.add('Krea2EditGroundedEncode','Negative: empty instruction, SAME references',(100,-10),{'prompt':'','grounding_px':1024,'system_prompt':''})
    g.link(prepare,2,enc,'prompt')
    for n in (enc,neg):
        g.link(clip,0,n,'clip');g.link(prepare,0,n,'image');g.link(prepare,1,n,'image_b')
    patch=g.add('Krea2EditModelPatch','Scene strength=1 / Subject strength=4 / FIT',(650,-580),{'ref_boost':4.0,'ref_boost_a':1.0,'fit_mode':'fit'})
    g.link(lora,0,patch,'model');g.link(a,0,patch,'source_latent');g.link(b,0,patch,'source_latent_b')
    g.link(vae,0,patch,'vae');g.link(prepare,0,patch,'source_image');g.link(prepare,1,patch,'source_image_b')
    g.link(empty,0,patch,'target_latent')
    posrb=g.add('ConditioningKrea2Rebalance','Screenshot rebalancer - POSITIVE',(650,-190),{'multiplier':1.0,'per_layer_weights':WEIGHTS})
    negrb=g.add('ConditioningKrea2Rebalance','Matching rebalancer - NEGATIVE',(650,45),{'multiplier':1.0,'per_layer_weights':WEIGHTS})
    g.link(enc,0,posrb,'conditioning');g.link(neg,0,negrb,'conditioning')
    choose=g.add('KreaIdentityRebalanceControl','Rebalancer OFF baseline / ON for fixed-seed A/B',(1090,-190),{'enabled':False})
    g.link(enc,0,choose,'positive');g.link(neg,0,choose,'negative')
    g.link(posrb,0,choose,'rebalanced_positive');g.link(negrb,0,choose,'rebalanced_negative')
    sample=g.add('KSampler','Generate ONE image / 12 steps / CFG 1',(1510,-530),{'seed':42,'control_after_generate':'fixed','steps':12,'cfg':1.0,'sampler_name':'euler','scheduler':'simple','denoise':1.0})
    g.link(patch,0,sample,'model');g.link(choose,0,sample,'positive');g.link(choose,1,sample,'negative');g.link(empty,0,sample,'latent_image')
    decode=g.add('VAEDecode','Native edit (before any upscaling)',(1920,-530))
    g.link(sample,0,decode,'samples');g.link(vae,0,decode,'vae')
    result=decode
    if recipe=='face_refine':
        result=g.add('KreaIdentityStitch','Refined head ONLY / outside crop unchanged',(2320,-530))
        g.link(scene,0,result,'original');g.link(decode,0,result,'refined_crop');g.link(crop,1,result,'crop_geometry')
    save=g.add('SaveImage','SAVE approved native image',(2720 if recipe=='face_refine' else 2320,-530),{'filename_prefix':'KreaIdentity/'+recipe})
    g.link(result,0,save,'images')
    text=('KREA IDENTITY 1.0 | No custom training. Upload original face/body and a scene photo. '
          'The three-reference mode creates a two-panel subject sheet, not a generated person. This is an EXPERIMENTAL use of reference sheets, '
          'not a guaranteed three-role fusion model. If likeness or single-person composition fails, switch reference_mode to Face only or use the two-image graph. '
          'Start at 1.5 MP; never exceed 2 MP here. Grounding=1024 favors likeness; try 768 if duplicate panels or weak edits. '
          'Turbo 12 steps / CFG 1 / Euler-simple / denoise 1 on EMPTY latent is intentional. '
          'Rebalancer starts OFF. Enable its single A/B switch without changing the fixed seed; use multiplier 1 and the screenshot weights. '
          'The multiplier does not disable non-unity per-layer weights. Ref boost is attention weighting, not an identity guarantee. '
          'Approve facial geometry before using the separate upscale graph. No LLM rewriting, no external inference API. '
          'For face refinement: crop must include head, neck and context; outside the crop is composited from the original, not regenerated.')
    note=g.add('Note','Read once - quality and identity controls',(1110,330),{'text':text});note['size']=[750,360]
    return g.save(filename)


def upscale_graph():
    g=Graph('upscale')
    image=g.add('LoadImage','Load an APPROVED single image',(-700,0))
    size=g.add('KreaIdentityUpscaleSize','2x first / 4096 maximum long edge',(-290,0))
    g.link(image,0,size,'image')
    dit=g.add('SeedVR2LoadDiTModel','SeedVR2 7B FP16 (no quantization)',(-290,250))
    vae=g.add('SeedVR2LoadVAEModel','Tiled VAE / CPU offload',(-290,600))
    up=g.add('SeedVR2VideoUpscaler','SINGLE image / zero extra noise',(160,0),{
        'seed':42,'control_after_generate':'fixed','batch_size':1,'uniform_batch_size':False,
        'temporal_overlap':0,'prepend_frames':0,'input_noise_scale':0.0,'latent_noise_scale':0.0,
        'color_correction':'lab','offload_device':'cpu','enable_debug':False})
    g.link(image,0,up,'image');g.link(dit,0,up,'dit');g.link(vae,0,up,'vae')
    g.link(size,0,up,'resolution');g.link(size,1,up,'max_resolution')
    save=g.add('SaveImage','Save upscale separately; retain the native original',(590,0),{'filename_prefix':'KreaIdentity/Upscaled'})
    g.link(up,0,save,'images')
    note=g.add('Note','Upscaling is not identity correction',(-700,560),{'text':
        'Requires ENABLE_KREA_IDENTITY=1 and KREA_IDENTITY_UPSCALE=1 at Pod startup. '
        'This accepts one IMAGE directly, not a video or extracted frame. The native Krea output is never overwritten. '
        'Start at 2x. 4x and an 8192 cap are available but are not evidence of real recovered detail. '
        'Compare facial proportions at matched display size. Reject an upscale that changes identity. '
        'Load from the preceding SaveImage result; this deliberately prevents upscaling every failed generation. '
        'No automatic face restoration or beautification is applied. GPU memory and quality still require a real RunPod test.'})
    note['size']=[760,270]
    return g.save('Krea_Identity_04_Upscale_v1_0.json')


def main():
    (HERE/'workflows').mkdir(exist_ok=True)
    edit_graph('three_reference','Krea_Identity_01_Face_Body_Scene_v1_0.json')
    edit_graph('two_reference','Krea_Identity_02_Scene_One_Reference_v1_0.json')
    edit_graph('face_refine','Krea_Identity_03_Face_Refine_v1_0.json')
    upscale_graph()

if __name__=='__main__':main()
