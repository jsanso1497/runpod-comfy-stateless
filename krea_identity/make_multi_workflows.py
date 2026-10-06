#!/usr/bin/env python3
"""Build v1.1 additions without rewriting the four working v1.0 workflows."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('krea_original_graph_builder', HERE/'make_workflows.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Graph(base.Graph):
    def __init__(self, recipe):
        super().__init__(recipe)
        self.stages = []
        self.groups = []
    def save(self, filename):
        data = super().save(filename)
        data['extra']['krea_identity'].update(version='1.1.0', stages=self.stages)
        data['extra']['ds'] = {'scale': 0.28, 'offset': [2220, 1400]}
        data['groups'] = self.groups
        (HERE/'workflows'/filename).write_text(json.dumps(data, indent=2)+'\n')
        return data


def note(g, title, text, x, y, width=720, height=230):
    n=g.add('Note', title, (x,y), {'text':text})
    n['size']=[width,height]
    return n


def references(g, prefix, person, x, y):
    face = g.add('LoadImage', f'{prefix}1 - {person}: best FACE photograph', (x,y))
    face['size']=[350,420]
    r1 = g.add('KreaIdentityReference', f'{prefix}1 - label + purpose (primary identity)', (x+390,y), {
        'label': f'{person} - primary face',
        'use_for': 'Facial identity, facial structure, eye/nose/mouth shapes and natural skin detail. Do not copy pose, clothing or background.',
        'include':True})
    r1['size']=[420,310]
    body = g.add('LoadImage', f'{prefix}2 - {person}: BODY photograph', (x,y+470))
    body['size']=[350,420]
    r2 = g.add('KreaIdentityReference', f'{prefix}2 - body reference; add more after this node', (x+390,y+470), {
        'label':f'{person} - physique',
        'use_for':'Body build and proportions only. Preserve the scene pose and scene clothing. Facial identity comes from the primary face panel.',
        'include':True})
    r2['size']=[420,310]
    g.link(face,0,r1,'image');g.link(body,0,r2,'image');g.link(r1,0,r2,'references')
    note(g, f'Add references for {person}',
         'Duplicate a LoadImage + Labeled Reference pair. Upload the next original photograph, enter its label and use_for purpose, '
         'connect this list output to the new node\'s references input, then connect the NEW list output to the edit node. '
         'Up to 6 active photographs per person. Put the strongest face first. Examples: side-profile geometry, hair only, wardrobe only. '
         'Use 2-3 complementary originals first; every extra panel shares the same conditioning resolution. '
         'Remove a body reference by connecting the face list directly to the edit node, or keep its uploaded file and switch include off.',
         x,y+960,810,235)
    return r2


def models(g, x=-200, y=-1300):
    model=g.add('UNETLoader','Krea 2 Turbo BF16 - unchanged quality baseline',(x,y),{'unet_name':'krea2_turbo_bf16.safetensors','weight_dtype':'default'})
    clip=g.add('CLIPLoader','Qwen3-VL 4B BF16 / type=krea2',(x,y+190))
    vae=g.add('VAELoader','Qwen Image VAE',(x,y+400))
    lora=g.add('LoraLoaderModelOnly','Existing Identity Edit v1.2 / NOT subject training',(x+410,y),{'lora_name':'krea2_identity_edit_v1_2.safetensors','strength_model':1.0})
    g.link(model,0,lora,'model')
    return lora,clip,vae


def stage(g, prepare, modelset, label, x=0, y=0, seed=42):
    lora,clip,vae=modelset
    a=g.add('VAEEncode',label+' / scene appearance',(x,y+550))
    b=g.add('VAEEncode',label+' / labeled subject appearance',(x,y+700))
    g.link(prepare,0,a,'pixels');g.link(prepare,1,b,'pixels')
    g.link(vae,0,a,'vae');g.link(vae,0,b,'vae')
    empty=g.add('EmptySD3LatentImage',label+' / aspect from scene or selected crop',(x,y+340),{'batch_size':1})
    g.link(prepare,3,empty,'width');g.link(prepare,4,empty,'height')
    pos=g.add('Krea2EditGroundedEncode',label+' / grounded prompt + SAME references',(x,y-340),{'prompt':'','grounding_px':1024,'system_prompt':''})
    neg=g.add('Krea2EditGroundedEncode',label+' / trained empty negative',(x,y-20),{'prompt':'','grounding_px':1024,'system_prompt':''})
    g.link(prepare,2,pos,'prompt')
    for n in (pos,neg):
        g.link(clip,0,n,'clip');g.link(prepare,0,n,'image');g.link(prepare,1,n,'image_b')
    patch=g.add('Krea2EditModelPatch',label+' / Scene=1, Subject=4, FIT',(x+430,y-590),{'ref_boost':4.0,'ref_boost_a':1.0,'fit_mode':'fit'})
    # Every pass starts from the shared unpatched editor, never a previous
    # person's ModelPatch. This prevents stacking person A and B references.
    g.link(lora,0,patch,'model');g.link(a,0,patch,'source_latent');g.link(b,0,patch,'source_latent_b')
    g.link(prepare,0,patch,'source_image');g.link(prepare,1,patch,'source_image_b')
    g.link(vae,0,patch,'vae');g.link(empty,0,patch,'target_latent')
    rbp=g.add('ConditioningKrea2Rebalance',label+' / screenshot rebalancer POSITIVE',(x+430,y-200),{'multiplier':1.0,'per_layer_weights':base.WEIGHTS})
    rbn=g.add('ConditioningKrea2Rebalance',label+' / screenshot rebalancer NEGATIVE',(x+430,y+35),{'multiplier':1.0,'per_layer_weights':base.WEIGHTS})
    g.link(pos,0,rbp,'conditioning');g.link(neg,0,rbn,'conditioning')
    switch=g.add('KreaIdentityRebalanceControl',label+' / enable rebalancer (OFF baseline)',(x+850,y-200),{'enabled':False})
    g.link(pos,0,switch,'positive');g.link(neg,0,switch,'negative')
    g.link(rbp,0,switch,'rebalanced_positive');g.link(rbn,0,switch,'rebalanced_negative')
    sampler=g.add('KSampler',label+' / 12 steps, CFG 1, fixed seed',(x+1270,y-520),{'seed':seed,'control_after_generate':'fixed','steps':12,'cfg':1.0,'sampler_name':'euler','scheduler':'simple','denoise':1.0})
    g.link(patch,0,sampler,'model');g.link(switch,0,sampler,'positive');g.link(switch,1,sampler,'negative');g.link(empty,0,sampler,'latent_image')
    decode=g.add('VAEDecode',label+' / native Krea image',(x+1670,y-520))
    g.link(sampler,0,decode,'samples');g.link(vae,0,decode,'vae')
    preview=g.add('PreviewImage',label+' / exact labeled references seen by Krea',(x+850,y+100))
    preview['size']=[700,540];g.link(prepare,1,preview,'images')
    g.stages.append({'prepare':prepare['id'],'patch':patch['id'],'sampler':sampler['id'],
                     'positive':pos['id'],'negative':neg['id'],'control':switch['id'],
                     'empty':empty['id'],'decode':decode['id']})
    return decode


def save_unframed(g,prepare,decode,label,x=2080,y=-520):
    trim=g.add('KreaIdentityUnframe','Remove grid margins / retain scene aspect',(x,y))
    g.link(decode,0,trim,'image');g.link(prepare,5,trim,'canvas_geometry')
    save=g.add('SaveImage','Save native image first; workflow 04 handles upscale',(x+400,y),{'filename_prefix':'KreaIdentity/'+label})
    g.link(trim,0,save,'images')
    return save


def directed():
    g=Graph('directed_reference')
    refs=references(g,'A','Replacement person',-2100,-1250)
    scene=g.add('LoadImage','SCENE - landscape, portrait or square',(-2100,50));scene['size']=[700,480]
    prep=g.add('KreaIdentityDirectedEdit','START HERE - who replaces whom',(-1190,-700))
    prep['size']=[670,680]
    g.link(scene,0,prep,'scene');g.link(refs,0,prep,'references')
    modelset=models(g)
    result=stage(g,prep,modelset,'Selected person')
    save_unframed(g,prep,result,'Directed')
    note(g,'DIRECTED REPLACEMENT - read once',
         'Identify the target by scene position + clothing, not just "the man" when several men appear. Example: '
         '"the man on the left in the gray jacket" -> "the man shown in group A". Other people remain in the prompt. '
         'Labels/purposes guide the model; they are not hard regional or attribute constraints. This graph edits the full scene. '
         'For exact background/other-person preservation, use workflow 07\'s crop/composite structure. '
         '1.5 MP is the default; output aspect automatically follows the uploaded scene. Keep the existing Turbo BF16, '
         'Identity Edit, FIT, 1/4 strengths and screenshot rebalancer. No new models or API calls.',-1190,180,690,310)
    return g.save('Krea_Identity_05_Labeled_References_Directed_v1_1.json')


def couple():
    g=Graph('couple_single_pass')
    a=references(g,'A','Man',-2500,-1900)
    b=references(g,'B','Woman',-2500,-570)
    scene=g.add('LoadImage','ONE SCENE - keep its orientation',(-1600,680));scene['size']=[680,440]
    prep=g.add('KreaIdentityCoupleEdit','START HERE - A replaces target A / B replaces target B',(-1600,-880))
    prep['size']=[700,970]
    g.link(scene,0,prep,'scene');g.link(a,0,prep,'references_a');g.link(b,0,prep,'references_b')
    modelset=models(g)
    result=stage(g,prep,modelset,'Both people together')
    save_unframed(g,prep,result,'Couple_SinglePass')
    note(g,'TWO PEOPLE / SINGLE PASS',
         'Describe the actual scene: A=man and B=woman are starting labels, not automatic gender detection or fixed left/right assignments. '
         'Edit BOTH target descriptions when their positions differ. Each person has a separate reference list and clothing choice. '
         'Both lists are placed into one labeled subject sheet, with the scene as the other Krea input. '
         'This is a single generation, NOT native per-person attention isolation. Faces can still blend or be misassigned. '
         'Start with face + body per person; add references only when they contribute missing detail. '
         '2 MP is the maximum working canvas here. For the strongest regional control, use workflow 07. '
         'Approve the native image before running workflow 04 Upscale.',2080,0,780,340)
    return g.save('Krea_Identity_06_Man_Woman_Single_Pass_v1_1.json')


def protected_couple():
    g=Graph('couple_protected')
    a=references(g,'A','Man',-2800,-1700)
    b=references(g,'B','Woman',-2800,-330)
    scene=g.add('LoadImage','ORIGINAL full-resolution scene',(-1900,-1530));scene['size']=[700,440]
    region_a=g.add('KreaIdentityRegion','1. Select MAN region (defaults to left side)',(-1900,-900))
    region_a['size']=[600,490]
    g.link(scene,0,region_a,'image')
    pa=g.add('KreaIdentityDirectedEdit','2. MAN replacement / descriptions refer to CROP',(-1200,-890),{
        'target_description':'the man in this crop',
        'replacement_description':'the man shown in this reference group',
        'extra_instruction':'Change only the selected man. Keep any other people visible in the context crop unchanged.',
        'megapixels':1.5})
    pa['size']=[670,690]
    g.link(region_a,0,pa,'scene');g.link(a,0,pa,'references')
    modelset=models(g,0,-1800)
    out_a=stage(g,pa,modelset,'PASS 1 / MAN',x=0,y=-350,seed=42)
    comp_a=g.add('KreaIdentityRegionComposite','3. Paste ONLY selected man area into original',(2080,-870))
    g.link(scene,0,comp_a,'original');g.link(out_a,0,comp_a,'generated')
    g.link(region_a,1,comp_a,'region_geometry');g.link(pa,5,comp_a,'canvas_geometry')
    save_a=g.add('SaveImage','Intermediate after man / retained for inspection',(2480,-870),{'filename_prefix':'KreaIdentity/Protected_A_Intermediate'})
    g.link(comp_a,0,save_a,'images')
    region_b=g.add('KreaIdentityRegion','4. Select WOMAN region / MAN region protected',(-1900,630),{'left':0.52,'top':0.02,'width':0.46,'height':0.96,'context_padding':0.12,'feather_pixels':16})
    region_b['size']=[600,490]
    g.link(comp_a,0,region_b,'image');g.link(region_a,2,region_b,'protect_mask')
    pb=g.add('KreaIdentityDirectedEdit','5. WOMAN replacement / separate references',(-1200,600),{
        'reference_group':'B',
        'target_description':'the woman in this crop',
        'replacement_description':'the woman shown in this reference group',
        'extra_instruction':'Change only the selected woman. Keep any other people visible in the context crop unchanged.',
        'megapixels':1.5})
    pb['size']=[670,690]
    g.link(region_b,0,pb,'scene');g.link(b,0,pb,'references')
    out_b=stage(g,pb,modelset,'PASS 2 / WOMAN',x=0,y=1250,seed=43)
    comp_b=g.add('KreaIdentityRegionComposite','6. Paste woman / man and outside masks unchanged',(2080,730))
    g.link(comp_a,0,comp_b,'original');g.link(out_b,0,comp_b,'generated')
    g.link(region_b,1,comp_b,'region_geometry');g.link(pb,5,comp_b,'canvas_geometry')
    save=g.add('SaveImage','FINAL full-resolution scene / both replacements',(2480,730),{'filename_prefix':'KreaIdentity/Protected_Couple_Final'})
    g.link(comp_b,0,save,'images')
    for region,y,label in ((region_a,-100,'MAN'),(region_b,1600,'WOMAN')):
        preview=g.add('PreviewImage',label+' region: bright=editable / dim=protected',(-1900,y))
        preview['size']=[640,500];g.link(region,3,preview,'images')
    note(g,'PROTECTED TWO-PERSON EDIT - configure both regions before generating',
         'The rectangles are user controls, NOT person detectors. Set left/top/width/height as fractions of the original scene (0..1). '
         'Include each target\'s OLD and NEW silhouette, hair, clothing and affected shadows. Context padding is visible to Krea but not automatically edited. '
         'A painted MASK from a LoadImage of this SAME original scene may override each rectangle: paint the edit area in Mask Editor, then connect MASK to edit_mask. '
         'White=edit, black=protect. Exact source dimensions are required. An empty mask raises an error rather than silently editing the wrong area. '
         'The second mask excludes EVERY pixel used by the first, including feathering. Nearby/overlapping bodies require carefully painted masks; '
         'an overly broad man rectangle can prevent the woman from updating. This uses two separate Krea generations plus compositing, not diffusion inpainting. '
         'The final image keeps the ORIGINAL scene dimensions. Unedited pixels and the first edited region are protected before any optional global upscale. '
         'Text targets refer to the local crop, so "the man in this crop" is often clearer than left/right. Each pass has its own fixed seed and rebalancer switch.',2080,1100,1000,540)
    return g.save('Krea_Identity_07_Man_Woman_Protected_Regions_v1_1.json')


def main():
    directed();couple();protected_couple()
    print('Generated three v1.1 graphs; original four graphs were not rewritten.')

if __name__=='__main__':main()
