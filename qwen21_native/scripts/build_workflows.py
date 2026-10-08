#!/usr/bin/env python3
"""Deterministic native-Qwen workflow generator. No downloaded model is needed."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSION="1.1.0"
BODY_LORA="bfs_body_swap_v1.0_qwen_2.1.safetensors"
HEAD_LORA="bfs_head_v1.1_qwen_2.1.safetensors"
CATALOG=[]
# (input sockets, output types, widget names). Required widget order matches the pinned node schemas.
S = {
 "LoadImage":([], ["IMAGE","MASK"], ["image","upload"]),
 "UNETLoader":([], ["MODEL"], ["unet_name","weight_dtype"]),
 "CLIPLoader":([], ["CLIP"], ["clip_name","type","device"]),
 "VAELoader":([], ["VAE"], ["vae_name"]),
 "LoraLoaderModelOnly":([("model","MODEL")], ["MODEL"], ["lora_name","strength_model"]),
 "QwenImage21Cache":([("model","MODEL")], ["MODEL"], ["device","dtype"]),
 "Q21NResizeBudget":([("image","IMAGE")], ["IMAGE","INT","INT"], ["megapixels","allow_upscale"]),
 "Q21NSquareCanvas":([("image","IMAGE")], ["IMAGE"], ["side"]),
 "Q21NEncodeReferences":([("clip","CLIP"),("vae","VAE"),("image_1","IMAGE"),("image_2","IMAGE"),("image_3","IMAGE"),("image_4","IMAGE"),("image_5","IMAGE"),("image_6","IMAGE"),("image_7","IMAGE"),("image_8","IMAGE")], ["CONDITIONING","CONDITIONING","LATENT"], ["prompt"]),
 "Q21NMaskGuard":([("image","IMAGE"),("mask","MASK"),("mask_source","IMAGE"),("original_scene","IMAGE")], ["IMAGE","MASK"], ["label"]),
 "AUSBOSS_NODES_CropForInpaint":([("image","IMAGE"),("mask","MASK")], ["IMAGE","MASK","AUSBOSS_STITCHER"], ["context_factor","blend_pixels","output_multiple","target_width","target_height","mask_grow","mask_blur","invert_mask","context_pixels","target_megapixels","rescale_algorithm","extend_left","extend_right","extend_up","extend_down","keep_inside"]),
 "KSampler":([("model","MODEL"),("positive","CONDITIONING"),("negative","CONDITIONING"),("latent_image","LATENT")], ["LATENT"], ["seed","control_after_generate","steps","cfg","sampler_name","scheduler","denoise"]),
 "VAEDecode":([("samples","LATENT"),("vae","VAE")], ["IMAGE"], []),
 "Q21NOpaqueImage":([("image","IMAGE"),("background","IMAGE")], ["IMAGE"], []),
 "AUSBOSS_NODES_StitchInpaint":([("stitcher","AUSBOSS_STITCHER"),("inpainted","IMAGE")], ["IMAGE","MASK"], ["fix_edge_halo","color_match","seam"]),
 "Q21NAuditPreservation":([("original","IMAGE"),("edited","IMAGE"),("blend_mask","MASK")], ["IMAGE","STRING"], []),
 "Q21NMaskUnion":([("a","MASK"),("b","MASK")], ["MASK"], []),
 "Q21NWriteHandoff":([("image","IMAGE")], ["IMAGE"], ["stage"]),
 "AUSBOSS_NODES_FreeMemory":([("value","*")], ["*"], []),
 "SaveImage":([("images","IMAGE")], [], ["filename_prefix"]),
 "ImageScaleBy":([("image","IMAGE")], ["IMAGE"], ["upscale_method","scale_by"]),
 "PreviewImage":([("images","IMAGE")], [], []),
 "MaskToImage":([("mask","MASK")], ["IMAGE"], []),
 "Note":([], [], ["text"]),
 "SeedVR2LoadDiTModel":([("torch_compile_args","TORCH_COMPILE_ARGS")], ["SEEDVR2_DIT"], ["model","device","blocks_to_swap","swap_io_components","offload_device","cache_model","attention_mode"]),
 "SeedVR2LoadVAEModel":([("torch_compile_args","TORCH_COMPILE_ARGS")], ["SEEDVR2_VAE"], ["model","device","encode_tiled","encode_tile_size","encode_tile_overlap","decode_tiled","decode_tile_size","decode_tile_overlap","tile_debug","offload_device","cache_model"]),
 "SeedVR2VideoUpscaler":([("image","IMAGE"),("dit","SEEDVR2_DIT"),("vae","SEEDVR2_VAE")], ["IMAGE"], ["seed","resolution","max_resolution","batch_size","uniform_batch_size","temporal_overlap","prepend_frames","color_correction","input_noise_scale","latent_noise_scale","offload_device","enable_debug"]),
}

class Graph:
    def __init__(self, title):
        self.title = title
        self.nodes=[]; self.links=[]; self.api={}; self.groups=[]
    def add(self, typ, title, xy, widgets=(), size=None):
        ins, outs, names = S[typ]
        assert len(widgets) == len(names), (typ, len(widgets), names)
        i=len(self.nodes)+1
        if size is None:
            size=[330, max(120, 80 + 26 * max(len(ins),len(widgets)))]
            if typ in ("LoadImage",): size=[330,350]
            if typ=="Q21NEncodeReferences": size=[510,570]
        n={"id":i,"type":typ,"pos":list(xy),"size":size,"flags":{},"order":i-1,"mode":0,
           "inputs":[{"name":name,"type":kind,"link":None} for name,kind in ins],
           "outputs":[{"name":kind.lower(),"type":kind,"links":[],"slot_index":j} for j,kind in enumerate(outs)],
           "properties":{"Node name for S&R":typ},"widgets_values":list(widgets),"title":title}
        self.nodes.append(n)
        if typ!="Note":
            vals={k:v for k,v in zip(names,widgets) if k not in ("control_after_generate","upload")}
            self.api[str(i)]={"class_type":typ,"inputs":vals,"_meta":{"title":title}}
        return i
    def connect(self,a,out,b,name):
        src=self.nodes[a-1]; dst=self.nodes[b-1]
        slot=next(j for j,p in enumerate(dst["inputs"]) if p["name"]==name)
        assert dst["inputs"][slot]["link"] is None
        st=src["outputs"][out]["type"]; dt=dst["inputs"][slot]["type"]
        assert st==dt or "*" in (st,dt), (st,dt,name)
        ident=len(self.links)+1
        self.links.append([ident,a,out,b,slot,dt if st=="*" else st])
        src["outputs"][out]["links"].append(ident); dst["inputs"][slot]["link"]=ident
        self.api[str(b)]["inputs"][name]=[str(a),out]
    def group(self,title,x,y,w,h):
        self.groups.append({"id":len(self.groups)+1,"title":title,"bounding":[x,y,w,h],"color":"#365b75","font_size":24,"flags":{}})
    def save(self,stem):
        self.api = {k:self.api[k] for k in sorted(self.api,key=int)}
        data={"last_node_id":len(self.nodes),"last_link_id":len(self.links),"nodes":self.nodes,"links":self.links,
              "groups":self.groups,"config":{},"extra":{"ds":{"scale":0.6,"offset":[50,50]},"qwen21_native_suite":{"version":"1.1.0","title":self.title}},"version":0.4}
        (ROOT/"workflows"/f"{stem}.json").write_text(json.dumps(data,indent=2)+"\n")
        (ROOT/"api_workflows"/f"{stem}.api.json").write_text(json.dumps(self.api,indent=2)+"\n")


S.update({
 "Q21NBlankCanvas":([], ["IMAGE"], ["width","height"]),
 "Q21NProtectRegion":([("original","IMAGE"),("edited","IMAGE"),("blend_mask","MASK"),("protection","MASK")],["IMAGE","MASK"],[]),
 "Q21NDisjointMasks":([("mask_a","MASK"),("mask_b","MASK")],["MASK","MASK"],[]),
 "Q21NComparePanel":([("before","IMAGE"),("after","IMAGE"),("reference","IMAGE")],["IMAGE"],["label"]),
})
SEED=18071042
COMMON="Preserve the camera viewpoint, perspective, pose, lighting, scene layout and unrelated people and objects from <image1>. Do not copy any reference background, framing or camera angle. Natural photographic texture; no beauty retouching or idealization."
FACE="<image2> is the sole primary facial identity anchor. Reproduce that exact person rather than a lookalike or a blend with the old face. Preserve the referenced person's natural asymmetry and apparent age."
HEAD=f"Edit the centered person's head and face in <image1> to be the exact person in <image2>. {FACE} Fit that identity to the head rotation, expression, gaze and head-to-body scale in <image1>. Keep the body and clothing unchanged. {COMMON}"
WHOLE=f"Replace the centered person in <image1>. {FACE} Use <image3> only for physique and anatomical proportions, not facial identity. Fit that physique to the existing pose, foreshortening and contact points, not the reference's apparent limb lengths. Preserve the scene person's clothing, fitted naturally to the replacement physique. {COMMON}"
BODY="Edit only the centered person's body in <image1> using the physique and proportions evidenced by <image2>. Preserve the existing head and face. Project the referenced physique into the current pose, foreshortening and contact points; do not copy apparent limb lengths or camera scale. Preserve the existing outfit and adapt its fit naturally. "+COMMON
MASK_NOTE="Paint on the MASK layer of the scene loader, not the RGB paint layer. White is editable. Qwen receives the complete context crop, NOT this mask; only AusBoss crop/stitch localizes the final change. Include the full old and intended new silhouette plus any shadows you need to change. Exclude foreground occluders. Do not assume a prompt provides pixel protection. Pixels outside the actual feathered blend footprint are audited."
REF_NOTE="Use real reference photos as the primary evidence. Generated angle/body/expression images are approximations and do not reveal unseen anatomy or prove likeness. One strong real face plus one useful angle is a better starting test than many conflicting images. All wired image loaders must be supplied."


def note(g,title,text,y=-400):
    return g.add("Note",title,(0,y),(text,),[1430,290])


def models(g):
    m=g.add("UNETLoader","BASE Qwen 2.1 BF16 | no LoRA by default",(460,0),("qwen_image_2.1_bf16.safetensors","default"))
    c=g.add("CLIPLoader","Qwen3-VL 8B BF16 | qwen_image",(460,200),("qwen3vl_8b_bf16.safetensors","qwen_image","default"))
    v=g.add("VAELoader","Qwen 2.1 VAE BF16",(460,400),("qwen_image_2.1_vae_bf16.safetensors",))
    return m,c,v


def image(g,title,filename,y=0):
    return g.add("LoadImage",title,(0,y),(filename,"image"))


def refs(g,specs,y=450):
    return [image(g,f"<image{i+2}> {role}",fn,y+i*430) for i,(role,fn) in enumerate(specs)]


def normalized_refs(g,raw,base_y=0,mp=1.0):
    result=[]
    for i,node in enumerate(raw):
        rr=g.add("Q21NResizeBudget",f"Reference {i+2} | <= {mp:g} MP | no detail invented",(850,base_y+600+i*165),(mp,False))
        g.connect(node,0,rr,"image"); result.append((rr,0))
    return result


def render(g,base,ref_outputs,ms,prompt,y=0,steps=40,seed=SEED,lora=None,strength=0.0):
    m,c,v=ms
    model=m
    if lora is not None:
        model=g.add("LoraLoaderModelOnly",f"COMPARISON ONLY | {strength:g}",(1350,y),(lora,strength))
        g.connect(m,0,model,"model")
    cache=g.add("QwenImage21Cache","Lossless KV cache | auto / default",(1350,y+190),("auto","default"))
    g.connect(model,0,cache,"model")
    enc=g.add("Q21NEncodeReferences","EDIT THIS PROMPT | image1=base crop",(1800,y),(prompt,))
    for a,o,k in ((c,0,"clip"),(v,0,"vae"),(base[0],base[1],"image_1")):
        g.connect(a,o,enc,k)
    for i,(a,o) in enumerate(ref_outputs,2):g.connect(a,o,enc,f"image_{i}")
    sam=g.add("KSampler",f"{steps} steps | CFG 1 | Euler/simple | fixed seed",(2400,y),(seed,"fixed",steps,1.0,"euler","simple",1.0),[340,330])
    for a,o,k in ((cache,0,"model"),(enc,0,"positive"),(enc,1,"negative"),(enc,2,"latent_image")):g.connect(a,o,sam,k)
    dec=g.add("VAEDecode","Decode native matching 64-channel latent",(2400,y+390))
    g.connect(sam,0,dec,"samples");g.connect(v,0,dec,"vae")
    opaque=g.add("Q21NOpaqueImage","Flatten generated alpha over source crop",(2400,y+560))
    g.connect(dec,0,opaque,"image");g.connect(*base,opaque,"background")
    return (opaque,0)


def save(g,output,stem,xy=(3440,0),handoff=None):
    node=g.add("SaveImage","Save separate candidate PNG",xy,(f"Qwen21_Native/{stem}/image",))
    g.connect(*output,node,"images")
    if handoff:
        h=g.add("Q21NWriteHandoff",f"Update {handoff}_latest.png | candidate, not auto-approved",(xy[0],xy[1]+180),(handoff,))
        g.connect(*output,h,"image")


def stage(g,scene,mask,raw_refs,ms,prompt,stem,y=0,mp=2.0,context=1.6,blend=16,steps=40,
          handoff=None,seed=SEED,lora=None,strength=0.0,align=None,protection=None,protection_source=None,panel=True,ref_mp=1.0):
    guard=g.add("Q21NMaskGuard","Check mask and source dimensions",(850,y),("Paint the intended edit region",))
    g.connect(*scene,guard,"image");g.connect(*mask,guard,"mask")
    if align:
        g.connect(align[0],0,guard,"mask_source");g.connect(align[1],0,guard,"original_scene")
    crop=g.add("AUSBOSS_NODES_CropForInpaint",f"AusBoss | {mp:g} MP crop | never resize full scene",(1260,y+590),
        (context,blend,32,0,0,0,0.0,False,32,mp,"bicubic",0,0,0,0,True),[365,610])
    g.connect(guard,0,crop,"image");g.connect(guard,1,crop,"mask")
    sized=normalized_refs(g,raw_refs,y,mp=ref_mp)
    result=render(g,(crop,0),sized,ms,prompt,y,steps,seed,lora,strength)
    stitch=g.add("AUSBOSS_NODES_StitchInpaint","Original-size stitch | color match OFF",(2850,y),(False,0.0,"classic"))
    g.connect(crop,2,stitch,"stitcher");g.connect(*result,stitch,"inpainted")
    final=(stitch,0);footprint=(stitch,1)
    if protection is not None:
        if protection_source:
            pg=g.add("Q21NMaskGuard","Verify hard-protection mask comes from same original",(2850,y+220),("Paint head/face to protect",))
            g.connect(*scene,pg,"image");g.connect(*protection,pg,"mask")
            g.connect(protection_source[0],0,pg,"mask_source");g.connect(protection_source[1],0,pg,"original_scene")
            protection=(pg,1)
        lock=g.add("Q21NProtectRegion","HARD PIXEL LOCK overrides edits and feather",(2850,y+420))
        for val,name in ((scene,"original"),(final,"edited"),(footprint,"blend_mask"),(protection,"protection")):g.connect(*val,lock,name)
        final=(lock,0);footprint=(lock,1)
    audit=g.add("Q21NAuditPreservation","Assert exact protected RGB pixels",(2850,y+650))
    for val,name in ((scene,"original"),(final,"edited"),(footprint,"blend_mask")):g.connect(*val,audit,name)
    output=(audit,0)
    save(g,output,stem,(3440,y),handoff)
    preview=g.add("PreviewImage","Inspect generated crop at native resolution",(3440,y+400),(),[350,340]);g.connect(*result,preview,"images")
    if panel:
        cp=g.add("Q21NComparePanel","RAW crop / generated crop / reference (before protection)",(3850,y),("RAW: "+stem,))
        g.connect(crop,0,cp,"before");g.connect(*result,cp,"after")
        if sized:g.connect(*sized[0],cp,"reference")
        save(g,(cp,0),stem+"/comparison",(4260,y))
    return output,footprint


def finish(g,stem,purpose,category="Native edits",needs="identity",hints=""):
    g.group("YOUR SOURCES / PAINTED MASKS",-20,-60,425,max(900,max((n['pos'][1]+n['size'][1] for n in g.nodes if n['type']=='LoadImage'),default=600)+100))
    g.group("BASE BF16 WEIGHTS",440,-60,370,690)
    g.save(stem)
    CATALOG.append({"file":stem+".json","category":category,"purpose":purpose,"requires":needs,"hint":hints,
        "reference_loaders":len([n for n in g.nodes if n['type']=='LoadImage'])})


def single(stem,purpose,prompt,specs,category="Native edits",filename="scene.png",mp=2.0,context=1.6,steps=40,handoff=None,protect_head=False,hints=""):
    g=Graph(purpose);note(g,stem+" | NATIVE QWEN, NO LoRA",purpose+"\n\n"+MASK_NOTE+"\n\n"+REF_NOTE+"\n"+hints)
    src=image(g,"<image1> BASE SCENE | paint edit mask",filename)
    raw=refs(g,specs);protect=None;protect_source=None
    if protect_head:
        p=image(g,"SAME SCENE | paint HEAD to PROTECT, not edit","scene.png",450+len(raw)*430)
        protect=(p,1);protect_source=(p,src)
    ms=models(g)
    stage(g,(src,0),(src,1),raw,ms,prompt,stem,mp=mp,context=context,steps=steps,handoff=handoff,protection=protect,protection_source=protect_source)
    finish(g,stem,purpose,category,hints=hints)


FACE_REF=("PRIMARY FACE identity","face_primary.png")
ANGLE_REF=("SAME PERSON supporting angle","face_angle.png")
FACE3_REF=("SAME PERSON second supporting view","face_support.png")
BODY_REF=("BODY geometry only","body_front.png")
SIDE_REF=("BODY supporting view","body_side.png")
GARMENT_REF=("WARDROBE only","garment.png")
EXPR_REF=("EXPRESSION only, same subject","expression.png")


def combined():
    stem="23_Sequential_Body_Then_Head_Native"
    g=Graph(stem);note(g,"NATIVE BODY > NATIVE HEAD | separate approval is safer",MASK_NOTE+"\nLoad the identical original scene in both scene loaders. Body mask excludes the head; head mask also protects the original head during pass 1. Pass 2 uses the generated body result. If the intended head moves beyond the prepainted mask, use 05 then 08 separately. Handoffs are generated candidates, not accepted truth.")
    scene=image(g,"ORIGINAL scene | BODY edit mask","scene.png")
    body=image(g,"BODY reference only","body_front.png",450)
    headmask=image(g,"SAME original scene | HEAD edit and protection mask","scene.png",900)
    face=image(g,"PRIMARY real face identity","face_primary.png",1350)
    angle=image(g,"Supporting genuine face angle","face_angle.png",1800)
    ms=models(g)
    a,am=stage(g,(scene,0),(scene,1),[body],ms,BODY,stem+"/body",handoff="body",protection=(headmask,1),protection_source=(headmask,scene))
    free=g.add("AUSBOSS_NODES_FreeMemory","Release cached models after body result",(460,2050));g.connect(*a,free,"value")
    b,bm=stage(g,(free,0),(headmask,1),[face,angle],ms,HEAD+" <image3> only clarifies the same person's identity from another view; do not blend faces.",stem+"/head",y=2250,handoff="head",align=(headmask,scene))
    u=g.add("Q21NMaskUnion","Union of both effective blend footprints",(2850,3650));g.connect(*am,u,"a");g.connect(*bm,u,"b")
    audit=g.add("Q21NAuditPreservation","Final audit against original scene",(3440,3650));g.connect(scene,0,audit,"original");g.connect(*b,audit,"edited");g.connect(u,0,audit,"blend_mask")
    save(g,(audit,0),stem+"/final",(3850,3650),"final")
    finish(g,stem,"Two native passes with hard head protection during body editing.","Sequential")


def two_people(full=False,one_pass=False):
    stem="26_Two_People_One_Pass_EXPERIMENTAL" if one_pass else "25_Two_Full_People_Sequential" if full else "24_Two_Heads_Sequential"
    g=Graph(stem);note(g,stem,MASK_NOTE+"\nPerson A and Person B get disjoint masks from the SAME original scene. Do not overlap masks at contact points. Sequential mode keeps reference groups separate and hard-protects the first edit during the second. Identify people by visible position, not assumed gender. One-pass mode is an experiment and can mix identities; prefer sequential.")
    a=image(g,"ORIGINAL scene | PERSON A mask","scene.png")
    b=image(g,"SAME original scene | PERSON B mask","scene.png",450)
    fa=image(g,"PERSON A primary face","person_a_face.png",900)
    fb=image(g,"PERSON B primary face","person_b_face.png",1350)
    ba=image(g,"PERSON A body","person_a_body.png",1800) if full else None
    bb=image(g,"PERSON B body","person_b_body.png",2250) if full else None
    ms=models(g)
    dis=g.add("Q21NDisjointMasks","Reject ambiguous overlapping subject masks",(460,800));g.connect(a,1,dis,"mask_a");g.connect(b,1,dis,"mask_b")
    align=g.add("Q21NMaskGuard","Confirm both masks use identical original RGB",(460,1000),("Paint person B",))
    g.connect(a,0,align,"image");g.connect(dis,1,align,"mask");g.connect(b,0,align,"mask_source");g.connect(a,0,align,"original_scene")
    if one_pass:
        union=g.add("Q21NMaskUnion","Union of both subject masks",(460,1200));g.connect(dis,0,union,"a");g.connect(align,1,union,"b")
        prompt="Edit the two people in <image1>. The person on the viewer's left gets facial identity ONLY from <image2> and physique ONLY from <image3>. The person on the viewer's right gets facial identity ONLY from <image4> and physique ONLY from <image5>. Do not mix reference groups. Preserve their individual poses, positions, clothing, head angles, expression and contact points. "+COMMON
        stage(g,(a,0),(union,0),[fa,ba,fb,bb],ms,prompt,stem)
    else:
        prompt=WHOLE if full else HEAD
        aout,af=stage(g,(a,0),(dis,0),[fa,ba] if full else [fa],ms,prompt,stem+"/person_a")
        free=g.add("AUSBOSS_NODES_FreeMemory","Order passes and unload cached models",(460,2650));g.connect(*aout,free,"value")
        bout,bf=stage(g,(free,0),(align,1),[fb,bb] if full else [fb],ms,prompt,stem+"/person_b",y=2850,protection=af)
        u=g.add("Q21NMaskUnion","Combined actual blend area",(2850,4300));g.connect(*af,u,"a");g.connect(*bf,u,"b")
        audit=g.add("Q21NAuditPreservation","Protect remaining original scene",(3440,4300));g.connect(a,0,audit,"original");g.connect(*bout,audit,"edited");g.connect(u,0,audit,"blend_mask")
        save(g,(audit,0),stem+"/final",(3850,4300),"final")
    finish(g,stem,"Two-person replacement with explicit identity ownership.","Multi-person",hints="Change left/right instructions for one-pass mode. Serial edits are the recommended baseline.")


def free_scene(prep=False):
    stem="28_Source_Prep_Fitted_Clothing_CANDIDATE" if prep else "27_Described_Scene_Face_Plus_Body"
    g=Graph(stem);note(g,stem,REF_NOTE+"\nThis graph generates the full frame, not a protected edit. Image1 is a blank canvas defining aspect ratio, image2 is face identity, image3 is body evidence. Edit the prompt for the setting. Keep output around 2 MP initially. The source-prep output is synthetic, not recovered body evidence.")
    face=image(g,"<image2> PRIMARY genuine face","face_primary.png")
    body=image(g,"<image3> Genuine body evidence","body_front.png",450)
    ms=models(g)
    base=g.add("Q21NBlankCanvas","<image1> blank canvas ONLY | editable dimensions",(850,0),(1408,1408) if prep else (1152,1728))
    rr=normalized_refs(g,[face,body])
    prompt="Create one photorealistic full-body reference photograph on the canvas in <image1>. Use <image2> only for the exact person's face and hair, and <image3> only for observed physique. One person standing front-facing, complete head, hands and feet visible, arms slightly away from torso, neutral expression, plain light-gray background and soft even lighting. Plain fitted sleeveless top and fitted mid-thigh athletic shorts without compression, padding or body reshaping. Keep bare feet only if supported by real source evidence; otherwise retain simple footwear. Do not slim, add muscle, exaggerate curves or guess concealed anatomy with false certainty. Output one view, not a collage." if prep else "Create one realistic photograph using <image1> only as a blank canvas for size and aspect ratio. <image2> is the sole face and hair identity anchor; <image3> controls only physique and body proportions. Show that exact person standing on a quiet tree-lined sidewalk in soft afternoon light, relaxed natural posture, wearing a plain shirt, fitted trousers and low-profile shoes. Full body visible, eye-level natural photographic perspective. Do not copy the reference backgrounds, poses or camera angles. Preserve natural asymmetry, apparent age and realistic skin and fabric."
    out=render(g,(base,0),rr,ms,prompt)
    save(g,out,stem,(2850,0),"reference" if prep else "final")
    finish(g,stem,"Generate a new scene from separately assigned face and body references." if not prep else "Optional synthetic body-reference normalization, never ground truth.","Source and new scene")


def crop_reference():
    stem="29_Reference_Crop_Export_NO_GENERATION"
    g=Graph(stem);note(g,stem,"NO MODEL LOADED. Paint around the useful genuine face, hand, arm or body detail. AusBoss crops it with context and at most 1 MP. It does not invent a new angle or normalize anatomy. Save this genuine crop before trying generated reference preparation.")
    src=image(g,"GENUINE photo | mask useful reference region","reference_original.png")
    guard=g.add("Q21NMaskGuard","Require a reference crop selection",(460,0),("Paint reference crop region",));g.connect(src,0,guard,"image");g.connect(src,1,guard,"mask")
    crop=g.add("AUSBOSS_NODES_CropForInpaint","Crop at ORIGINAL resolution; no generative processing",(850,0),(1.15,0,32,0,0,0,0.0,False,0,0.0,"bicubic",0,0,0,0,True))
    g.connect(guard,0,crop,"image");g.connect(guard,1,crop,"mask")
    resize=g.add("Q21NResizeBudget","Cap reference at 1 MP, do not upscale",(1260,0),(1.0,False));g.connect(crop,0,resize,"image")
    save(g,(resize,0),stem,(1720,0),"reference")
    finish(g,stem,"Extract genuine reference detail without a generative model.","Source and new scene")


def compare(bfs=False,body=False):
    stem="91_AB_Body_Native_vs_BFS" if body else "90_AB_Head_Native_vs_BFS" if bfs else "32_Head_Three_Seed_Comparison"
    g=Graph(stem);note(g,stem,MASK_NOTE+"\nCONTROLLED COMPARISON. All branches use identical references, crop settings, prompt and sampling settings. LoRA comparison changes only adapter strength, with seed fixed. Seed comparison changes only seed. Keep each output separately. No branch overwrites a handoff or declares a winner. BFS is optional and requires ENABLE_BFS_COMPARISONS=1.")
    scene=image(g,"BASE SCENE | paint head mask" if not body else "BASE SCENE | paint body mask","scene.png")
    ref=image(g,"PRIMARY body reference" if body else "PRIMARY genuine face reference","body_front.png" if body else "face_primary.png",450)
    ms=models(g)
    strengths=[None,0.25,0.5,0.75,1.0] if bfs else [None,None,None]
    prompt=BODY if body else HEAD
    for i,strength in enumerate(strengths):
        label="native" if strength is None else f"bfs_{strength:g}"
        if not bfs:label=f"seed_{SEED+i}"
        stage(g,(scene,0),(scene,1),[ref],ms,prompt,stem+"/"+label,y=i*1850,
            seed=SEED if bfs else SEED+i,lora=(BODY_LORA if body else HEAD_LORA) if strength is not None else None,strength=strength or 0.0,ref_mp=0.59 if body else 1.0)
    finish(g,stem,"Matched native/adapter comparison; not an assertion of superior likeness." if bfs else "Three seeds with every other setting held constant.","Comparisons",needs="bfs_optional" if bfs else "identity")


def upscaling(local=False):
    stem="35_SeedVR2_Local_Protected_Finish" if local else "34_SeedVR2_Full_Frame_7B_FP16"
    g=Graph(stem);note(g,stem,"OPTIONAL UPSCALE PROFILE. Requires SeedVR2 7B FP16 and its VAE. Approve the native image first; this synthesizes detail and can change identity. Noise and compilation are OFF. Short edge 2048, long edge capped at 4096. It can downsize a larger image, so skip full-frame enhancement when already large enough. Local mode stitches the enhanced patch into the unchanged native-size scene; it does not increase the whole image dimensions. Enable tiled VAE only if needed for memory.")
    src=image(g,"ACCEPTED native image"+(" | mask enhancement region" if local else ""),"qwen21_native_handoff/head_latest.png")
    base=(src,0);crop=None
    if local:
        guard=g.add("Q21NMaskGuard","Choose local enhancement region",(460,0),("Paint enhancement region",));g.connect(src,0,guard,"image");g.connect(src,1,guard,"mask")
        crop=g.add("AUSBOSS_NODES_CropForInpaint","Local patch + context | 1 MP",(850,0),(1.5,16,32,0,0,0,0.0,False,32,1.0,"bicubic",0,0,0,0,True));g.connect(guard,0,crop,"image");g.connect(guard,1,crop,"mask");base=(crop,0)
    free=g.add("AUSBOSS_NODES_FreeMemory","Release Qwen before SeedVR2",(1280,0));g.connect(*base,free,"value")
    dit=g.add("SeedVR2LoadDiTModel","7B FP16 | SDPA | CPU offload",(1280,220),("seedvr2_ema_7b_fp16.safetensors","cuda:0",0,False,"cpu",False,"sdpa"))
    vae=g.add("SeedVR2LoadVAEModel","FP16 VAE | untiled quality baseline",(1280,610),("ema_vae_fp16.safetensors","cuda:0",False,1024,128,False,1024,128,"false","cpu",False))
    up=g.add("SeedVR2VideoUpscaler","Still image | batch 1 | no added noise",(1790,0),(SEED,2048,4096,1,False,0,0,"lab",0.0,0.0,"cpu",False))
    g.connect(free,0,up,"image");g.connect(dit,0,up,"dit");g.connect(vae,0,up,"vae")
    out=(up,0)
    if local:
        st=g.add("AUSBOSS_NODES_StitchInpaint","Stitch at original scene size",(2340,0),(False,0.0,"classic"));g.connect(crop,2,st,"stitcher");g.connect(up,0,st,"inpainted")
        au=g.add("Q21NAuditPreservation","Assert unchanged protected scene",(2800,0));g.connect(src,0,au,"original");g.connect(st,0,au,"edited");g.connect(st,1,au,"blend_mask");out=(au,0)
    save(g,out,stem,(3250,0))
    finish(g,stem,"Optional generative enhancement of a local patch." if local else "Optional full-frame generative upscaling.","Finish",needs="upscale")


def build_all():
    CATALOG.clear()
    for f in ('workflows','api_workflows'):(ROOT/f).mkdir(exist_ok=True)
    single("00_START_HERE_Scene_Face_Body","One masked native replacement with separate facial identity and physique.",WHOLE,[FACE_REF,BODY_REF],handoff="final")
    single("01_Person_Multi_Reference","Two face and two body references with explicit ownership.",WHOLE.replace('<image3> only for physique','<image4> only for physique')+" <image3> supports the same face identity from another angle; <image5> supports the same body anatomy. The primary face remains <image2>.",[FACE_REF,ANGLE_REF,BODY_REF,SIDE_REF],handoff="final")
    single("02_Person_Keep_Scene_Wardrobe","Identity and physique replacement without importing reference clothing.",WHOLE+" Do not import any garment from the reference photos; retain the exact design, fabric and color worn in the scene.",[FACE_REF,BODY_REF],handoff="final")
    single("03_Person_Transfer_Body_Wardrobe","Transfer the body reference's clothing with separately anchored face.",WHOLE.replace("Preserve the scene person's clothing, fitted naturally to the replacement physique.","Use the clothing and footwear shown in <image3>, with their actual design and fit adapted to the scene pose."),[FACE_REF,("BODY and WARDROBE source","body_wardrobe.png")],handoff="final")
    single("04_Person_Separate_Wardrobe_Reference","Separate scene, face, physique and wardrobe sources.",WHOLE.replace("Preserve the scene person's clothing, fitted naturally to the replacement physique.","Take the outfit only from <image4>, fit it to the referenced physique, and ignore the garment model's identity."),[FACE_REF,BODY_REF,GARMENT_REF],handoff="final")
    single("05_Body_Only_Hard_Protect_Head","Body edit with an explicit pixel-locked head protection mask.",BODY,[BODY_REF],handoff="body",protect_head=True,hints="The second scene loader MUST use the identical original and a painted head/face protection mask. That region is restored exactly after stitching.")
    single("06_Body_And_Wardrobe_Hard_Protect_Head","Body plus reference outfit with the original head pixel-locked.",BODY.replace("Preserve the existing outfit and adapt its fit naturally.","Transfer the outfit from <image2> and adapt it to the current pose."),[("BODY plus desired clothes","body_wardrobe.png")],handoff="body",protect_head=True)
    single("07_Head_One_Reference","Minimal native head replacement from one genuine face anchor.",HEAD,[FACE_REF],category="Identity",handoff="head",context=1.8)
    single("08_Head_Three_Identity_References","Head replacement with one primary face and two supporting views.",HEAD+" <image3> and <image4> only clarify the SAME person's appearance from additional views. Do not average identities or copy reference pose.",[FACE_REF,ANGLE_REF,FACE3_REF],category="Identity",handoff="head",context=1.8)
    single("09_Face_Only_Keep_Hair","Face edit while preserving the existing hairstyle and head outline.",HEAD.replace("head and face","face").replace("Keep the body and clothing unchanged.","Keep the existing hair, hairline, ears and outer head silhouette unchanged. Change only the interior face. <image3> only clarifies the same identity from an additional view."),[FACE_REF,ANGLE_REF],category="Identity",handoff="head",hints="Paint the face only, leaving hair outside the feather. Image3 is supporting identity evidence only.")
    single("10_Head_Hair_And_Hairline","Complete head, hair and hairline replacement from identity references.",HEAD+" Use <image3> only as supporting evidence for the same person's hair, hairline and side profile. Include natural hair volume without increasing the head size.",[FACE_REF,ANGLE_REF],category="Identity",handoff="head",hints="Mask the complete old and new hair silhouette, ears and required neck transition.")
    single("11_Profile_Angle_Matched_Identity","Primary identity plus an actual angle-matched profile reference.",HEAD+" <image3> shows the same person at a useful side angle. Use it to resolve projection and occlusion, while <image2> remains the primary identity anchor. Match <image1>'s exact head rotation.",[FACE_REF,("GENUINE angle-matched profile","face_profile.png")],category="Identity",handoff="head")
    single("12_Rescue_Likeness_From_Originals","Replace a drifted face using original photographs, not the drifted render.",HEAD+" The face currently in <image1> has drifted and is NOT an identity reference. Correct that identity using <image2>; <image3> only clarifies the same real person. Keep the existing head angle and expression, not the incorrect facial geometry.",[FACE_REF,ANGLE_REF],category="Identity",handoff="head")
    single("13_Expression_Image2_Only","Transfer expression while taking all likeness from the base image.","Image1 is the sole identity and appearance anchor. Edit only the centered person's facial expression in <image1>. Use <image2> solely for expression intensity and facial muscle movement, adapted to <image1>'s own anatomy. Preserve the base person's facial structure, natural asymmetry, skin, age, hair, head position, gaze direction, body and clothing. Do not copy <image2>'s physical features, head angle, lighting or perspective. "+COMMON,[EXPR_REF],category="Expression",handoff="expression",context=1.8)
    single("14_Identity_And_Expression_Separate_Refs","Face identity from image2, expression only from image3.",HEAD.replace("expression, gaze and head-to-body scale","gaze and head-to-body scale")+" Use <image3> ONLY for expression and facial muscle movement, adapted to <image2>'s identity. Do not take anatomy, skin, hair, head angle or likeness from the expression reference.",[FACE_REF,EXPR_REF],category="Expression",handoff="expression")
    single("15_Expression_Text_Only","Expression-only editing without additional reference images.","Edit only the centered person's facial expression in <image1>: a subtle natural closed-mouth smile. Preserve the exact identity, face structure, asymmetry, skin texture, apparent age, head angle, gaze direction, hair, body, pose, lighting, clothing and background. Adapt only the facial muscles required for the smile. Do not beautify.",[],category="Expression",handoff="expression",hints="Replace the smile description with your desired expression.")
    single("16_Local_Repair_No_Reference","Small cleanup using only the accepted image.","Use <image1> as the base photograph. Repair the visible local blending defect on the centered subject. Preserve the exact established identity, expression, head angle, pose, clothing and lighting. Do not redesign already-correct anatomy. Smooth only the inconsistent transition, retain normal skin texture and correct detail.",[],category="Repair",handoff="repair",hints="Replace the generic defect description with exactly what needs repair.")
    single("17_Local_Repair_With_Anatomy_Reference","No-LoRA counterpart to the previous local repair workflow.","Repair the local anatomical or blending error on the centered subject in <image1>. Use <image2> only as evidence for the relevant anatomy of this person. Preserve the established identity, clothing, pose and scene. Project anatomy into the base perspective rather than copying the reference camera angle. Keep already-correct details unchanged.",[("RELEVANT genuine anatomical detail","anatomy_reference.png")],category="Repair",handoff="repair")
    single("18_Hand_And_Wrist_Repair","Local hand anatomy and wrist connection correction.","Correct the centered hand and its connection to the wrist in <image1>. Preserve its gesture, contact points, arm position and foreshortening. Use <image2> only for this person's hand proportions and observed details, not its gesture or scale. Preserve fingers hidden by objects rather than adding visible fingers through occluders. Retain skin texture, scene lighting and all correct surrounding details.",[("GENUINE hand reference","hand_reference.png")],category="Repair",handoff="repair",context=2.0)
    single("19_Limb_Perspective_Repair","Correct arm or leg proportions in the existing camera perspective.","Correct the centered limb's anatomy and connection to the body in <image1>. Use <image2> only for the person's true limb proportions and thickness. Project those proportions into <image1>'s existing pose and foreshortening. Do not copy the reference's apparent length, perspective or camera scale. Preserve identity, contact points, clothing, lighting and correct surrounding anatomy.",[("GENUINE limb/body evidence","limb_reference.png")],category="Repair",handoff="repair",context=2.0)
    single("20_Neck_Head_Seam_Repair","Repair the neck transition without intentionally replacing the face.","Repair only the transition between neck, jaw and shoulders in <image1> so they form one coherent person. Preserve the existing face identity, expression, head rotation and hairstyle. Use <image2> only for neck and shoulder proportion evidence. Match the scene light and skin transitions without smoothing away normal texture. Do not replace the head again.",[("GENUINE head and shoulders","neck_reference.png")],category="Repair",handoff="repair",context=2.0,hints="Mask neck and seam only; keep the accepted face outside the feather.")
    single("21_Skin_Texture_Light_Touch","Conservative texture repair from an original close-up.","Reduce the artificial waxy texture on the centered person's face in <image1>. Use <image2> only for authentic texture and observed skin details of this same person. Preserve exact face shape, expression, age, asymmetry, skin tone, existing marks, light and shadow. Do not enlarge pores, invent freckles, change makeup, sharpen edges aggressively or beautify. Keep already-natural texture unchanged.",[("GENUINE unretouched skin/face reference","skin_reference.png")],category="Repair",handoff="repair",hints="Generated pores and microdetail are not recovered evidence; compare against the original.")
    single("22_Wardrobe_Only_Keep_Person","Change clothing while retaining the existing person and pose.","Keep the exact person, face, hair, physique, pose and skin from <image1>. Replace only the visible outfit using the garment from <image2>. Ignore the clothing model's body and identity. Adapt garment fit, folds, occlusions and shadows to the existing body and scene perspective. Do not change body proportions. "+COMMON,[GARMENT_REF],category="Wardrobe",handoff="wardrobe")
    combined();two_people();two_people(full=True);two_people(full=True,one_pass=True)
    free_scene();free_scene(prep=True);crop_reference()
    single("30_Head_High_Detail_4MP_50Steps","Higher-resolution head-crop comparison, not an automatic quality upgrade.",HEAD+" <image3> provides supporting genuine angle evidence only.",[FACE_REF,ANGLE_REF],category="Quality tests",mp=4.0,steps=50,context=1.8,handoff="head",hints="More memory and time. Compare likeness with 07/08 before preferring this result.")
    single("31_Head_Lean_1MP_BF16","Lower crop-memory variant retaining the same BF16 weights.",HEAD,[FACE_REF],category="Quality tests",mp=1.0,context=1.8,handoff="head")
    compare()
    g=Graph("33_Final_Lanczos_No_Model");note(g,"NO GENERATIVE MODEL","Non-generative enlargement only. 2x doubles width and height. It interpolates pixels and does not recover unseen detail or change facial geometry through synthesis. Skip if the native original-size composite is already sufficient.")
    im=image(g,"ACCEPTED native image","qwen21_native_handoff/head_latest.png")
    up=g.add("ImageScaleBy","Lanczos 2x | editable factor",(500,0),("lanczos",2.0));g.connect(im,0,up,"image")
    save(g,(up,0),"33_Final_Lanczos_No_Model",(1000,0));finish(g,"33_Final_Lanczos_No_Model","Non-generative enlargement without any added model.","Finish")
    upscaling();upscaling(local=True);compare(bfs=True);compare(bfs=True,body=True)
    (ROOT/'config/workflow_catalog.json').write_text(json.dumps({"version":VERSION,"workflows":CATALOG},indent=2)+'\n')
    print(f"Generated {len(CATALOG)} UI workflows and matching API graphs.")

if __name__=='__main__':build_all()
