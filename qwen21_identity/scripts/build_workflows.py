#!/usr/bin/env python3
"""Generate editable ComfyUI UI graphs and matching API graphs from one definition."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BODY_LORA = "bfs_body_swap_v1.0_qwen_2.1.safetensors"
HEAD_LORA = "bfs_head_v1.1_qwen_2.1.safetensors"
BODY_PROMPT = """body_swap: Use <image1> as the base scene. Replace only the person centered in this crop with the person from <image2>. Preserve the clothing, physique, body shape and anatomical proportions from <image2>. Retain the pose, camera viewpoint, head direction, gaze, expression, contact points, lighting and environment of <image1>. Project the reference physique into the scene's perspective; do not copy the reference camera angle, apparent limb lengths or apparent body scale. Keep surrounding people and foreground objects unchanged. Photographic realism, natural skin and fabric texture."""
HEAD_PROMPT = """head_swap: Use <image1> as the base. Replace the centered person's complete head with the identity from <image2>, preserving that reference's face shape, facial proportions, hairline, hair, eye color, nose structure, lips, jaw and visible distinguishing features. Follow the head rotation, gaze direction and expression in <image1>, not the reference portrait's neutral pose. Match the scene's lighting, perspective and believable head-to-body scale. Preserve the existing body, clothing, environment and other people. Natural photographic skin texture without beauty smoothing or exaggerated sharpening."""
PREP_PROMPT = """Create one square, full-body studio reference photograph of the same person. <image1> controls physique, body proportions and clothing. <image2> controls face identity, hairline and hair. Keep observed anatomy and distinctive facial features, without slimming, beautifying, aging or athletic enhancement. Standing upright facing the camera, weight balanced, feet visible and uncropped, arms relaxed with a small gap from the torso, hands naturally visible. Level camera at mid-torso height, neutral perspective. Plain light gray seamless background, soft even light. The person occupies most of the image height with a small margin above the head and below the feet. One person and one view, no collage or text. Do not invent unseen anatomy when the source does not show it; use a complete body source."""
REPAIR_PROMPT = """Use <image1> as the base crop. Repair the visible anatomical or blending error on the centered subject while preserving the established identity, pose, clothing and scene lighting. Use <image2> only to check the relevant identity or anatomy, never to copy its camera perspective. Keep all already-correct details unchanged. Natural photographic detail and believable contact shadows."""

# (input sockets, output types, widget names). Required widget order matches the pinned node schemas.
S = {
 "LoadImage":([], ["IMAGE","MASK"], ["image","upload"]),
 "UNETLoader":([], ["MODEL"], ["unet_name","weight_dtype"]),
 "CLIPLoader":([], ["CLIP"], ["clip_name","type","device"]),
 "VAELoader":([], ["VAE"], ["vae_name"]),
 "LoraLoaderModelOnly":([("model","MODEL")], ["MODEL"], ["lora_name","strength_model"]),
 "QwenImage21Cache":([("model","MODEL")], ["MODEL"], ["device","dtype"]),
 "Q21ResizeBudget":([("image","IMAGE")], ["IMAGE","INT","INT"], ["megapixels","allow_upscale"]),
 "Q21SquareCanvas":([("image","IMAGE")], ["IMAGE"], ["side"]),
 "Q21EncodeReferences":([("clip","CLIP"),("vae","VAE"),("image_1","IMAGE"),("image_2","IMAGE"),("image_3","IMAGE")], ["CONDITIONING","CONDITIONING","LATENT"], ["prompt"]),
 "Q21MaskGuard":([("image","IMAGE"),("mask","MASK"),("mask_source","IMAGE"),("original_scene","IMAGE")], ["IMAGE","MASK"], ["label"]),
 "AUSBOSS_NODES_CropForInpaint":([("image","IMAGE"),("mask","MASK")], ["IMAGE","MASK","AUSBOSS_STITCHER"], ["context_factor","blend_pixels","output_multiple","target_width","target_height","mask_grow","mask_blur","invert_mask","context_pixels","target_megapixels","rescale_algorithm","extend_left","extend_right","extend_up","extend_down","keep_inside"]),
 "KSampler":([("model","MODEL"),("positive","CONDITIONING"),("negative","CONDITIONING"),("latent_image","LATENT")], ["LATENT"], ["seed","control_after_generate","steps","cfg","sampler_name","scheduler","denoise"]),
 "VAEDecode":([("samples","LATENT"),("vae","VAE")], ["IMAGE"], []),
 "Q21OpaqueImage":([("image","IMAGE"),("background","IMAGE")], ["IMAGE"], []),
 "AUSBOSS_NODES_StitchInpaint":([("stitcher","AUSBOSS_STITCHER"),("inpainted","IMAGE")], ["IMAGE","MASK"], ["fix_edge_halo","color_match","seam"]),
 "Q21AuditPreservation":([("original","IMAGE"),("edited","IMAGE"),("blend_mask","MASK")], ["IMAGE","STRING"], []),
 "Q21MaskUnion":([("a","MASK"),("b","MASK")], ["MASK"], []),
 "Q21WriteHandoff":([("image","IMAGE")], ["IMAGE"], ["stage"]),
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
            if typ=="Q21EncodeReferences": size=[430,400]
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
        data={"last_node_id":len(self.nodes),"last_link_id":len(self.links),"nodes":self.nodes,"links":self.links,
              "groups":self.groups,"config":{},"extra":{"ds":{"scale":0.6,"offset":[50,50]},"qwen21_identity_kit":{"version":"1.0.0","title":self.title}},"version":0.4}
        (ROOT/"workflows"/f"{stem}.json").write_text(json.dumps(data,indent=2)+"\n")
        (ROOT/"api_workflows"/f"{stem}.api.json").write_text(json.dumps(self.api,indent=2)+"\n")


def models(g, y=0):
    m=g.add("UNETLoader","BF16 Qwen 2.1 ONLY",(450,y),("qwen_image_2.1_bf16.safetensors","default"))
    c=g.add("CLIPLoader","Matching Qwen3-VL 8B | type=qwen_image",(450,y+190),("qwen3vl_8b_bf16.safetensors","qwen_image","default"))
    v=g.add("VAELoader","Matching Qwen 2.1 VAE",(450,y+390),("qwen_image_2.1_vae_bf16.safetensors",))
    return m,c,v


def stage(g, scene, mask_node, ref, ms, y=0, head=False, original_for_mask=None, name="body", repair=False):
    m,c,v=ms
    guard=g.add("Q21MaskGuard",f"Check {name} mask",(850,y),(f"Paint {name} region",))
    g.connect(scene,0,guard,"image");g.connect(mask_node,1,guard,"mask")
    if original_for_mask is not None:
        g.connect(mask_node,0,guard,"mask_source");g.connect(original_for_mask,0,guard,"original_scene")
    crop=g.add("AUSBOSS_NODES_CropForInpaint",f"{name.upper()}: context crop | 2 MP | Multiple 32",(1240,y),
        (1.8 if head else 1.35,16 if head else 24,32,0,0,0,0.0,False,32,2.0,"bicubic",0,0,0,0,True),[360,590])
    g.connect(guard,0,crop,"image");g.connect(guard,1,crop,"mask")
    refsize=g.add("Q21ResizeBudget","Face reference <= 1 MP" if head else "Body reference <= 0.59 MP",(1240,y+650),(1.0 if head else 0.59,False))
    g.connect(ref,0,refsize,"image")
    model=m
    if not repair:
        l=g.add("LoraLoaderModelOnly",f"{name.upper()} adapter only (separate base branch)",(1690,y),(HEAD_LORA if head else BODY_LORA,1.0))
        g.connect(m,0,l,"model");model=l
    cache=g.add("QwenImage21Cache","Lossless KV cache | no quantization",(1690,y+190),("auto","default"))
    g.connect(model,0,cache,"model")
    enc=g.add("Q21EncodeReferences",f"{name.upper()} PROMPT | image1=crop, image2=reference",(1690,y+380),
        (REPAIR_PROMPT if repair else HEAD_PROMPT if head else BODY_PROMPT,))
    g.connect(c,0,enc,"clip");g.connect(v,0,enc,"vae");g.connect(crop,0,enc,"image_1");g.connect(refsize,0,enc,"image_2")
    sample=g.add("KSampler","40 steps | CFG 1 | Euler/simple | fixed seed",(2200,y),(18071001+(1 if head else 0),"fixed",40,1.0,"euler","simple",1.0),[350,320])
    for src,out,target in ((cache,0,"model"),(enc,0,"positive"),(enc,1,"negative"),(enc,2,"latent_image")):
        g.connect(src,out,sample,target)
    dec=g.add("VAEDecode","Decode matching Qwen latent",(2200,y+380))
    g.connect(sample,0,dec,"samples");g.connect(v,0,dec,"vae")
    op=g.add("Q21OpaqueImage","Flatten any generated alpha over crop",(2200,y+550))
    g.connect(dec,0,op,"image");g.connect(crop,0,op,"background")
    stitch=g.add("AUSBOSS_NODES_StitchInpaint","Stitch into untouched original | tone match OFF",(2640,y),(False,0.0,"classic"))
    g.connect(crop,2,stitch,"stitcher");g.connect(op,0,stitch,"inpainted")
    audit=g.add("Q21AuditPreservation","Check pixels outside actual feathered blend",(2640,y+210))
    g.connect(scene,0,audit,"original");g.connect(stitch,0,audit,"edited");g.connect(stitch,1,audit,"blend_mask")
    save=g.add("SaveImage",f"Save full-size {name} result",(3060,y),(f"Qwen21/{name}/image",))
    g.connect(audit,0,save,"images")
    hand=g.add("Q21WriteHandoff",f"Update {name}_latest for next workflow",(3060,y+190),(name,))
    g.connect(audit,0,hand,"image")
    preview=g.add("PreviewImage","Inspect native-resolution generated crop",(3060,y+400),(),[350,330])
    g.connect(op,0,preview,"images")
    return audit,stitch


def build_all():
    for folder in ("workflows","api_workflows"):(ROOT/folder).mkdir(exist_ok=True)
    # Optional reference normalization, using the same three base weights and no adapters.
    g=Graph("00 Source preparation: two references, no identity adapter")
    g.add("Note","READ FIRST",(0,-280),("OPTIONAL, not required when your original references are suitable. Supply a genuine full-body photo plus a face photo of the same person. The square canvas controls output shape. This is a generated approximation, not recovered ground truth. One additional 32-aligned reference can be wired into image_3. Save the accepted image and compare with originals.",),[790,220])
    body=g.add("LoadImage","PRIMARY: genuine full-body / clothing photo",(0,0),("body.png","image"))
    face=g.add("LoadImage","FACE: same person, identity reference",(0,430),("face.png","image"))
    m,c,v=models(g)
    sq=g.add("Q21SquareCanvas","Square 1024 reference canvas, no stretching",(850,0),(1024,));g.connect(body,0,sq,"image")
    rr=g.add("Q21ResizeBudget","Face reference <= 1 MP",(850,220),(1.0,False));g.connect(face,0,rr,"image")
    enc=g.add("Q21EncodeReferences","EDITABLE source-prep instruction",(1270,0),(PREP_PROMPT,))
    for a,o,k in ((c,0,"clip"),(v,0,"vae"),(sq,0,"image_1"),(rr,0,"image_2")):g.connect(a,o,enc,k)
    ca=g.add("QwenImage21Cache","Lossless cache",(1270,480),("auto","default"));g.connect(m,0,ca,"model")
    sam=g.add("KSampler","No LoRA | 40-step source normalization",(1780,0),(18071000,"fixed",40,1.0,"euler","simple",1.0))
    for a,o,k in ((ca,0,"model"),(enc,0,"positive"),(enc,1,"negative"),(enc,2,"latent_image")):g.connect(a,o,sam,k)
    dec=g.add("VAEDecode","Decode",(2190,0));g.connect(sam,0,dec,"samples");g.connect(v,0,dec,"vae")
    op=g.add("Q21OpaqueImage","Opaque photographic reference",(2190,180));g.connect(dec,0,op,"image");g.connect(sq,0,op,"background")
    sv=g.add("SaveImage","Save candidate, then check likeness",(2600,0),("Qwen21/reference/image",));g.connect(op,0,sv,"images")
    ha=g.add("Q21WriteHandoff","Reference handoff",(2600,200),("reference",));g.connect(op,0,ha,"image")
    g.save("00_Source_Prep_2_References")
    for head,repair,stem in ((False,False,"01_Body_Swap_Masked"),(True,False,"02_Head_Refinement_Masked"),(True,True,"04_Local_Repair_No_LoRA")):
        name="repair" if repair else "head" if head else "body"
        g=Graph(stem)
        g.add("Note","READ FIRST",(0,-290),("Paint a real MASK, not RGB marks: right-click the scene loader > Open in Mask Editor > mask layer > save. Cover the old silhouette and enough room for its replacement, excluding foreground occluders. Crop/stitch handles localization; Qwen redraws the crop, NOT a masked empty latent. The original full scene is never resized. Edit the prompt to identify the correct person. Inspect the body before proceeding to the head pass. Handoff files are overwritten intentionally; historical Save Image files remain.",),[1180,210])
        filename="qwen21_handoff/head_latest.png" if repair else "qwen21_handoff/body_latest.png" if head else "scene.png"
        scene=g.add("LoadImage","SCENE: paint region to edit",(0,0),(filename,"image"))
        ref=g.add("LoadImage","REFERENCE: face / relevant anatomy" if head else "REFERENCE: whole body + desired clothing",(0,450),("face.png" if head else "body.png","image"))
        ms=models(g)
        stage(g,scene,scene,ref,ms,head=head,name=name,repair=repair)
        g.group("1. YOUR IMAGES + MASK",-30,-50,410,890)
        g.group("2. SHARED BF16 BASE FILES",420,-50,360,660)
        g.group("3. LOCAL EDIT + ORIGINAL-SIZE COMPOSITE",810,-50,2650,950)
        g.save(stem)
    # Convenient combined graph: 3 unique image files, 2 independently painted scene masks.
    g=Graph("03 Combined body then head (use staged graphs to approve intermediate results)")
    g.add("Note","READ FIRST",(0,-300),("THREE UNIQUE SOURCE IMAGES: scene, body, face. Load the SAME original scene into BOTH scene loaders. Paint the full-person mask on the first, and the head/hair/neck mask on the second. The second loader supplies a mask only; its RGB must match the original. The head pass receives the generated BODY RESULT, not the old scene. Use the separate head workflow if the new head has moved outside the prepainted mask. Fixed seeds help preserve caching. Running again overwrites the latest handoff files, not historical outputs.",),[1240,220])
    scene=g.add("LoadImage","SCENE A: paint full-person mask",(0,0),("scene.png","image"))
    body=g.add("LoadImage","BODY reference: clothing + physique",(0,440),("body.png","image"))
    scene_head=g.add("LoadImage","SCENE B: same original; paint HEAD mask only",(0,1010),("scene.png","image"))
    face=g.add("LoadImage","FACE reference: identity + hair",(0,1450),("face.png","image"))
    ms=models(g)
    body_done,body_st=stage(g,scene,scene,body,ms,y=0,name="body")
    free=g.add("AUSBOSS_NODES_FreeMemory","Release cached GPU models between stages",(450,1060));g.connect(body_done,0,free,"value")
    head_done,head_st=stage(g,free,scene_head,face,ms,y=1050,head=True,original_for_mask=scene,name="head")
    union=g.add("Q21MaskUnion","Union of both actual blend footprints",(3490,1050));g.connect(body_st,1,union,"a");g.connect(head_st,1,union,"b")
    audit=g.add("Q21AuditPreservation","Final audit against ORIGINAL scene",(3490,1240));g.connect(scene,0,audit,"original");g.connect(head_done,0,audit,"edited");g.connect(union,0,audit,"blend_mask")
    sv=g.add("SaveImage","FINAL native-scene-size PNG",(3910,1240),("Qwen21/final_native/image",));g.connect(audit,0,sv,"images")
    g.group("PASS 1: BODY",820,-50,2640,960);g.group("PASS 2: HEAD FROM BODY RESULT",820,1000,2640,970)
    g.save("03_Combined_Body_Then_Head")
    g=Graph("05 Non-generative enlargement (no additional model files)")
    g.add("Note","READ FIRST",(0,-250),("This is a lossless PNG export after Lanczos enlargement, not newly recovered detail. It does not synthesize a different face. Skip enlargement when the stitched scene is already large enough. The main workflows preserve the original scene dimensions. 2x means twice the width AND height, four times the pixels. For generative enhancement, use the optional SeedVR2 profile and compare against the original identity.",),[1110,180])
    im=g.add("LoadImage","Accepted final image",(0,0),("qwen21_handoff/head_latest.png","image"))
    up=g.add("ImageScaleBy","Non-generative final resize",(450,0),("lanczos",2.0));g.connect(im,0,up,"image")
    sv=g.add("SaveImage","Final resized PNG",(900,0),("Qwen21/final_lanczos_2x/image",));g.connect(up,0,sv,"images")
    g.save("05_Final_Resize_No_Extra_Model")
    g=Graph("06 Optional SeedVR2 FP16 7B still-image finish")
    g.add("Note","OPTIONAL UPSCALE IMAGE PROFILE ONLY",(0,-280),("Requires the qwen21-identity-upscale image: exactly two additional model files. Approve the native face first. This synthesizes detail and may alter likeness. Default target is 2048 pixels on the SHORT edge, capped at 4096 on the long edge, batch 1. This can downsize an already larger scene: skip it in that case. Compare eyes, mouth, jaw, skin marks and hairline against the native image. Noise injection and compilation are OFF. Start with untiled VAE; enable both VAE tiling switches if memory is insufficient. The whole frame is enhanced here; original pixel preservation applies to the native composite, not this optional finish.",),[1240,220])
    im=g.add("LoadImage","Accepted image only",(0,0),("qwen21_handoff/head_latest.png","image"))
    free=g.add("AUSBOSS_NODES_FreeMemory","Release Qwen before optional upscaling",(440,0));g.connect(im,0,free,"value")
    dit=g.add("SeedVR2LoadDiTModel","7B FP16 | SDPA | CPU offload",(440,230),("seedvr2_ema_7b_fp16.safetensors","cuda:0",0,False,"cpu",False,"sdpa"),[390,310])
    vae=g.add("SeedVR2LoadVAEModel","FP16 VAE | untiled quality baseline",(440,620),("ema_vae_fp16.safetensors","cuda:0",False,1024,128,False,1024,128,"false","cpu",False),[390,420])
    up=g.add("SeedVR2VideoUpscaler","Single image | no injected noise",(970,0),(18071003,2048,4096,1,False,0,0,"lab",0.0,0.0,"cpu",False),[430,500])
    g.connect(free,0,up,"image");g.connect(dit,0,up,"dit");g.connect(vae,0,up,"vae")
    sv=g.add("SaveImage","Separate enhanced candidate, retain native",(1510,0),("Qwen21/final_seedvr2_candidate/image",));g.connect(up,0,sv,"images")
    g.save("06_Optional_SeedVR2_FP16_7B")
    print("Generated 7 UI workflows and 7 matching API graphs (SeedVR2 is optional).")

if __name__=="__main__":
    build_all()
