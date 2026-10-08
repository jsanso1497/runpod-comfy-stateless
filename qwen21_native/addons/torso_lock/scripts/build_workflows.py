#!/usr/bin/env python3
"""Deterministic, model-free construction of four focused torso workflows."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSION="1.0.0"
S={
 "LoadImage":([], ["IMAGE","MASK"], ["image","upload"]),
 "Q21TOptionalImage":([], ["IMAGE"], ["image","upload"]),
 "UNETLoader":([], ["MODEL"], ["unet_name","weight_dtype"]),
 "CLIPLoader":([], ["CLIP"], ["clip_name","type","device"]),
 "VAELoader":([], ["VAE"], ["vae_name"]),
 "QwenImage21Cache":([("model","MODEL")], ["MODEL"], ["device","dtype"]),
 "Q21TPrepareMasks":([("base_image","IMAGE"),("edit_mask","MASK"),("original_image","IMAGE"),("protect_mask","MASK")],
    ["IMAGE","MASK","MASK","MASK","IMAGE","STRING"], ["mode","inward_feather_px","protection_padding_px"]),
 "Q21TTorsoReferences":([("base_crop","IMAGE"),("primary_torso","IMAGE"),("secondary_torso","IMAGE"),("third_torso","IMAGE"),("garment","IMAGE")],
    ["Q21T_REFERENCES","STRING","STRING"], ["task","garment_description","extra_instructions","reference_megapixels"]),
 "Q21TNativeEncode":([("clip","CLIP"),("vae","VAE"),("ordered_images","Q21T_REFERENCES"),("prompt","STRING")],
    ["CONDITIONING","CONDITIONING","LATENT"], []),
 "AUSBOSS_NODES_CropForInpaint":([("image","IMAGE"),("mask","MASK")],["IMAGE","MASK","AUSBOSS_STITCHER"],
    ["context_factor","blend_pixels","output_multiple","target_width","target_height","mask_grow","mask_blur","invert_mask","context_pixels","target_megapixels","rescale_algorithm","extend_left","extend_right","extend_up","extend_down","keep_inside"]),
 "KSampler":([("model","MODEL"),("positive","CONDITIONING"),("negative","CONDITIONING"),("latent_image","LATENT")],
    ["LATENT"], ["seed","control_after_generate","steps","cfg","sampler_name","scheduler","denoise"]),
 "VAEDecode":([("samples","LATENT"),("vae","VAE")],["IMAGE"],[]),
 "Q21TOpaqueCrop":([("generated","IMAGE"),("source_crop","IMAGE")],["IMAGE"],[]),
 "AUSBOSS_NODES_StitchInpaint":([("stitcher","AUSBOSS_STITCHER"),("inpainted","IMAGE")],["IMAGE","MASK"],["fix_edge_halo","color_match","seam"]),
 "Q21TFinalize":([("base_locked","IMAGE"),("original_image","IMAGE"),("candidate","IMAGE"),("paste_alpha","MASK"),("hard_protection","MASK"),("mask_report","STRING"),("reference_roles","STRING"),("compiled_prompt","STRING")],
    ["IMAGE","IMAGE","IMAGE","STRING"],[]),
 "Q21TSaveVerified":([("image","IMAGE"),("edit_matte","IMAGE"),("comparison","IMAGE"),("audit_json","STRING")],[],["filename_prefix"]),
 "PreviewImage":([("images","IMAGE")],[],[]),
 "SaveImage":([("images","IMAGE")],[],["filename_prefix"]),
 "Note":([],[],["text"]),
}
OUTPUT_NAMES={
 "Q21TPrepareMasks":["base_with_original_lock","context_union_mask","strict_paste_alpha","hard_protection","mask_preview","mask_report"],
 "Q21TTorsoReferences":["ordered_images","compiled_prompt","reference_roles"],
 "Q21TNativeEncode":["positive","negative","matching_empty_latent"],
 "Q21TFinalize":["VERIFIED_FINAL","final_comparison","actual_edit_matte","audit_json"],
 "AUSBOSS_NODES_CropForInpaint":["image","mask","stitcher"],
 "AUSBOSS_NODES_StitchInpaint":["image","blend_mask"],
}

class Graph:
    def __init__(self,title):
        self.title=title;self.nodes=[];self.links=[];self.api={};self.groups=[]
    def add(self,typ,title,xy,widgets=(),size=None):
        ins,outs,names=S[typ]
        assert len(widgets)==len(names),(typ,widgets,names)
        i=len(self.nodes)+1
        if size is None:
            size=[360,max(120,85+28*max(len(ins),len(widgets)))]
            if typ in ("LoadImage","Q21TOptionalImage"):size=[390,390]
        node={"id":i,"type":typ,"title":title,"pos":list(xy),"size":size,"flags":{},"order":i-1,"mode":0,
            "inputs":[{"name":k,"type":t,"link":None} for k,t in ins],
            "outputs":[{"name":OUTPUT_NAMES.get(typ,[t.lower() for t in outs])[j],"type":t,"links":[],"slot_index":j} for j,t in enumerate(outs)],
            "properties":{"Node name for S&R":typ},"widgets_values":list(widgets)}
        self.nodes.append(node)
        if typ!="Note":
            self.api[str(i)]={"class_type":typ,"inputs":{k:v for k,v in zip(names,widgets) if k not in ("upload","control_after_generate")},"_meta":{"title":title}}
        return i
    def connect(self,a,out,b,name):
        src=self.nodes[a-1];dst=self.nodes[b-1]
        slot=next(j for j,p in enumerate(dst["inputs"]) if p["name"]==name)
        assert dst["inputs"][slot]["link"] is None
        assert src["outputs"][out]["type"]==dst["inputs"][slot]["type"]
        lid=len(self.links)+1
        self.links.append([lid,a,out,b,slot,dst["inputs"][slot]["type"]])
        src["outputs"][out]["links"].append(lid);dst["inputs"][slot]["link"]=lid
        self.api[str(b)]["inputs"][name]=[str(a),out]
    def group(self,title,x,y,w,h):
        self.groups.append({"id":len(self.groups)+1,"title":title,"bounding":[x,y,w,h],"color":"#365b75","font_size":24,"flags":{}})
    def save(self,stem):
        ui={"last_node_id":len(self.nodes),"last_link_id":len(self.links),"nodes":self.nodes,"links":self.links,
            "groups":self.groups,"config":{},"extra":{"ds":{"scale":0.45,"offset":[30,430]},"qwen21_torso_lock":{"version":VERSION,"title":self.title}},"version":0.4}
        (ROOT/"workflows"/(stem+".json")).write_text(json.dumps(ui,indent=2)+"\n")
        (ROOT/"api_workflows"/(stem+".api.json")).write_text(json.dumps(self.api,indent=2)+"\n")

DEFAULT_GARMENT="A plain black full-coverage athletic sports bra with natural fit, without added padding or push-up reshaping."
MASK_GUIDE=("A: Load the scene and paint an EDIT mask over the old top, the new garment silhouette, and the skin to reveal. "
 "Include a small transition margin to remove old cloth/shadows. B: Load the SAME original photo and paint PROTECT over "
 "the already-visible abdomen including the belly button. Leave only a narrow transition at the old hem editable. "
 "White in B means KEEP, not edit. Do not paint the RGB layer. Protection wins wherever masks overlap.\n\n"
 "Orange preview = editable; cyan = original pixels locked. Qwen sees the unmarked context crop including the protected "
 "abdomen, not this colored preview or the masks. The final composite is capped inside A and restores B after stitching. "
 "No navel detection or 3D registration is performed. Check the final for duplicate/invented details.")


def prepare(g,refine=False):
    scene=g.add("LoadImage","A | ACCEPTED CANDIDATE: paint only the skin/seam to repair" if refine else "A | ORIGINAL SCENE: paint clothing / newly exposed skin EDIT mask",(0,0),("torso_accepted.png" if refine else "scene.png","image"))
    original=g.add("LoadImage","B | ORIGINAL PHOTO: paint exposed ABDOMEN / NAVEL to PROTECT",(0,500),("scene.png","image"))
    masks=g.add("Q21TPrepareMasks","Mask boundaries: inward feather only | protection wins",(1020,0),("refine_accepted_candidate" if refine else "first_pass_same_scene",8,0),[380,330])
    for src,out,name in ((scene,0,"base_image"),(scene,1,"edit_mask"),(original,0,"original_image"),(original,1,"protect_mask")):
        g.connect(src,out,masks,name)
    preview=g.add("PreviewImage","MASK CHECK ONLY | orange=edit | cyan=locked original",(1020,450),(),[380,430]);g.connect(masks,4,preview,"images")
    g.group("A + B: MATCHED SCENES / MASKS",-25,-60,455,1030)
    g.group("MASKS AND HARD BOUNDARIES",990,-60,430,990)
    return scene,original,masks


def final_nodes(g,original,masks,candidate,stem,plan=None,x=3680):
    fin=g.add("Q21TFinalize","FINAL PIXEL LOCK + exact preservation audit",(x,0),(),[380,340])
    for a,o,k in ((masks,0,"base_locked"),(original,0,"original_image"),(candidate[0],candidate[1],"candidate"),
                  (masks,2,"paste_alpha"),(masks,3,"hard_protection"),(masks,5,"mask_report")):
        g.connect(a,o,fin,k)
    if plan is not None:
        g.connect(plan,2,fin,"reference_roles");g.connect(plan,1,fin,"compiled_prompt")
    saver=g.add("Q21TSaveVerified","SAVE THIS: verified final + matching audit / mask / comparison",(x+480,0),(f"Qwen21_Torso/{stem}/final",),[410,360])
    for out,name in ((0,"image"),(1,"comparison"),(2,"edit_matte"),(3,"audit_json")):g.connect(fin,out,saver,name)
    pv=g.add("PreviewImage","BASE / VERIFIED FINAL | after stitch and protection",(x,440),(),[850,520]);g.connect(fin,1,pv,"images")
    g.group("FINAL, NEVER THE RAW GENERATED CROP",x-30,-60,960,1080)


def generation(refine=False):
    stem="37_Torso_Skin_Seam_Repair_Locked" if refine else "36_Sports_Bra_Torso_Landmark_Lock"
    g=Graph(stem)
    extra=("FOLLOW-UP ONLY: Image A is your accepted result, image B remains the original photograph. Paint only the new skin/seam error in A. "
           "The protected original pixels are restored before and after this refinement. Do not queue automatically; use only when needed.\n\n") if refine else "START HERE. Native Qwen only, no BFS. Real torso references are evidence; the scene owns placement and lighting.\n\n"
    g.add("Note",stem+" | READ BEFORE FIRST QUEUE",(0,-470),(extra+MASK_GUIDE,),[2350,360])
    scene,original,masks=prepare(g,refine)
    primary=g.add("LoadImage","C | REQUIRED: closest-angle GENUINE torso photograph",(510,0),("torso_primary.png","image"))
    second=g.add("Q21TOptionalImage","D | OPTIONAL: second genuine torso angle",(510,480),("__none__","image"))
    third=g.add("Q21TOptionalImage","E | OPTIONAL: third useful torso angle",(510,960),("__none__","image"))
    garment=None
    if not refine:garment=g.add("Q21TOptionalImage","F | OPTIONAL: sports bra photo, garment design ONLY",(510,1440),("__none__","image"))
    crop=g.add("AUSBOSS_NODES_CropForInpaint","CONTEXT = edit + protected abdomen | 2 MP | full scene stays native",(1460,0),
        (1.2,0,32,0,0,0,0.0,False,64,2.0,"bicubic",0,0,0,0,True),[380,640])
    g.connect(masks,0,crop,"image");g.connect(masks,1,crop,"mask")
    plan=g.add("Q21TTorsoReferences","GARMENT / LOCAL INSTRUCTION | reference numbers assigned automatically",(1900,0),
        ("repair_skin_seam" if refine else "replace_crop_top",DEFAULT_GARMENT,
         "Correct only the visible seam or skin-tone mismatch; keep accepted anatomy and the sports bra unchanged." if refine else "",1.0),[530,660])
    for a,o,k in ((crop,0,"base_crop"),(primary,0,"primary_torso"),(second,0,"secondary_torso"),(third,0,"third_torso")):g.connect(a,o,plan,k)
    if garment:g.connect(garment,0,plan,"garment")
    m=g.add("UNETLoader","Qwen 2.1 BF16 | NO LoRA",(1900,790),("qwen_image_2.1_bf16.safetensors","default"))
    clip=g.add("CLIPLoader","Qwen3-VL 8B BF16 | qwen_image",(1900,990),("qwen3vl_8b_bf16.safetensors","qwen_image","default"))
    vae=g.add("VAELoader","Qwen 2.1 VAE BF16",(1900,1210),("qwen_image_2.1_vae_bf16.safetensors",))
    cache=g.add("QwenImage21Cache","Lossless KV cache",(2500,790),("auto","default"));g.connect(m,0,cache,"model")
    enc=g.add("Q21TNativeEncode","Native edit: resolution=0 | matching empty latent",(2500,0),(),[365,240])
    for a,o,k in ((clip,0,"clip"),(vae,0,"vae"),(plan,0,"ordered_images"),(plan,1,"prompt")):g.connect(a,o,enc,k)
    sampler=g.add("KSampler","40 steps | Euler/simple | CFG 1 | fixed seed",(2910,0),(18071042,"fixed",40,1.0,"euler","simple",1.0),[360,340])
    for a,o,k in ((cache,0,"model"),(enc,0,"positive"),(enc,1,"negative"),(enc,2,"latent_image")):g.connect(a,o,sampler,k)
    dec=g.add("VAEDecode","Decode native crop",(2910,440));g.connect(sampler,0,dec,"samples");g.connect(vae,0,dec,"vae")
    opaque=g.add("Q21TOpaqueCrop","Flatten alpha over SOURCE CROP",(2910,650));g.connect(dec,0,opaque,"generated");g.connect(crop,0,opaque,"source_crop")
    stitch=g.add("AUSBOSS_NODES_StitchInpaint","Reproject crop only | no tone matching or outward feather",(3310,0),(False,0.0,"classic"),[340,230])
    g.connect(crop,2,stitch,"stitcher");g.connect(opaque,0,stitch,"inpainted")
    final_nodes(g,original,masks,(stitch,0),stem,plan)
    g.add("Note","QUALITY / MASK NOTES",(1460,770),
        ("The union mask on Crop For Inpaint sets CONTEXT only. Keep its blend/grow/blur at 0. Final paste uses the separate strict alpha from Prepare Masks.\n\n"
         "Denoise 1.0 is intentional: the native encoder provides an empty matching latent. Lowering it is not a preservation slider.\n\n"
         "Start with 2 MP. Try 4 MP only as a comparison; never resize the whole scene or globally upscale the final when pixel preservation matters.\n\n"
         "Leave optional refs at __none__. If a garment photo is supplied it controls the outfit; otherwise edit garment_description. The actual compiled prompt and reference roles are saved in the audit JSON.",),[380,550])
    g.group("C TO F: GENUINE REFERENCES",480,-60,440,1950 if garment else 1470)
    g.group("PREPARED NATIVE QWEN EDIT",1870,-60,580,1440)
    g.save(stem)


def mask_check():
    stem="36A_Check_Masks_Only_NO_MODELS";g=Graph(stem)
    g.add("Note","CHECK BEFORE SPENDING GPU TIME",(0,-400),(MASK_GUIDE+"\n\nThis workflow does not load or sample a model. Copy the two painted Load Image nodes into workflow 36, or select their saved masked PNGs there.",),[1800,290])
    _,_,masks=prepare(g)
    sv=g.add("SaveImage","Save mask layout preview ONLY",(1490,0),("Qwen21_Torso/mask_check/preview",));g.connect(masks,4,sv,"images")
    g.save(stem)


def aligned_patch():
    stem="38_Aligned_Photographic_Patch_NO_AI";g=Graph(stem)
    g.add("Note","OPTIONAL EXPERT ROUTE | NO REGISTRATION OR GENERATION",(0,-460),
        ("Supply a photographically aligned candidate on the EXACT original-size scene canvas. Align/warp the torso patch externally first. "
         "This workflow does NOT align different viewpoints or synthesize unseen anatomy. It only pastes the aligned candidate through A, feathers INWARD, and restores B. "
         "Use after making a near-angle photographic composite, not with a raw differently framed torso reference.\n\n"+MASK_GUIDE,),[2230,350])
    _,original,masks=prepare(g)
    candidate=g.add("LoadImage","C | ALREADY ALIGNED full-scene composite | identical canvas",(510,0),("aligned_torso_candidate.png","image"))
    final_nodes(g,original,masks,(candidate,0),stem,x=1490)
    g.save(stem)


def main():
    for sub in ("workflows","api_workflows","config"):(ROOT/sub).mkdir(exist_ok=True)
    generation();generation(True);mask_check();aligned_patch()
    catalog={"version":VERSION,"new_models":[],"workflows":[
        {"file":"36_Sports_Bra_Torso_Landmark_Lock.json","purpose":"Main garment + newly revealed skin edit"},
        {"file":"36A_Check_Masks_Only_NO_MODELS.json","purpose":"Inspect mask ownership without loading models"},
        {"file":"37_Torso_Skin_Seam_Repair_Locked.json","purpose":"Optional skin/seam repair after approval"},
        {"file":"38_Aligned_Photographic_Patch_NO_AI.json","purpose":"Strict paste of an already aligned photographic composite"},
    ]}
    (ROOT/"config/catalog.json").write_text(json.dumps(catalog,indent=2)+"\n")
    print("Generated 4 UI/API workflow pairs.")

if __name__=="__main__":main()
