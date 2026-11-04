#!/usr/bin/env python3
"""Generate native App Mode workflows and equivalent API graphs deterministically."""
from __future__ import annotations
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"
SYSTEM = (
    "Write a complete English scene caption for a still-image generator. Return only the caption, "
    "without headings or explanation. Begin with the medium and subject. Retain every requested "
    "subject, quantity, action, garment, color, composition, style and exclusion. Preserve quoted "
    "text exactly in its original language; describe its position and typography when relevant. "
    "Add coherent, plausible material, lighting, camera and background detail only where the "
    "request leaves it open. Do not add people, products, brands, claims or text the user did "
    "not request. Describe visible details, not quality-tag lists or generation settings. "
    "Prefer realistic textures and consistent illumination for photographs; do not impose "
    "photography on an illustration request. Aim for 150 to 250 non-repetitive words, shorter "
    "when the original is already complete. This is text-to-image generation, not image editing."
)
MODEL = "moss_image_sft_bf16.safetensors"
ENCODER = "qwen3.5_9b_bf16.safetensors"
VAE = "flux2-vae.safetensors"
LORA = "armV5.safetensors"

class Graph:
    def __init__(self, name):
        self.name, self.nodes, self.links, self.api = name, [], [], {}
        self.app_inputs, self.app_outputs = [], []

    def node(self, kind, title, widgets=None, inputs=None, outputs=None, pos=None):
        widgets, inputs, outputs = widgets or {}, inputs or [], outputs or []
        i = len(self.nodes) + 1
        n = {"id": i, "type": kind, "title": title,
             "pos": pos or [((i-1) % 4)*410, ((i-1)//4)*310],
             "size": [360, 250 if any(isinstance(v, str) and len(v)>80 for v in widgets.values()) else 160],
             "flags": {}, "order": i-1, "mode": 0,
             "inputs": [{"name": k, "type": t, "link": None} for k,t in inputs],
             "outputs": [{"name": k, "type": t, "links": [], "slot_index": s} for s,(k,t) in enumerate(outputs)],
             "properties": {"Node name for S&R": kind},
             "widgets_values": list(widgets.values()), "widgets_values_named": widgets.copy()}
        self.nodes.append(n)
        self.api[str(i)] = {"class_type": kind, "inputs": {k:v for k,v in widgets.items()
                             if k not in ("control_after_generate", "upload")}, "_meta": {"title": title}}
        return i

    def link(self, source, slot, target, name, widget=False):
        s,t = self.nodes[source-1], self.nodes[target-1]
        match = next((k for k,v in enumerate(t["inputs"]) if v["name"]==name), None)
        typ = s["outputs"][slot]["type"]
        if match is None:
            match = len(t["inputs"])
            t["inputs"].append({"name": name, "type": typ, "link": None})
        if widget:
            t["inputs"][match]["widget"] = {"name": name}
        lid = len(self.links)+1
        t["inputs"][match]["link"] = lid
        s["outputs"][slot]["links"].append(lid)
        self.links.append([lid,source,slot,target,match,typ])
        self.api[str(target)]["inputs"][name] = [str(source),slot]

    def expose(self, node, widget, description):
        self.app_inputs.append([node, widget, {"description": description}])

    def save(self, description):
        n = self.node("MarkdownNote", "Read before running", {"text": description})
        del self.api[str(n)]
        g = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "moss-mobile/v1/"+self.name)),
             "revision": 0, "last_node_id": len(self.nodes), "last_link_id": len(self.links),
             "nodes": self.nodes, "links": self.links, "groups": [], "config": {},
             "extra": {"linearMode": True, "linearData": {"inputs": self.app_inputs, "outputs": self.app_outputs},
                       "frontendVersion": "1.57.4", "ds": {"scale": 0.6, "offset": [30,30]},
                       "moss_mobile": {"release": "1.0.0", "description": description}},
             "version": 0.4}
        for sub,data,suffix in (("workflows",g,".app.json"),("api",self.api,".api.json")):
            p=ROOT/sub/(self.name+suffix); p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(data,indent=2)+"\n")


def models(g, editing=False):
    m = g.node("UNETLoader", "MOSS SFT - full BF16", {"unet_name": MODEL, "weight_dtype": "default"}, outputs=[("MODEL","MODEL")])
    c = g.node("CLIPLoader", "Shared Qwen3.5 9B - full BF16", {"clip_name":ENCODER,"type":"moss_image","device":"default"},outputs=[("CLIP","CLIP")])
    v = g.node("VAELoader", "Full FLUX.2 VAE", {"vae_name":VAE}, outputs=[("VAE","VAE")])
    if editing:
        l = g.node("LoraLoaderModelOnly", "HQ Edit armV5 - native ai-toolkit", {"lora_name":LORA,"strength_model":1.0}, inputs=[("model","MODEL")],outputs=[("MODEL","MODEL")])
        g.link(m,0,l,"model"); m=l
    if not editing:
        # Match the current official SFT generation recipe: SLG is off.
        return m,c,v
    # Native MOSS already has shift 5. No SD3 multiplier node is inserted.
    s = g.node("SkipLayerGuidanceDiTSimple", "Keep SLG layer 22", {"double_layers":"","single_layers":"22","start_percent":0.0,"end_percent":1.0}, inputs=[("model","MODEL")],outputs=[("MODEL","MODEL")])
    g.link(m,0,s,"model")
    return s,c,v


def sample(g,m,pos,neg,v,width_source,height_source,editing=True):
    latent = g.node("EmptyFlux2LatentImage","Matched latent canvas",{"width":1024,"height":1024,"batch_size":1},outputs=[("LATENT","LATENT")])
    g.link(*width_source,latent,"width",widget=True); g.link(*height_source,latent,"height",widget=True)
    steps,cfg,sampler=(50,3.0,"euler") if editing else (25,2.5,"er_sde")
    k = g.node("KSampler", "HQ edit sampling" if editing else "Official SFT generation sampling", {"seed":42,"control_after_generate":"randomize","steps":steps,"cfg":cfg,"sampler_name":sampler,"scheduler":"simple","denoise":1.0},inputs=[("model","MODEL"),("positive","CONDITIONING"),("negative","CONDITIONING"),("latent_image","LATENT")],outputs=[("LATENT","LATENT")])
    for s,slot,name in [(m,0,"model"),(pos,0,"positive"),(neg,0,"negative"),(latent,0,"latent_image")]:g.link(s,slot,k,name)
    d=g.node("VAEDecode","Decode without an automatic face/detail pass",inputs=[("samples","LATENT"),("vae","VAE")],outputs=[("IMAGE","IMAGE")])
    g.link(k,0,d,"samples"); g.link(v,0,d,"vae")
    return d


def save_output(g,image):
    s=g.node("SaveImage","Finished PNG",{"filename_prefix":"MOSS_Mobile/"+g.name},inputs=[("images","IMAGE")],outputs=[("images","IMAGE")])
    g.link(image,0,s,"images"); g.app_outputs=[str(s)]


def generate():
    g=Graph("01_Generate_HQ")
    m,c,v=models(g)
    p=g.node("PrimitiveStringMultiline","Describe your image",{"value":"A professional editorial photograph of a ceramic vase and fresh foliage beside a large window, soft natural daylight, lifelike surface textures, a restrained neutral palette and clear optical detail."},outputs=[("STRING","STRING")])
    g.expose(p,"value","Describe the subject, framing, lighting and style. Put any exact visible text in quotes.")
    size=g.node("MossMobileCanvas","Shape and detail",{"shape":"Portrait 4:5","detail":"Standard (~1 MP)"},outputs=[("width","INT"),("height","INT")])
    g.expose(size,"shape","Exact aspect ratio, with dimensions aligned to 16 pixels.")
    g.expose(size,"detail","Higher detail increases native generation size and compute. No fake resize-only upscale.")
    sys=g.node("PrimitiveStringMultiline","Generation-only expansion instructions",{"value":SYSTEM},outputs=[("STRING","STRING")])
    te=g.node("TextGenerate","Optional expansion using the same encoder",{
        "prompt":"","max_length":1024,"sampling_mode":"on","sampling_mode.temperature":0.7,
        "sampling_mode.top_k":64,"sampling_mode.top_p":0.95,"sampling_mode.min_p":0.05,
        "sampling_mode.repetition_penalty":1.05,"sampling_mode.seed":0,"sampling_mode.presence_penalty":0.0,
        "thinking":False,"use_default_template":True,"mtp":"auto"},
        inputs=[("clip","CLIP"),("system_prompt","STRING")],outputs=[("generated_text","STRING"),("thinking","STRING")])
    g.link(c,0,te,"clip");g.link(sys,0,te,"system_prompt");g.link(p,0,te,"prompt",widget=True)
    choice=g.node("MossMobilePromptChoice","Expand short prompts",{"expand_prompt":True},inputs=[("original","STRING"),("expanded","STRING")],outputs=[("STRING","STRING")])
    g.link(p,0,choice,"original");g.link(te,0,choice,"expanded")
    g.expose(choice,"expand_prompt","ON expands a short idea using the existing encoder. OFF is faster and sends your exact words. Never used by editing apps.")
    positive=g.node("CLIPTextEncode","Positive caption",{"text":""},inputs=[("clip","CLIP")],outputs=[("CONDITIONING","CONDITIONING")])
    negative=g.node("CLIPTextEncode","Model-recommended negatives",{"text":NEGATIVE},inputs=[("clip","CLIP")],outputs=[("CONDITIONING","CONDITIONING")])
    g.link(c,0,positive,"clip");g.link(choice,0,positive,"text",widget=True);g.link(c,0,negative,"clip")
    final=sample(g,m,positive,negative,v,(size,0),(size,1),editing=False);save_output(g,final)
    g.save("Native App Mode. Full BF16 MOSS SFT and Qwen3.5 9B, full VAE, current official SFT sampling recipe: 25 er_sde/simple steps, CFG 2.5, native shift 5, SLG off. Expansion is generation-only and uses no additional model. No Turbo, quantization, automatic face replacement, or hallucinating upscaler. Defaults are quality-first starting points, not a universal optimum. GPU quality validation remains required.")


def load_image(g,title,filename,description):
    n=g.node("LoadImage",title,{"image":filename,"upload":"image"},outputs=[("IMAGE","IMAGE"),("MASK","MASK")])
    g.expose(n,"image",description)
    return n


def edit(reference=False,masked=False):
    name="03_Reference_Edit_HQ" if reference else "04_Protected_Mask_Edit_HQ" if masked else "02_Edit_HQ"
    g=Graph(name)
    m,c,v=models(g,True)
    src=load_image(g,"Image 1 - photo to edit","upload_your_photo.png","Upload the photo to edit. PNG or JPEG; export HEIC/ProRAW as a supported image first.")
    source_for_prep=src
    if masked:
        mask=load_image(g,"Edit mask - white changes, black protects","upload_white_on_black_mask.png","Upload an RGB black/white mask at EXACTLY the original photo's dimensions. White reveals the edit; black copies original pixels; grey blends. This is not an alpha mask.")
        guard=g.node("MossMobileMaskGuard","Check mask before sampling",inputs=[("image","IMAGE"),("mask_image","IMAGE")],outputs=[("IMAGE","IMAGE")])
        g.link(src,0,guard,"image");g.link(mask,0,guard,"mask_image");source_for_prep=guard
    prep=g.node("MossMobilePrepare","Prepare source without cropping",{"megapixels":1.0,"upscale_small":True},inputs=[("image","IMAGE")],outputs=[("working_image","IMAGE"),("width","INT"),("height","INT"),("geometry","MOSS_GEOMETRY")])
    g.link(source_for_prep,0,prep,"image")
    g.expose(prep,"megapixels","Working resolution: 1 MP is the baseline. 1.5 or 2 MP samples more native detail but costs more memory and has not been quality-benchmarked here. Output returns to the original canvas size.")
    refs=[prep]
    if reference:
        other=load_image(g,"Image 2 - visual reference","upload_reference.png","Upload the reference. Say which property to take from it and which image controls the composition.")
        refprep=g.node("MossMobilePrepare","Prepare reference independently",{"megapixels":1.0,"upscale_small":True},inputs=[("image","IMAGE")],outputs=[("working_image","IMAGE"),("width","INT"),("height","INT"),("geometry","MOSS_GEOMETRY")])
        g.link(other,0,refprep,"image");refs.append(refprep)
    prompt="Change only the requested element in image 1. Preserve the subject identity, pose, framing, lighting and all unrequested content."
    if reference:prompt="Edit image 1 using image 2 only as a reference for the requested change. Image 1 controls the framing, position, pose and background. Preserve everything not explicitly requested."
    p=g.node("PrimitiveStringMultiline","Describe the change",{"value":prompt},outputs=[("STRING","STRING")])
    g.expose(p,"value","Replace this with the actual change. Keep the instruction specific; it goes to the editor unchanged.")
    enc=[]
    for label,caption in (("positive",""),("negative",NEGATIVE)):
        inputs=[("clip","CLIP")]+[(f"images.image_{k}","IMAGE") for k in range(len(refs))]+[("vae","VAE")]
        e=g.node("TextEncodeMOSSImageEdit","Reference conditioning - "+label,{"prompt":caption,"edit_format":"ai-toolkit","image_strength":1.0},inputs=inputs,outputs=[("CONDITIONING","CONDITIONING")])
        # One empty optional input gives the frontend autogrow widget its next slot.
        g.nodes[e-1]["inputs"].insert(1+len(refs),{"name":f"images.image_{len(refs)}","type":"IMAGE","shape":7,"link":None})
        g.link(c,0,e,"clip");g.link(v,0,e,"vae")
        for k,ref in enumerate(refs):g.link(ref,0,e,f"images.image_{k}")
        if label=="positive":g.link(p,0,e,"prompt",widget=True)
        enc.append(e)
    decoded=sample(g,m,*enc,v,(prep,1),(prep,2))
    restore=g.node("MossMobileRestore","Match the original output dimensions",inputs=[("generated","IMAGE"),("geometry","MOSS_GEOMETRY")],outputs=[("IMAGE","IMAGE")])
    g.link(decoded,0,restore,"generated");g.link(prep,3,restore,"geometry")
    result=restore
    if masked:
        comp=g.node("MossMobileComposite","Copy protected pixels from the original",inputs=[("original","IMAGE"),("edited","IMAGE"),("mask_image","IMAGE")],outputs=[("IMAGE","IMAGE")])
        g.link(src,0,comp,"original");g.link(restore,0,comp,"edited");g.link(mask,0,comp,"mask_image");result=comp
    save_output(g,result)
    g.save("MOSS SFT + community HQ Edit armV5, LoRA strength 1.0, native ai-toolkit format on BOTH conditionings, 50 Euler/simple steps, CFG 3, SLG layer 22. Not latent img2img. No prompt rewriting. Source is fitted to the selected working resolution (1 MP by default) and bottom/right padded to 16 pixels, not cropped. Padding is removed and the output returns to the exact original dimensions. Dimensions do not guarantee pixel-perfect generated content or restore missing source detail. " + ("This app hard-composites original pixels outside the uploaded mask. It is NOT a dedicated mask-conditioned inpainting model. Grey mask pixels blend; there is no implicit feathering or grow." if masked else "Unmasked image edits can still drift. The two-reference adapter has not been independently benchmarked here." if reference else "Use Protected Mask Edit when unchanged pixels must be preserved exactly."))

if __name__=="__main__":
    generate();edit();edit(reference=True);edit(masked=True)
    print("Wrote four native App Mode workflows and four API graphs.")
