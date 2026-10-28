"""Editable canvases, paired API prompts and strict native-recipe adaptation."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import uuid
from .config import ROOT, atomic_json

AKATZ_LORA = "h3_character_swap_v1.5.safetensors"
UPSTREAM_LORA = "h3-v2-challenge-eval-20261013/v2-1000.safetensors"
DIRECT_PROMPT = ("Replace the selected person in <Video 1> with the same character shown in <Picture 1>. "
                 "Keep the reference character's face, hair, body proportions and clothing. "
                 "Follow the source performance's pose, gaze, expression, gestures, object interactions and movement at the source timing. "
                 "Preserve the source scene, lighting, camera and other people. The selected performer is replaced, not duplicated.")


def validate_graph(g):
    nodes = {n["id"]: n for n in g["nodes"]}
    if len(nodes) != len(g["nodes"]):
        raise ValueError("Duplicate graph node IDs")
    links = {}
    for row in g.get("links", []):
        if len(row) != 6 or row[0] in links:
            raise ValueError("Invalid graph link")
        ident, src, out, dst, inp, kind = row
        if src not in nodes or dst not in nodes or min(out, inp) < 0:
            raise ValueError("Dangling graph link")
        if ident not in (nodes[src]["outputs"][out].get("links") or []) or nodes[dst]["inputs"][inp].get("link") != ident:
            raise ValueError("Nonreciprocal graph link")
        links[ident] = row
    for n in nodes.values():
        for p in n.get("inputs", []):
            if p.get("link") is not None and p["link"] not in links:
                raise ValueError("Missing input link")
        for p in n.get("outputs", []):
            if any(i not in links for i in p.get("links") or []):
                raise ValueError("Missing output link")
    return True


def set_widget(node, name, value):
    named = node.get("widgets_values_named", {})
    values = node.get("widgets_values", [])
    if name not in named:
        raise ValueError(f"Upstream {node['type']} has no named {name} widget; review before adapting")
    keys = list(named)
    index = keys.index(name)
    if index >= len(values) or values[index] != named[name]:
        raise ValueError("Upstream widget order changed; refusing positional edits")
    values[index] = value
    named[name] = value


def replace_value(obj, old, new):
    if isinstance(obj, dict):
        return {k: replace_value(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_value(v, old, new) for v in obj]
    return new if obj == old else obj


def prune_to_outputs(graph, output_ids):
    graph = copy.deepcopy(graph)
    keep = set(output_ids)
    changed = True
    while changed:
        changed = False
        for row in graph["links"]:
            if row[3] in keep and row[1] not in keep:
                keep.add(row[1]); changed = True
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] in keep]
    graph["links"] = [r for r in graph["links"] if r[1] in keep and r[3] in keep]
    lids = {r[0] for r in graph["links"]}
    for n in graph["nodes"]:
        for p in n.get("outputs", []):
            p["links"] = [i for i in p.get("links") or [] if i in lids]
    graph["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "h3-swap/mask-preview/2026-10-22"))
    validate_graph(graph)
    return graph


def adapt_akatz(graph, api):
    graph = replace_value(copy.deepcopy(graph), UPSTREAM_LORA, AKATZ_LORA)
    api = replace_value(copy.deepcopy(api), UPSTREAM_LORA, AKATZ_LORA)
    # Keep the developer's graph, loops, masks, history, attention and sampler intact.
    for n in graph["nodes"]:
        t = n["type"]
        if t == "H3RelayPrepareSource":
            set_widget(n, "duration_seconds", 124 / 24)
        elif t == "MiniMaxH3ReferenceToVideo":
            set_widget(n, "prompt", DIRECT_PROMPT)
        elif t == "LoadVideo":
            set_widget(n, "file", "source.mp4")
        elif t == "LoadImage":
            set_widget(n, "image", "character.png")
        elif t == "SaveVideo":
            set_widget(n, "filename_prefix", "h3_character_swap/akatz_v15")
    for n in api.values():
        t, v = n["class_type"], n["inputs"]
        if t == "H3RelayPrepareSource": v["duration_seconds"] = 124 / 24
        elif t == "MiniMaxH3ReferenceToVideo": v["prompt"] = DIRECT_PROMPT
        elif t == "LoadVideo": v["file"] = "source.mp4"
        elif t == "LoadImage": v["image"] = "character.png"
        elif t == "SaveVideo": v["filename_prefix"] = "h3_character_swap/akatz_v15"
    verify_akatz_recipe(api)
    validate_graph(graph)
    by_id = {n["id"]: n for n in graph["nodes"]}
    roots = [r[3] for r in graph["links"] if by_id[r[1]]["type"] == "H3RelayExpandMask" and by_id[r[3]]["type"] == "PreviewImage"]
    if not roots:
        raise ValueError("Upstream mask review branch is missing")
    return graph, api, prune_to_outputs(graph, roots)


def verify_akatz_recipe(api):
    classes = {n["class_type"] for n in api.values()}
    must = {"SAM3_VideoTrack", "H3RelayWindowPlan", "H3RelayApplyEditMask", "H3RelayApplyHistory", "H3RelayTextMemory", "H3RelaySamplingMemory", "MiniMaxH3ReferenceToVideo", "StartLoop", "EndLoop"}
    if not must <= classes:
        raise ValueError("Upstream native recipe changed")
    samplers = [n["inputs"] for n in api.values() if n["class_type"] == "KSamplerSelect"]
    schedules = [n["inputs"] for n in api.values() if n["class_type"] == "BasicScheduler"]
    if not any(n.get("sampler_name") == "er_sde" for n in samplers):
        raise ValueError("Akatz er_sde recipe not found")
    if not any(n.get("scheduler") == "normal" and n.get("steps") == 8 for n in schedules):
        raise ValueError("Akatz eight-step normal schedule not found")
    loras = [n["inputs"] for n in api.values() if n["class_type"] == "LoraLoaderModelOnly"]
    expected = {AKATZ_LORA, "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"}
    if {n.get("lora_name") for n in loras} != expected or any(n.get("strength_model") != 1 for n in loras):
        raise ValueError("The matched Akatz adapters must stay separate and at strength 1")


class Graph:
    def __init__(self, name):
        self.name = name; self.nodes = []; self.links = []; self.api = {}
    def add(self, kind, title, x, y, inputs=(), outputs=(), widgets=None, size=(350, 170)):
        ident = len(self.nodes) + 1
        widgets = widgets or {}
        n = {"id": ident, "type": kind, "title": title, "pos": [x, y], "size": list(size), "flags": {}, "order": ident-1, "mode": 0,
             "inputs": [{"name": a, "type": b, "link": None} for a,b in inputs],
             "outputs": [{"name": a, "type": b, "links": []} for a,b in outputs],
             "properties": {"Node name for S&R": kind}, "widgets_values": list(widgets.values()), "widgets_values_named": dict(widgets)}
        self.nodes.append(n)
        self.api[str(ident)] = {"class_type": kind, "inputs": {k:v for k,v in widgets.items() if k not in ("upload", "control_after_generate")}, "_meta": {"title": title}}
        return ident
    def connect(self, src, out, dst, field, kind=None):
        s, d = self.nodes[src-1], self.nodes[dst-1]
        matches = [i for i,p in enumerate(d["inputs"]) if p["name"] == field]
        if not matches:
            i = len(d["inputs"])
            port = {"name": field, "type": kind or s["outputs"][out]["type"], "link": None}
            if field in d["widgets_values_named"]: port["widget"] = {"name": field}
            d["inputs"].append(port)
        else: i = matches[0]
        lid = len(self.links)+1
        self.links.append([lid,src,out,dst,i,kind or s["outputs"][out]["type"]])
        s["outputs"][out]["links"].append(lid); d["inputs"][i]["link"] = lid
        self.api[str(dst)]["inputs"][field] = [str(src),out]
    def result(self):
        g = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL,self.name)), "revision": 0, "last_node_id": len(self.nodes), "last_link_id": len(self.links), "nodes": self.nodes, "links": self.links, "groups": [], "config": {}, "extra": {"ds": {"scale": 0.65, "offset": [50,50]}, "validation": "source candidate; GPU execution not certified"}, "version": 0.4}
        validate_graph(g)
        return g, self.api


def add_image(g, x, y):
    return g.add("LoadImage", "CHARACTER: one approved identity / outfit reference", x,y,outputs=[("IMAGE","IMAGE"),("MASK","MASK")],widgets={"image":"character.png","upload":"image"},size=(350,360))


def prompt_studio():
    g = Graph("H3 Akatz Long Analysis 2026-10-27")
    v = g.add("LoadVideo", "SOURCE: the same selected clip as the native render", 0, 0,
        outputs=[("VIDEO", "VIDEO")], widgets={"file": "source.mp4"}, size=(350, 350))
    im = add_image(g, 0, 420)
    a = g.add("H3SwapAnalyzeAkatz", "AKATZ ONLY: 16K output / 32K context / 16 samples", 430, 0,
        inputs=[("video", "VIDEO"), ("character", "IMAGE")],
        outputs=[("clip_prompt", "STRING"), ("chain_prompt", "STRING"),
                 ("report", "STRING"), ("contact_sheet", "IMAGE")],
        widgets={"target_person": "the main performer", "start_seconds": 0., "frames": 124,
                 "wardrobe": "Reference clothing", "reuse_cached_draft": True,
                 "detail_level": "Exhaustive", "max_output_tokens": 16384,
                 "context_tokens": "32768", "sampled_frames": 16,
                 "reference_layout": "Single character image", "timeout_seconds": 1800},
        size=(490, 570))
    g.connect(v, 0, a, "video"); g.connect(im, 0, a, "character")
    for i, title in enumerate(("DETAILED CLIP PROMPT: only for this analyzed interval",
                               "CHAIN-SAFE PROMPT: identity + source-following, no replayed timeline",
                               "ANALYSIS REPORT: evidence, actual tokens, limits and saved files")):
        output = g.add("PreviewAny", title, 980 + i*560, 0, inputs=[("source", "*")],
                       outputs=[("output", "*")], size=(530, 700))
        g.connect(a, i, output, "source", "STRING")
    sheet = g.add("PreviewImage", "All source samples inspected (audio NOT analyzed)",
                  430, 650, inputs=[("images", "IMAGE")], outputs=[], size=(1100, 700))
    g.connect(a, 3, sheet, "images")
    return g.result()


def bfs_graph():
    g=Graph("H3 BFS guide baseline 2026-10-22")
    v=g.add("LoadVideo","SOURCE PERFORMANCE",0,0,outputs=[("VIDEO","VIDEO")],widgets={"file":"source.mp4"},size=(340,320))
    im=add_image(g,0,390)
    prep=g.add("H3SwapClipWindow","EXACT SOURCE WINDOW: 124 frames; author's low-res baseline",400,0,inputs=[("video","VIDEO")],outputs=[("images","IMAGE"),("audio","AUDIO"),("width","INT"),("height","INT"),("length","INT"),("report","STRING")],widgets={"start_seconds":0.0,"frames":124,"short_edge":384},size=(420,230));g.connect(v,0,prep,"video")
    prompt=g.add("H3SwapBFSPrompt","Paste approved CHARACTER DETAILS here; template stays intact",400,330,outputs=[("prompt","STRING")],widgets={"character_details":"person with the face, hair, physique and clothing visible in the character reference"},size=(420,230))
    model=g.add("UNETLoader","H3 Ref2VA: developer INT8 baseline",900,-350,outputs=[("MODEL","MODEL")],widgets={"unet_name":"minimax_h3_ref2va_pruned_int8_convrot.safetensors","weight_dtype":"default"})
    turbo=g.add("LoraLoaderModelOnly","BFS matched DMD8 adapter (not Akatz's Turbo)",1300,-350,inputs=[("model","MODEL")],outputs=[("MODEL","MODEL")],widgets={"lora_name":"minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors","strength_model":1.0});g.connect(model,0,turbo,"model")
    adapter=g.add("LoraLoaderModelOnly","BFS CHARACTER SWAP v1.1",1700,-350,inputs=[("model","MODEL")],outputs=[("MODEL","MODEL")],widgets={"lora_name":"minimax_h3_character_swap_v1.1_ref2va_fro.safetensors","strength_model":1.0});g.connect(turbo,0,adapter,"model")
    shift=g.add("MiniMaxH3SigmaShift","Native H3 flow shifts",2100,-350,inputs=[("model","MODEL")],outputs=[("MODEL","MODEL")],widgets={"shift_video":12.0,"shift_audio":3.0});g.connect(adapter,0,shift,"model")
    clip=g.add("CLIPLoader","H3 text/vision encoder",900,-80,outputs=[("CLIP","CLIP")],widgets={"clip_name":"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","type":"minimax","device":"default"})
    vae=g.add("VAELoader","Original H3 video VAE",900,170,outputs=[("VAE","VAE")],widgets={"vae_name":"minimax_h3_video_vae_fp16.safetensors"})
    avae=g.add("VAELoader","H3 audio VAE",900,380,outputs=[("VAE","VAE")],widgets={"vae_name":"minimax_h3_audio_vae_fp32.safetensors"})
    cond=g.add("MiniMaxH3ReferenceToVideo","CHARACTER ONLY: source video goes to Add Guide",1300,0,inputs=[("clip","CLIP"),("vae","VAE"),("audio_vae","VAE"),("ref_images.ref_image_0","IMAGE")],outputs=[("positive","CONDITIONING"),("latent","LATENT")],widgets={"prompt":"","width":672,"height":384,"length":124,"ref_image_size":"match"},size=(430,440))
    for src,field in ((clip,"clip"),(vae,"vae"),(avae,"audio_vae")):g.connect(src,0,cond,field)
    g.connect(im,0,cond,"ref_images.ref_image_0");g.connect(prompt,0,cond,"prompt")
    for out,field in ((2,"width"),(3,"height"),(4,"length")):g.connect(prep,out,cond,field)
    guide=g.add("MiniMaxH3AddGuide","SOURCE VIDEO GUIDE at frame 0: trained BFS path",1800,30,inputs=[("positive","CONDITIONING"),("latent","LATENT"),("vae","VAE"),("image","IMAGE")],outputs=[("positive","CONDITIONING")],widgets={"frame_idx":0},size=(360,230))
    g.connect(cond,0,guide,"positive");g.connect(cond,1,guide,"latent");g.connect(vae,0,guide,"vae");g.connect(prep,0,guide,"image")
    guider=g.add("BasicGuider","CFG 1 equivalent: native positive guidance",2250,0,inputs=[("model","MODEL"),("conditioning","CONDITIONING")],outputs=[("GUIDER","GUIDER")]);g.connect(shift,0,guider,"model");g.connect(guide,0,guider,"conditioning")
    noise=g.add("RandomNoise","Fixed seed for comparison",2250,240,outputs=[("NOISE","NOISE")],widgets={"noise_seed":42,"control_after_generate":"fixed"})
    sampler=g.add("KSamplerSelect","BFS author's first-pass sampler",2250,460,outputs=[("SAMPLER","SAMPLER")],widgets={"sampler_name":"euler"})
    schedule=g.add("BasicScheduler","BFS 8 steps / simple / full denoise",2250,670,inputs=[("model","MODEL")],outputs=[("SIGMAS","SIGMAS")],widgets={"scheduler":"simple","steps":8,"denoise":1.0});g.connect(shift,0,schedule,"model")
    sample=g.add("SamplerCustomAdvanced","Generate one guide-aligned clip",2700,0,inputs=[("noise","NOISE"),("guider","GUIDER"),("sampler","SAMPLER"),("sigmas","SIGMAS"),("latent_image","LATENT")],outputs=[("output","LATENT"),("denoised_output","LATENT")],size=(350,330))
    for src,field in ((noise,"noise"),(guider,"guider"),(sampler,"sampler"),(schedule,"sigmas")):g.connect(src,0,sample,field)
    g.connect(cond,1,sample,"latent_image")
    dec=g.add("VAEDecode","Decode raw generated frames",3100,0,inputs=[("samples","LATENT"),("vae","VAE")],outputs=[("IMAGE","IMAGE")]);g.connect(sample,0,dec,"samples");g.connect(vae,0,dec,"vae")
    video=g.add("CreateVideo","24 fps with ORIGINAL source audio",3500,0,inputs=[("images","IMAGE"),("audio","AUDIO")],outputs=[("VIDEO","VIDEO")],widgets={"fps":24.0,"bit_depth":"auto","color_space":"sRGB","codec":"none"});g.connect(dec,0,video,"images");g.connect(prep,1,video,"audio")
    save=g.add("H3SwapSaveVideo","RAW BFS result; evaluate before any upscale",3900,0,inputs=[("video","VIDEO")],widgets={"filename_prefix":"h3_character_swap/bfs_v11","crf":16});g.connect(video,0,save,"video")
    compare=g.add("H3SwapComparison","SOURCE left / GENERATED right",3100,340,inputs=[("source","IMAGE"),("generated","IMAGE")],outputs=[("IMAGE","IMAGE")]);g.connect(prep,0,compare,"source");g.connect(dec,0,compare,"generated")
    cv=g.add("CreateVideo","Compare at the original timing",3500,340,inputs=[("images","IMAGE"),("audio","AUDIO")],outputs=[("VIDEO","VIDEO")],widgets={"fps":24.0,"bit_depth":"auto","color_space":"sRGB","codec":"none"});g.connect(compare,0,cv,"images");g.connect(prep,1,cv,"audio")
    cs=g.add("H3SwapSaveVideo","Source / generated comparison MP4",3900,340,inputs=[("video","VIDEO")],widgets={"filename_prefix":"h3_character_swap/bfs_v11_comparison","crf":16});g.connect(cv,0,cs,"video")
    return g.result()


def face_body_studio():
    g = Graph('H3 Face Body Long Analysis 2026-10-27.2')
    video = g.add('LoadVideo', 'SOURCE VIDEO: selected original performer and interval', 0, 0,
        outputs=[('VIDEO','VIDEO')], widgets={'file':'source.mp4'}, size=(350,300))
    face = g.add('LoadImage', 'PICTURE 1: FACE / HEAD / HAIR', 0, 360,
        outputs=[('IMAGE','IMAGE'),('MASK','MASK')],
        widgets={'image':'face.png','upload':'image'}, size=(350,350))
    body = g.add('LoadImage', 'PICTURE 2: BODY / PROPORTIONS / OUTFIT', 0, 770,
        outputs=[('IMAGE','IMAGE'),('MASK','MASK')],
        widgets={'image':'body.png','upload':'image'}, size=(350,450))
    a = g.add('H3SwapAnalyzeFaceBody', 'ONE CHARACTER: separate face and body evidence', 430, 0,
        inputs=[('video','VIDEO'),('face','IMAGE'),('body','IMAGE')],
        outputs=[('clip_prompt','STRING'),('chain_prompt','STRING'),('report','STRING'),('samples','IMAGE')],
        widgets={'target_person':'person','start_seconds':0.,'frames':124,'reuse_cached_draft':True,
                 'detail_level':'Exhaustive','max_output_tokens':16384,'context_tokens':'32768',
                 'sampled_frames':16,'timeout_seconds':1800}, size=(480,600))
    g.connect(video,0,a,'video');g.connect(face,0,a,'face');g.connect(body,0,a,'body')
    for i, title in enumerate(('SELECTED-CLIP PROMPT: only this analyzed interval',
                               'CHAIN-SAFE PROMPT: paste into workflow 09',
                               'REPORT: evidence, uncertainty and actual token use')):
        out = g.add('PreviewAny',title,1000+i*550,0,inputs=[('source','*')],
                    outputs=[('output','*')],size=(520,700))
        g.connect(a,i,out,'source','STRING')
    sheet = g.add('PreviewImage','All SOURCE samples (not a reference collage)',430,680,
                  inputs=[('images','IMAGE')],size=(1080,580))
    g.connect(a,3,sheet,'images')
    return g.result()


def face_body_chain():
    g = Graph('H3 Low Memory Face Body Chain 2026-10-27.2')
    control = g.add('H3SwapDiskChainFaceBody','FACE + BODY: one source performer, two reference photos',0,0,
        outputs=[('status_and_review_link','STRING')],
        widgets={'action':'Start new run','run_name':'face_body_test_01','source_video':'source.mp4',
                 'face_image':'face.png','body_image':'body.png','duration_seconds':10.,'start_seconds':0.,
                 'resolution':'768','window_frames':'124','target_person':'person','selection':'largest',
                 'selection_index':0,'mask_expansion':64,'seed':904239,'review_mode':'Review every chunk',
                 'max_retries':2,'additional_instructions':''},size=(650,850))
    status = g.add('PreviewAny','STATUS: open /_swap/chain on this same service',740,0,
                  inputs=[('source','*')],outputs=[('output','*')],size=(700,650))
    g.connect(control,0,status,'source','STRING')
    note = g.add('Note','Reference roles and first test',740,710,widgets={'text':
        'Upload the full source video and TWO separate photos into /workspace/input. Refresh the file lists. '
        'Picture 1 is the FACE close-up; Picture 2 is the BODY and COMPLETE OUTFIT. '
        'Do NOT combine them into a sheet for this workflow. Use workflow 04 for optional two-image long analysis, '
        'then paste its CHAIN-SAFE PROMPT into additional_instructions. Use the SAME target_person wording. '
        'Leaving that field blank uses the built-in deliberate role prompt. '
        'Check a full-person edit mask first. Queue ONCE, review each candidate at /_swap/chain. '
        'Both references are passed to every short job. This is prompt-directed, not hard face/body control. '
        'Use a NEW run name to change references/settings. Existing single-ref runs remain single-ref. '
        'The first test is 10 seconds; set 48 only after a good face/body and join review.'},size=(700,350))
    canvas, api = g.result()
    api.pop(str(note), None)
    return canvas, api


def build_local(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for filename, pair in (("00_Analyze_Clip_Build_Prompts", prompt_studio()), ("03_BFS_V11_Guide_Comparison", bfs_graph()), ("04_Analyze_Face_Body_Build_Prompts", face_body_studio()), ("09_H3_LowMemory_Face_Body_Chain", face_body_chain())):
        g,api=pair
        atomic_json(output/(filename+".json"),g);atomic_json(output/(filename+".api.json"),api)


def build_all(relay):
    relay=Path(relay)
    dest=ROOT/"workflows"
    build_local(dest)
    source=relay/"example_workflows/toolkit/H3-Character-Swap.json"
    source_api=relay/"example_workflows/toolkit/H3-Character-Swap.api.json"
    graph,api,preview=adapt_akatz(json.loads(source.read_text()),json.loads(source_api.read_text()))
    atomic_json(dest/"01_Akatz_V15_Mask_Check.json",preview)
    atomic_json(dest/"02_Akatz_V15_Native_Replacement.json",graph)
    atomic_json(dest/"02_Akatz_V15_Native_Replacement.api.json",api)
    atomic_json(ROOT/"workflow-provenance.json", {"relay_source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(), "relay_api_sha256":hashlib.sha256(source_api.read_bytes()).hexdigest(), "changes":["canonical adapter filename", "5.166667-second test window", "neutral input filenames", "explicit editable reference-role prompt", "output subfolder", "separate mask-only ancestor graph"], "sampling_and_history":"preserved from pinned developer graph", "bfs":"simplified guide-based first pass, not an unmodified upstream canvas; author low-resolution baseline and Euler/simple eight-step schedule"})

if __name__=="__main__":
    build_local(ROOT/"workflows")
