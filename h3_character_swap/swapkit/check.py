"""Static and live-schema checks. These never claim GPU or visual validation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
from .config import ROOT, variants
from .graphs import validate_graph


def graph_nodes(g):
    special={s["id"] for s in g.get("definitions",{}).get("subgraphs",[])}
    result={n["type"] for n in g["nodes"] if n["type"] not in special | {"Note","MarkdownNote","Reroute","PrimitiveNode"}}
    for s in g.get("definitions",{}).get("subgraphs",[]): result |= graph_nodes(s)
    return result


def static_report():
    files=list((ROOT/"workflows").glob("*.json"))
    canvases=[p for p in files if not p.name.endswith(".api.json")]
    if len(canvases)<4: raise ValueError("Build must include prompt studio, mask review, Akatz and BFS graphs")
    for p in canvases: validate_graph(json.loads(p.read_text()))
    locked=json.loads((ROOT/"assets.lock.json").read_text())["assets"]
    for row in locked:
        if len(row["sha256"])!=64 or len(row["revision"])!=40 or row["bytes"]<=0:
            raise ValueError("Unpinned runtime asset")
    return {"ok":True,"canvases":len(canvases),"locked_assets":len(locked),"gpu_tested":False,"scope":"JSON topology and metadata locks only"}


def inspect_schema(schema, selected=("akatz",), check_models=True):
    needed=set()
    for p in (ROOT/"workflows").glob("*.json"):
        if p.name.endswith(".api.json") or ("BFS" in p.name and "bfs" not in selected) or ("Akatz" in p.name and "akatz" not in selected): continue
        needed |= graph_nodes(json.loads(p.read_text()))
    missing=sorted(needed-set(schema))
    problems=[]
    if check_models:
        rows=json.loads((ROOT/"assets.lock.json").read_text())["assets"]
        lookup={"diffusion_models":("UNETLoader","unet_name"),"text_encoders":("CLIPLoader","clip_name"),"vae":("VAELoader","vae_name"),"loras":("LoraLoaderModelOnly","lora_name"),"checkpoints":("CheckpointLoaderSimple","ckpt_name")}
        for row in rows:
            if row["group"] not in {"common"}|set(selected): continue
            category,relative=row["destination"].split("/",1)
            kind,field=lookup[category]
            if kind not in schema: continue
            inputs={}
            for group in ("required","optional"): inputs.update(schema[kind].get("input",{}).get(group,{}))
            choices=inputs.get(field,[[]])[0]
            if not isinstance(choices,list) or relative not in [str(x).replace("\\","/") for x in choices]:
                problems.append("Model not visible: "+row["destination"])
    return {"ok":not missing and not problems,"missing_nodes":missing,"problems":problems,"gpu_tested":False,"scope":"Registered nodes and model-menu visibility; not inference"}


def live_report(selected=None,url="http://127.0.0.1:8189/object_info",check_models=True):
    if selected is None:selected=variants()
    with urllib.request.urlopen(url,timeout=60) as r: schema=json.load(r)
    return inspect_schema(schema,selected,check_models)


def smoke():
    """CI starts the built ComfyUI on CPU and checks real native/custom-node imports."""
    with tempfile.TemporaryDirectory(prefix="h3-swap-smoke-") as d:
        log=Path(d)/"comfy.log"
        with log.open("w") as h:
            p=subprocess.Popen([sys.executable,"/opt/ComfyUI/main.py","--cpu","--listen","127.0.0.1","--port","8289","--user-directory",str(Path(d)/"user"),"--disable-auto-launch"],cwd="/opt/ComfyUI",stdout=h,stderr=subprocess.STDOUT)
            try:
                for _ in range(180):
                    if p.poll() is not None: raise ValueError("CPU ComfyUI startup failed")
                    try:
                        report=live_report({"akatz","bfs"},"http://127.0.0.1:8289/object_info",False)
                        if not report["ok"]:raise ValueError(json.dumps(report))
                        report["scope"]="Built-image CPU startup and registered-node imports; no model weights or GPU inference"
                        print(json.dumps(report,indent=2));return
                    except (OSError,TimeoutError): time.sleep(1)
                raise ValueError("CPU schema smoke test timed out")
            except Exception:
                print(log.read_text(errors="replace")[-24000:])
                raise
            finally:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill()


def main():
    p=argparse.ArgumentParser();p.add_argument("--static",action="store_true");p.add_argument("--live",action="store_true");p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    if a.smoke:smoke();return
    report=live_report() if a.live else static_report()
    print(json.dumps(report,indent=2));raise SystemExit(0 if report["ok"] else 1)

if __name__=="__main__":main()
