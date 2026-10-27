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
    """Collect backend node types, accepting absent/null optional definitions.

    Some frontend exports contain ``"definitions": null`` even without
    subgraphs. Do not treat that as a missing-node failure. Malformed
    non-null values still fail, and every real node remains schema-checked.
    Collect subgraph IDs across this document before filtering node types,
    so a nested subgraph may reference another definition in the same file.
    """
    types, subgraph_ids = set(), set()

    def visit(graph, location):
        if not isinstance(graph, dict):
            raise ValueError(f"{location} must be a workflow object")
        nodes = graph.get("nodes")
        if not isinstance(nodes, list):
            raise ValueError(f"{location}.nodes must be a list")
        for index, node in enumerate(nodes):
            if not isinstance(node, dict) or not isinstance(node.get("type"), str) or not node["type"]:
                raise ValueError(f"{location}.nodes[{index}] must have a nonempty type")
            types.add(node["type"])

        definitions = graph.get("definitions")
        if definitions is None:
            return
        if not isinstance(definitions, dict):
            raise ValueError(f"{location}.definitions must be an object or null")
        subgraphs = definitions.get("subgraphs")
        if subgraphs is None:
            return
        if not isinstance(subgraphs, list):
            raise ValueError(f"{location}.definitions.subgraphs must be a list or null")
        for index, subgraph in enumerate(subgraphs):
            child = f"{location}.definitions.subgraphs[{index}]"
            if not isinstance(subgraph, dict) or not isinstance(subgraph.get("id"), str) or not subgraph["id"]:
                raise ValueError(f"{child} must have a nonempty subgraph id")
            subgraph_ids.add(subgraph["id"])
            visit(subgraph, child)

    visit(g, "workflow")
    return types - subgraph_ids - {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}


def static_report():
    files=list((ROOT/"workflows").glob("*.json"))
    canvases=[p for p in files if not p.name.endswith(".api.json")]
    if len(canvases)<4: raise ValueError("Build must include prompt studio, mask review, Akatz and BFS graphs")
    for p in canvases:
        canvas = json.loads(p.read_text())
        validate_graph(canvas)
        # Exercise the same traversal used by the live check during the build.
        graph_nodes(canvas)
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
    if (ROOT/"workflows/06_H3_LowMemory_Chain_Control.json").exists():
        # The launcher queues these internal nodes later; check them during
        # the image smoke test as well, not only when a paid GPU run starts.
        needed |= {"H3SwapDiskChain", "H3SwapDiskSave", "H3SwapDiskHistory", "H3SwapClipPrep", "PreviewAny"}
        recipe = ROOT/"swapkit/data/akatz_disk_base.api.json"
        if recipe.exists():
            needed |= {n["class_type"] for n in json.loads(recipe.read_text()).values()}
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
        # The pinned ComfyUI CLI rejects a nonexistent --user-directory.
        # Prepare it before spawning, just as the normal Pod runtime does.
        user_dir = Path(d) / "user"
        user_dir.mkdir(parents=True, exist_ok=True)
        log=Path(d)/"comfy.log"
        with log.open("w") as h:
            p=subprocess.Popen([sys.executable,"/opt/ComfyUI/main.py","--cpu","--listen","127.0.0.1","--port","8289","--user-directory",str(user_dir),"--disable-auto-launch"],cwd="/opt/ComfyUI",stdout=h,stderr=subprocess.STDOUT)
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
