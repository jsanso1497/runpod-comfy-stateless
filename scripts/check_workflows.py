#!/usr/bin/env python3
"""Check graph links, dual-reference wiring, and real ComfyUI node schemas. Never queues inference."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def load(path):
    return json.loads(Path(path).read_text())


def validate_graph(graph: dict) -> None:
    nodes = {n["id"]: n for n in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Duplicate node ids")
    links = {l[0]: l for l in graph["links"]}
    if len(links) != len(graph["links"]):
        raise ValueError("Duplicate link ids")
    for lid, source, si, target, ti, kind in links.values():
        out = nodes[source]["outputs"][si]
        inp = nodes[target]["inputs"][ti]
        if out["type"] != kind or inp["type"] != kind or lid not in out["links"] or inp["link"] != lid:
            raise ValueError(f"Inconsistent link {lid}")
        if nodes[source]["order"] >= nodes[target]["order"]:
            raise ValueError("Invalid dependency order")
    for n in nodes.values():
        for slot, inp in enumerate(n.get("inputs", [])):
            lid = inp.get("link")
            if lid is not None and (lid not in links or links[lid][3:5] != [n["id"], slot]):
                raise ValueError("Dangling input link")
        for slot, out in enumerate(n.get("outputs", [])):
            for lid in out.get("links") or []:
                if lid not in links or links[lid][1:3] != [n["id"], slot]:
                    raise ValueError("Dangling output link")
    if graph["last_node_id"] < max(nodes) or graph["last_link_id"] < max(links, default=0):
        raise ValueError("Incorrect graph counters")
    if graph.get("extra", {}).get("ollama_h3"):
        builders = [n for n in nodes.values() if n["type"] == "EverydayOllamaH3Prompt"]
        if len(builders) != 1:
            raise ValueError("Exactly one ordered-reference prompt builder is required")
        b = builders[0]
        def src(node, field):
            inp = next(i for i in node["inputs"] if i["name"] == field)
            return tuple(links[inp["link"]][1:3]) if inp.get("link") else None
        for h3 in [n for n in nodes.values() if n["type"] == "MiniMaxH3ReferenceToVideo"]:
            for field, slot in [("prompt", 0), ("ref_images.ref_image_0", 1), ("ref_images.ref_image_1", 2), ("length", 3)]:
                if src(h3, field) != (b["id"], slot):
                    raise ValueError("H3 must receive the builder's exact prompt, reference order and length")
            gate = next(n for n in nodes.values() if n["type"] == "EverydayH3FilesAfterPrompt")
            if src(gate, "prompt_ready") != (b["id"], 0):
                raise ValueError("Model filenames must depend on the completed prompt")
            for n in nodes.values():
                if n["type"] in ("UNETLoader", "CLIPLoader", "VAELoader"):
                    field = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name"}[n["type"]]
                    if src(n, field)[0] != gate["id"]:
                        raise ValueError("H3 model loaders must be gated until Ollama unloads")
    if graph.get("extra", {}).get("refmod_quality") == "original_pixels_to_qwen":
        refnode = next(n for n in nodes.values() if n["type"] == "MiniMaxH3ReferenceToVideo")
        if any(i["name"] == "vae" and i.get("link") is not None for i in refnode["inputs"]):
            raise ValueError("RefMod workflow must not attach native VAE reference latents a second time")
        extracts = [n for n in nodes.values() if n["type"] == "MiniMaxH3RefModExtract"]
        applies = [n for n in nodes.values() if n["type"] == "MiniMaxH3RefModApply"]
        if len(extracts) != 2 or len(applies) != 2:
            raise ValueError("Two references require two individual full RefMods and exactly two Apply nodes")
        expected_slots = {1, 2}
        actual_slots = set()
        for ex in extracts:
            w = ex["widgets_values_named"]
            if not (w["mode"] == "Full Reference" and w["ref_resolution"] == 2048
                    and w["extraction_preset"] == "manual" and w["multiplier"] == 1
                    and w["identity"] == 0 and not w["merge"] and not w["motion_only"]
                    and w["budget_policy"] == "error"):
                raise ValueError("RefMod identity path must not compress, scramble, repeat or truncate references")
            ref_input = next(i for i in ex["inputs"] if i["name"] == "refs_image.ref_image_0")
            lk = links[ref_input["link"]]
            if nodes[lk[1]]["type"] != "EverydayOllamaH3Prompt":
                raise ValueError("RefMod source must be the exact original image passed by the prompt builder")
            actual_slots.add(lk[2])
        if actual_slots != expected_slots:
            raise ValueError("RefMod subject order must include picture 1 and picture 2 exactly once")
        for ap in applies:
            w = ap["widgets_values_named"]
            if not (w["retention"] == 1.0 and w["curve_direction"] == "constant"
                    and w["curve_shape"] == "linear" and w["curve_value"] == 1.0
                    and w["scramble_seed"] == -1 and not w["override"]):
                raise ValueError("RefMod default retention must be unweakened and ordered")
    if graph.get("extra", {}).get("refmod_quality") == "saved_refs_attached_once":
        if any(n["type"] == "MiniMaxH3RefModApply" for n in nodes.values()):
            raise ValueError("Saved RefMod Text Encode already attaches references; never apply them twice")
    single_person = graph.get("extra", {}).get("single_person_h3")
    if single_person:
        def instances(kind):
            return [n for n in nodes.values() if n["type"] == kind]
        def source_of(n, field):
            inp = next(i for i in n["inputs"] if i["name"] == field)
            return tuple(links[inp["link"]][1:3]) if inp.get("link") else None
        if single_person != "saved":
            sets = instances("QualityPersonReferenceSet")
            if len(sets) != 1 or not source_of(sets[0], "headshot"):
                raise ValueError("One primary headshot/reference set is required")
            if len(instances("QualityOptionalBodyImage")) != 8:
                raise ValueError("The person workflow must expose eight optional body slots")
        for creator in instances("QualityCreatePersonRefMod"):
            vals = creator["widgets_values_named"]
            if vals["ref_resolution"] != 2048 or vals["max_total_tokens"] != 131072 or not vals["save"]:
                raise ValueError("Supplied one-person defaults retain full-resolution views and save the bundle")
        if single_person == "originals":
            prompt = instances("QualitySinglePersonPrompt")[0]
            enc = instances("QualityPersonNativeConditioning")[0]
            creator = instances("QualityCreatePersonRefMod")[0]
            for n in (enc, creator):
                if source_of(n, "references") != (prompt["id"], 1):
                    raise ValueError("Original photos and RefMods must share the completed prompt's exact ordered set")
            if source_of(enc, "prompt") != (prompt["id"], 0) or source_of(enc, "length") != (prompt["id"], 2):
                raise ValueError("H3 prompt and length must follow the one-person prompt builder")
            applies = instances("MiniMaxH3RefModApply")
            if len(applies) != 1:
                raise ValueError("Apply the one-person reference bundle exactly once")
            a = applies[0]
            if source_of(a, "conditioning") != (enc["id"], 0) or source_of(a, "mods") != (creator["id"], 0):
                raise ValueError("The one-person Apply node must receive original-pixel conditioning and the same bundle")
            vals = a["widgets_values_named"]
            if vals["retention"] != 1 or vals["curve_direction"] != "constant" or vals["scramble_seed"] != -1:
                raise ValueError("Do not weaken or scramble the supplied one-person reference baseline")
            if source_of(instances("BasicGuider")[0], "conditioning") != (a["id"], 0):
                raise ValueError("The video sampler must receive the reference-applied conditioning")
        if single_person in ("originals", "saved"):
            ptype = "QualitySinglePersonPrompt" if single_person == "originals" else "QualitySavedPersonPrompt"
            p = instances(ptype)[0]
            gate = instances("EverydayH3FilesAfterPrompt")[0]
            if source_of(gate, "prompt_ready") != (p["id"], 0 if single_person == "originals" else 1):
                raise ValueError("H3 loaders must wait for the one-person prompt")
            for loader in instances("UNETLoader") + instances("CLIPLoader") + instances("VAELoader"):
                field = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name"}[loader["type"]]
                origin = source_of(loader, field)
                if not origin or origin[0] != gate["id"]:
                    raise ValueError("Every H3 loader must remain gated after the prompt")
            if instances("BasicScheduler")[0]["widgets_values_named"]["steps"] != 25:
                raise ValueError("Keep the supplied 25-step quality baseline")
        if single_person == "saved":
            if instances("MiniMaxH3RefModApply"):
                raise ValueError("Saved-ref encoding already attaches references; do not apply them twice")
            check = instances("QualitySavedPersonPrompt")[0]
            enc = instances("MiniMaxH3RefModTextEncode")[0]
            if source_of(enc, "mods") != (check["id"], 0) or source_of(enc, "prompt") != (check["id"], 1):
                raise ValueError("Saved person roles and references must stay paired")
    profile = graph.get("extra", {}).get("everyday", {}).get("profile")
    rebalance_nodes = [n for n in nodes.values() if "Rebalance" in n["type"]]
    if rebalance_nodes and profile != "krea2":
        raise ValueError("Rebalance is allowed only in the Krea 2 recipe workflows")
    if profile != "krea2":
        return
    patches = [n for n in nodes.values() if n["type"] == "Krea2EditModelPatch"]
    encoders = [n for n in nodes.values() if n["type"] == "Krea2EditGroundedEncode"]
    sampler = next(n for n in nodes.values() if n["type"] == "KSampler")
    if len(patches) != 1 or len(encoders) != 2:
        raise ValueError("Krea identity editing requires its model patch and two grounded encoders")
    def source(n, name):
        inp = next(i for i in n["inputs"] if i["name"] == name)
        return tuple(links[inp["link"]][1:3]) if inp["link"] else None
    patch = patches[0]
    if not source(patch, "target_latent") or source(patch, "target_latent") != source(sampler, "latent_image"):
        raise ValueError("The patch and sampler must use the same target latent")
    for field, semantic in [("source_image", "image"), ("source_image_b", "image_b")]:
        if any(source(patch, field) != source(enc, semantic) for enc in encoders):
            raise ValueError("Appearance and semantic references must match, in the same order")
    if graph.get("extra", {}).get("krea_rebalance") == "grounded_positive_and_negative":
        scalers = [n for n in nodes.values() if n["type"] == "ConditioningKrea2Rebalance"]
        if len(scalers) != 2 or len(rebalance_nodes) != 2:
            raise ValueError("Krea needs exactly two per-layer Rebalance nodes, not an alternate editing schedule")
        seen_encoders = set()
        seen_settings = []
        for branch in ("positive", "negative"):
            origin = source(sampler, branch)
            if not origin or nodes[origin[0]]["type"] != "ConditioningKrea2Rebalance":
                raise ValueError("Both Krea sampler branches must pass through Krea Rebalance")
            scaler = nodes[origin[0]]
            if scaler["mode"] != 0:
                raise ValueError("The supplied Krea Rebalance nodes must be active")
            grounded = source(scaler, "conditioning")
            if not grounded or nodes[grounded[0]]["type"] != "Krea2EditGroundedEncode":
                raise ValueError("Rebalance must follow the original image-grounded encoder")
            seen_encoders.add(grounded[0])
            if branch == "negative" and nodes[grounded[0]]["widgets_values_named"]["prompt"] != "":
                raise ValueError("Raw/CFG3 identity editing uses an empty, image-grounded negative")
            settings = scaler["widgets_values_named"]
            import math
            weights = [float(x.strip()) for x in settings["per_layer_weights"].split(",")]
            if len(weights) != 12 or not all(math.isfinite(x) and x > 0 for x in weights):
                raise ValueError("Keep all twelve Krea encoder layer bands present and finite")
            if not math.isfinite(settings["multiplier"]) or settings["multiplier"] <= 0:
                raise ValueError("Krea global conditioning multiplier must be positive and finite")
            seen_settings.append(settings)
        if len(seen_encoders) != 2 or seen_settings[0] != seen_settings[1]:
            raise ValueError("Positive/negative must remain separately grounded and use matching layer scaling")
    required_lora = [n for n in nodes.values() if n["type"] == "LoraLoaderModelOnly" and n["widgets_values_named"]["lora_name"] == "krea2_identity_edit_v1_2.safetensors"]
    if len(required_lora) != 1 or required_lora[0]["mode"] != 0 or required_lora[0]["widgets_values_named"]["strength_model"] != 1.0:
        raise ValueError("Required identity adapter is absent or disabled")


def graph_paths(config: Path, active_only=False):
    from catalog import selected_profiles, workflow_sources
    runtime = load(config / "runtime.json")
    active = (selected_profiles(config) if active_only else set(runtime["profiles"])) | {"shared"}
    # Only this update's graphs have the extra.everyday marker.
    return [p for p in workflow_sources(config)
            if load(p).get("extra", {}).get("everyday", {}).get("profile") in active]


def validate_schema(graph: dict, objects: dict, check_weights: bool) -> None:
    for n in graph["nodes"]:
        kind = n["type"]
        if kind in ("Note", "MarkdownNote"):
            continue
        if kind not in objects:
            raise ValueError(f"Missing registered node: {kind}")
        spec = objects[kind]
        groups = spec.get("input", {})
        required = groups.get("required", {})
        allowed = {**required, **groups.get("optional", {})}
        connected = {i["name"] for i in n.get("inputs", []) if i.get("link") is not None}
        widgets = n.get("widgets_values_named", {})
        provided = connected | set(widgets)
        for name in required:
            if name not in provided and not any(s.startswith(name + ".") for s in provided):
                raise ValueError(f"{kind} is missing required input {name}")
        for name in connected:
            if name not in allowed and name.split(".")[0] not in allowed:
                # V3 autogrow inputs can be flattened into their template input name.
                if not ((kind == "MiniMaxH3ReferenceToVideo" and name.startswith("ref_")) or (kind == "MiniMaxH3RefModExtract" and name.startswith("refs_image.ref_image_"))):
                    raise ValueError(f"Unknown socket: {kind}.{name}")
        for name, value in widgets.items():
            if name in {"control_after_generate", "upload"}:
                continue
            if name not in allowed:
                raise ValueError(f"Unknown widget: {kind}.{name}")
            info = allowed[name]
            choices = info[0]
            is_weight = name in {"unet_name", "clip_name", "vae_name", "lora_name"}
            if name == "image" or (is_weight and not check_weights):
                continue
            if isinstance(choices, list) and value not in choices:
                raise ValueError(f"Invalid or missing value for {kind}.{name}: {value}")
        outs = spec.get("output", [])
        if len(outs) < len(n.get("outputs", [])):
            raise ValueError(f"Output count mismatch: {kind}")
        for i, out in enumerate(n.get("outputs", [])):
            # MatchType is resolved by the frontend to the connected CONDITIONING type.
            # Restrict the allowance to this known polymorphic output, not arbitrary nodes.
            if kind == "MiniMaxH3RefModApply" and i == 0:
                if out["type"] != "CONDITIONING" or n["inputs"][0]["type"] != "CONDITIONING":
                    raise ValueError("RefMod Apply must preserve native CONDITIONING type")
                continue
            if outs[i] != out["type"]:
                raise ValueError(f"Output type mismatch: {kind}[{i}]")


def check_server(config: Path, port: int, timeout: int, active_only: bool, check_weights: bool) -> None:
    import requests
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    objects = None
    while time.monotonic() < deadline:
        try:
            r = requests.get(base + "/object_info", timeout=15)
            if r.status_code == 200:
                objects = r.json()
                break
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2)
    if objects is None:
        raise RuntimeError("ComfyUI did not return /object_info before the timeout")
    for path in graph_paths(config, active_only):
        graph = load(path)
        validate_graph(graph)
        validate_schema(graph, objects, check_weights)
        print(f"WORKFLOW SCHEMA PASS: {path.name}", flush=True)
    print("Schema/registration checks only. No GPU generation or identity-fidelity test has run.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["static", "server", "build-smoke"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--comfy-home", type=Path)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    for path in graph_paths(args.config):
        validate_graph(load(path))
    if args.command == "static":
        print("Static graph checks: PASS")
        return
    if args.command == "server":
        check_server(args.config, args.port, args.timeout, True, True)
        return
    if args.comfy_home is None:
        parser.error("build-smoke needs --comfy-home")
    logfile = args.comfy_home / "build-smoke.log"
    with logfile.open("w") as log:
        child = subprocess.Popen([sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port),
                                  "--disable-auto-launch", "--enable-manager", "--use-pytorch-cross-attention"],
                                 cwd=args.comfy_home, stdout=log, stderr=subprocess.STDOUT)
        try:
            check_server(args.config, args.port, args.timeout, False, False)
        except Exception:
            log.flush()
            print("\n".join(logfile.read_text(errors="replace").splitlines()[-100:]), file=sys.stderr)
            raise
        finally:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"WORKFLOW CHECK FAILED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
