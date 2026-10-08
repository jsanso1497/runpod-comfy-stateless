"""One-person H3 references: one headshot plus up to eight independent body images.

Reference roles are prompt instructions, not a face-recognition/identity guarantee.
Uses the installed, pinned RefMod pack. No model downloads or public routes here.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid

NONE = "(none)"
MAX_BODIES = 8
PERSON_TYPE = "H3_PERSON_REFERENCES"
ROLE_MARKER = "[SINGLE_PERSON_V1 "
MODES = ["Use instruction (no Ollama)", "Generate with Ollama"]
DEFAULT_INSTRUCTION = (
    "One photorealistic full-body shot of the same person. They stand naturally, "
    "shift their weight slightly and make a small natural head movement. Keep their "
    "whole body and both feet inside the frame. Preserve the clothing chosen by the "
    "reference priorities, against a plain neutral background. Stationary camera, "
    "one continuous shot, no extra people, no speech, "
    "no music, and no on-screen text."
)
SINGLE_PERSON_SYSTEM = Path(__file__).with_name("single_person_system_prompt.txt").read_text(encoding="utf-8")


def registered(name):
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"Required node {name} is unavailable. Deploy the rebuilt multi-reference image.")
    return cls


def call_node(node_type, **kwargs):
    cls = registered(node_type)
    result = getattr(cls(), cls.FUNCTION)(**kwargs)
    if isinstance(result, dict):
        return result["result"]
    return result.result if hasattr(result, "result") else result


def validate_image(image, label):
    import torch
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
        raise ValueError(f"{label}: upload one still RGB image, not an image batch or video.")
    if min(image.shape[1:3]) < 32 or not image.isfinite().all():
        raise ValueError(f"{label}: the image must be at least 32 pixels on each side with finite pixel values.")
    return image


def image_hash(image):
    import torch
    x = image.detach().to(device="cpu", dtype=torch.float32)
    pixels = x.clamp(0, 1).mul(255).round().to(torch.uint8).contiguous().numpy().tobytes()
    return hashlib.sha256(str(tuple(x.shape)).encode("ascii") + pixels).hexdigest()


def clean_label(label):
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_")[:48]
    if not value:
        raise ValueError("Enter a short person label, for example test_person.")
    return value


def collect_references(headshot, person_label, **body_images):
    extras = set(body_images) - {f"body_{i}" for i in range(1, MAX_BODIES + 1)}
    if extras:
        raise ValueError("Unknown body reference slots: " + ", ".join(sorted(extras)))
    images = [validate_image(headshot, "Primary headshot")]
    roles = ["primary_headshot"]
    for i in range(1, MAX_BODIES + 1):
        image = body_images.get(f"body_{i}")
        if image is not None:
            images.append(validate_image(image, f"Body photo {i}"))
            roles.append(f"body_{i}")
    hashes = [image_hash(im) for im in images]
    if len(hashes) != len(set(hashes)):
        raise ValueError("The same image was selected twice. Keep one copy; set unused body slots to (none).")
    return {"images": tuple(images), "roles": tuple(roles), "hashes": tuple(hashes),
            "person_label": clean_label(person_label)}


def reference_ledger(roles):
    if not 1 <= len(roles) <= MAX_BODIES + 1 or roles[0] != "primary_headshot":
        raise ValueError("The first reference must be the primary headshot; use up to eight body references.")
    lines = ["ONE PERSON ONLY. All supplied images show different views of <Subject 1>.",
             "<Picture 1>: primary facial identity, apparent age, facial geometry, hairline and hair."]
    for i, role in enumerate(roles[1:], 2):
        if not re.fullmatch(r"body_[1-8]", role):
            raise ValueError("Unexpected body-reference role.")
        priority = "PRIMARY WARDROBE and body proportions" if i == 2 else "supporting body proportions and alternate viewing angle"
        lines.append(f"<Picture {i}>: {role.replace('_', ' ')} of the SAME person; {priority}.")
    lines.append("Do not make a different person for each reference, blend several identities, or create a collage/reference sheet.")
    lines.append("When faces conflict, the headshot takes priority. Body photos must not change that person's face or age.")
    if len(roles) > 1:
        lines.append("When outfits conflict, <Picture 2> supplies wardrobe unless the instruction explicitly requests a wardrobe change. Do not mix outfits from other body views.")
    else:
        lines.append("Only a headshot was supplied; hidden body proportions are not established by a reference.")
    lines.append("Other reference backgrounds, crops and poses are not mandatory output compositions. Never infer a real person's name or biography.")
    return "\n".join(lines)


def validate_single_person_text(text, count):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter an instruction for this person.")
    if re.search(r"<\s*Subject\s+(?!1\s*>)[0-9]+\s*>", text, flags=re.I):
        raise ValueError("This workflow is for ONE person, <Subject 1>. Use the existing two-person workflow for other subjects.")
    for n in re.findall(r"<\s*Picture\s+(\d+)\s*>", text, flags=re.I):
        if not 1 <= int(n) <= count:
            raise ValueError(f"The prompt mentions a picture that is not supplied. This set contains {count} pictures.")
    if re.search(r"<\s*(?:Video|Audio)\s+\d+\s*>", text, flags=re.I):
        raise ValueError("This workflow has still-image references only, not source video/audio.")


def manual_prompt(roles, instruction):
    validate_single_person_text(instruction, len(roles))
    return ("subject_definitions:\n" + reference_ledger(roles) + "\n\n"
            "summary:\n[reference generation] One person, <Subject 1>, following the requested shot.\n\n"
            "retention_analysis:\nPreserve facial identity from <Picture 1> and use the other supplied views "
            "only for the same person's body and the stated wardrobe priorities. Retention is an instruction, "
            "not a guarantee.\n\n"
            "detailed_description:\n[Shot 1]\n" + instruction.strip())


class QualityOptionalBodyImage:
    @classmethod
    def INPUT_TYPES(cls):
        # Standard image_upload widget, with a real empty state. Uses ComfyUI's upload route.
        native = registered("LoadImage").INPUT_TYPES()["required"]["image"]
        options = [NONE] + [name for name in native[0] if name != NONE]
        return {"required": {"image": (options, {"image_upload": True, "default": NONE})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "Quality/H3 One Person"
    DESCRIPTION = "Optional body reference. Upload one photo or leave (none). Empty slots do not add an image."

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if image in (NONE, "", None):
            return True
        return registered("LoadImage").VALIDATE_INPUTS(image)

    @classmethod
    def IS_CHANGED(cls, image):
        if image in (NONE, "", None):
            return "empty"
        return registered("LoadImage").IS_CHANGED(image)

    def load(self, image):
        if image in (NONE, "", None):
            return (None,)
        pixels = call_node("LoadImage", image=image)[0]
        return (validate_image(pixels, "Body photo"),)


class QualityPersonReferenceSet:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"headshot": ("IMAGE",), "person_label": ("STRING", {"default": "test_person"})},
                "optional": {f"body_{i}": ("IMAGE",) for i in range(1, MAX_BODIES + 1)}}
    RETURN_TYPES = (PERSON_TYPE, "STRING")
    RETURN_NAMES = ("person_references", "reference_map")
    FUNCTION = "collect"
    CATEGORY = "Quality/H3 One Person"
    DESCRIPTION = "One primary headshot plus up to eight body views, all of ONE person. Keeps separate image shapes; never makes a crop-matched batch."
    def collect(self, headshot, person_label, **bodies):
        refs = collect_references(headshot, person_label, **bodies)
        return refs, reference_ledger(refs["roles"])


class QualitySinglePersonPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "references": (PERSON_TYPE,),
            "mode": (MODES, {"default": MODES[0]}),
            "instruction": ("STRING", {"default": DEFAULT_INSTRUCTION, "multiline": True}),
            "length": ("INT", {"default": 124, "min": 124, "max": 362, "step": 17}),
            "variation_seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            "temperature": ("FLOAT", {"default": 0.2, "min": 0, "max": 1, "step": 0.05}),
            "save_prompt": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("STRING", PERSON_TYPE, "INT")
    RETURN_NAMES = ("h3_prompt", "same_references", "length")
    FUNCTION = "build"
    CATEGORY = "Quality/H3 One Person"
    DESCRIPTION = "Binds ALL the head/body images to one subject. Defaults to your instruction, with no LLM. Optional Ollama sees every selected image and unloads before H3."

    def build(self, references, mode, instruction, length, variation_seed, temperature, save_prompt):
        if mode not in MODES:
            raise ValueError("Unknown one-person prompt mode.")
        if length < 124 or length > 362 or length % 17 != 5:
            raise ValueError("Length must be on the H3 17k+5 frame grid between 124 and 362.")
        roles, images = references["roles"], references["images"]
        validate_single_person_text(instruction, len(images))
        if mode == MODES[0]:
            prompt = manual_prompt(roles, instruction)
            meta = {"mode": "instruction_only"}
        else:
            module = importlib.import_module(registered("EverydayOllamaH3Prompt").__module__)
            import comfy.model_management as mm
            cfg = dict(module.settings())
            # More context, not smaller photographs. The existing helper settings are unchanged.
            cfg["context_length"] = 32768
            with module.LOCK:
                mm.throw_exception_if_processing_interrupted()
                mm.unload_all_models()
                mm.soft_empty_cache()
                prompt, meta = module.generate(cfg, SINGLE_PERSON_SYSTEM,
                    reference_ledger(roles) + "\n\nREQUESTED SHOT:\n" + instruction.strip(),
                    list(images), length, variation_seed, temperature, mm.throw_exception_if_processing_interrupted)
            validate_single_person_text(prompt, len(images))
            # The deterministic role ledger stays in the final text, even if the helper under-specifies it.
            prompt = reference_ledger(roles) + "\n\n" + prompt
        if save_prompt:
            import folder_paths
            root = Path(folder_paths.get_output_directory()) / "prompt_assistant"
            root.mkdir(parents=True, exist_ok=True)
            stem = "single_person_" + time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:8]
            (root / (stem + ".txt")).write_text(prompt + "\n", encoding="utf-8")
            meta.update({"frame_count": length, "roles": list(roles), "source_hashes": list(references["hashes"])})
            (root / (stem + ".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return prompt, references, length


def refmod_package():
    cls = registered("MiniMaxH3RefModExtract")
    return cls.__module__.rsplit(".", 1)[0]


def encode_person_bundle(references, vae, ref_resolution, max_total_tokens, interrupt=lambda: None):
    if ref_resolution not in range(256, 2049, 64):
        raise ValueError("Reference resolution must be 256-2048, in 64-pixel increments.")
    if not isinstance(max_total_tokens, int) or max_total_tokens <= 0:
        raise ValueError("Set a positive token budget. Overflow stops; no views will be silently dropped.")
    roles, images = references["roles"], references["images"]
    reference_ledger(roles)
    settings_key = "full-v1-" + str(ref_resolution)
    fingerprint = hashlib.sha256((settings_key + json.dumps(list(zip(roles, references["hashes"])))).encode()).hexdigest()[:16]
    bundle_name = references["person_label"] + "_" + fingerprint
    mods = []
    tokens = 0
    for index, (role, image) in enumerate(zip(roles, images)):
        interrupt()
        # One image per extraction is essential: multi-image Extract would anchor
        # every crop to the headshot canvas and turn the set into one video-kind mod.
        description = f"{ROLE_MARKER}{role}] One view of the SAME person; "
        description += "primary facial identity" if index == 0 else "supporting body/wardrobe reference, not a separate person"
        output = call_node("MiniMaxH3RefModExtract", name=f"{bundle_name}_{role}",
            mode="Full Reference", refs_image={"ref_image_0": image}, vae=vae,
            ref_resolution=ref_resolution, pool_h=64, pool_w=64, latent_frames=1,
            identity=0, multiplier=1, max_tokens=max_total_tokens, description=description,
            save=False, concept_type="identity", background_retention=1.0,
            merge=False, motion_only=False, extraction_preset="manual", budget_policy="error")
        new = output[0]
        if len(new) != 1 or new[0][0].kind != "image" or new[0][1] != 1.0:
            raise RuntimeError("RefMod returned an unexpected reference layout. No bundle was saved.")
        mod, _ = new[0]
        tokens += int(mod.token_count)
        if tokens > max_total_tokens:
            raise ValueError(f"References require {tokens} tokens, above the {max_total_tokens} budget. No bundle saved; no views truncated. Use fewer references or explicitly adjust resolution/budget.")
        mods.extend(new)
    return mods, bundle_name, tokens


class QualityCreatePersonRefMod:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"references": (PERSON_TYPE,), "vae": ("VAE",),
            "ref_resolution": ("INT", {"default": 2048, "min": 256, "max": 2048, "step": 64}),
            "max_total_tokens": ("INT", {"default": 131072, "min": 1024, "max": 1048576, "step": 1024}),
            "save": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("person_mods", "saved_details")
    FUNCTION = "create"
    OUTPUT_NODE = True
    CATEGORY = "Quality/H3 One Person"
    DESCRIPTION = "Independently Full-encodes each photo, then saves ONE ordered bundle with separate shapes. No pooling, common-canvas crop, extra copies, or silent truncation."
    def create(self, references, vae, ref_resolution, max_total_tokens, save):
        import comfy.model_management as mm
        mods, name, tokens = encode_person_bundle(references, vae, ref_resolution, max_total_tokens,
                                                 mm.throw_exception_if_processing_interrupted)
        path = "Not saved (in-memory references only)"
        if save:
            import folder_paths
            roots = folder_paths.get_folder_paths("refmods")
            root = Path(roots[0]) if roots else Path(folder_paths.models_dir) / "refmods"
            parent = root / "identities"
            if parent.is_symlink() or not parent.resolve().is_relative_to(root.resolve()):
                raise ValueError("RefMod destination must remain inside the registered reference folder.")
            parent.mkdir(parents=True, exist_ok=True)
            destination = parent / name
            if destination.with_suffix(".safetensors").is_symlink():
                raise ValueError("Refusing a symlink RefMod destination.")
            bundle = importlib.import_module(refmod_package() + ".bundle")
            path = bundle.save_bundle(str(destination), name, mods)
        details = f"{len(mods)} views of ONE person; {tokens} reference tokens.\n{path}\n" + reference_ledger(references["roles"])
        return {"ui": {"text": [details]}, "result": (mods, details)}


class QualityPersonNativeConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "references": (PERSON_TYPE,),
            "prompt": ("STRING", {"forceInput": True}),
            "width": ("INT", {"default": 1344, "min": 32, "max": 4096, "step": 32}),
            "height": ("INT", {"default": 768, "min": 32, "max": 4096, "step": 32}),
            "length": ("INT", {"default": 124, "min": 124, "max": 362, "step": 17})}}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("text_vision_conditioning", "latent")
    FUNCTION = "encode"
    CATEGORY = "Quality/H3 One Person"
    DESCRIPTION = "Sends all ORIGINAL images, in headshot-first order, to H3's native vision encoder. Visual VAE references are supplied by RefMod exactly once downstream."
    def encode(self, clip, references, prompt, width, height, length):
        validate_single_person_text(prompt, len(references["images"]))
        return call_node("MiniMaxH3ReferenceToVideo", clip=clip, prompt=prompt,
            width=width, height=height, length=length, ref_image_size="max", vae=None,
            ref_images={f"ref_image_{i}": im for i, im in enumerate(references["images"])})


def saved_roles(mods):
    if not 1 <= len(mods) <= MAX_BODIES + 1:
        raise ValueError("Select ONE person bundle made with 08_H3_Create_Person_RefMod. Leave every other loader slot at (none).")
    roles = []
    prefixes = set()
    for mod, strength in mods:
        if mod.kind != "image" or not math.isfinite(strength) or strength != 1.0:
            raise ValueError("Use one image-reference bundle with strength 1.0 and copies 1, not a compressed/stacked/video reference.")
        match = re.match(r"\[SINGLE_PERSON_V1 (primary_headshot|body_[1-8])\]", mod.description or "")
        if not match:
            raise ValueError("This file has no headshot/body role map. Create it with 08_H3_Create_Person_RefMod first.")
        role = match.group(1)
        suffix = "_" + role
        if not mod.name.endswith(suffix):
            raise ValueError("Saved reference name/role mismatch. Recreate the person bundle.")
        prefixes.add(mod.name[:-len(suffix)])
        roles.append(role)
    if len(prefixes) != 1:
        raise ValueError("References came from different saved sets. Choose one complete person bundle.")
    body_numbers = [int(role.rsplit("_", 1)[1]) for role in roles[1:] if role.startswith("body_")]
    if body_numbers != sorted(body_numbers):
        raise ValueError("Body-reference order changed. Choose the complete bundle without reordering its members.")
    if len(set(roles)) != len(roles):
        raise ValueError("Duplicate roles detected. Select the person bundle once, with copies 1.")
    reference_ledger(roles)
    return roles


class QualitySavedPersonPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mods": ("H3_REF_MODS",),
            "instruction": ("STRING", {"default": DEFAULT_INSTRUCTION, "multiline": True})}}
    RETURN_TYPES = ("H3_REF_MODS", "STRING", "STRING")
    RETURN_NAMES = ("checked_person_mods", "h3_prompt", "reference_map")
    FUNCTION = "build"
    CATEGORY = "Quality/H3 One Person"
    DESCRIPTION = "Checks that the selected bundle is one headshot plus body views. Builds ONE-subject text with explicit face/wardrobe priorities."
    def build(self, mods, instruction):
        roles = saved_roles(mods)
        return mods, manual_prompt(roles, instruction), reference_ledger(roles)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    QualityOptionalBodyImage, QualityPersonReferenceSet, QualitySinglePersonPrompt,
    QualityCreatePersonRefMod, QualityPersonNativeConditioning, QualitySavedPersonPrompt)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QualityOptionalBodyImage": "Optional Body Photo (upload or none)",
    "QualityPersonReferenceSet": "ONE Person: Headshot + Body References",
    "QualitySinglePersonPrompt": "ONE Person: Reference-Aware Prompt",
    "QualityCreatePersonRefMod": "Create ONE-Person Multi-Image RefMod",
    "QualityPersonNativeConditioning": "H3 Original Images: ONE Person",
    "QualitySavedPersonPrompt": "Saved Person: Face + Body Role Map"}
