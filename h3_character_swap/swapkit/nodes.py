"""Small input/output and prompt-review helpers; H3 inference remains native."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from . import media, prompts
from .config import atomic_json, data_root


class H3SwapClipWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2_592_000.0, "step": 0.01}), "frames": ("INT", {"default": 124, "min": 22, "max": 362, "step": 17}), "short_edge": ([384, 576, 768], {"default": 384})}}
    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("source_frames", "source_audio", "width", "height", "length", "clip_report")
    FUNCTION = "prepare"
    CATEGORY = "H3 Character Replacement"

    def prepare(self, video, start_seconds, frames, short_edge):
        import torch
        import comfy.model_management as mm
        p, base_start, base_duration = media.source_file(video)
        print("[H3 SWAP] Preparing one bounded, frame-aligned guide window", flush=True)
        images, audio, report = media.decode_clip(p, start_seconds, frames, int(short_edge), base_start, base_duration, mm.throw_exception_if_processing_interrupted)
        return torch.from_numpy(images), {"waveform": torch.from_numpy(audio)[None], "sample_rate": 48000}, report["width"], report["height"], report["frames"], json.dumps(report, indent=2)


class H3SwapAnalyzeClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "character": ("IMAGE",), "target_person": ("STRING", {"default": "the main performer", "multiline": True}), "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2_592_000.0, "step": 0.01}), "frames": ("INT", {"default": 124, "min": 22, "max": 362, "step": 17}), "wardrobe": (["Reference clothing", "Source clothing (experimental)"],), "reuse_cached_draft": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("akatz_prompt_to_review", "bfs_character_details_to_review", "analysis_report", "sampled_source_contact_sheet")
    FUNCTION = "analyze"
    CATEGORY = "H3 Character Replacement"

    def analyze(self, video, character, target_person, start_seconds, frames, wardrobe, reuse_cached_draft):
        import torch
        import comfy.model_management as mm
        from PIL import Image
        p, base_start, base_duration = media.source_file(video)
        if character.ndim != 4 or character.shape[0] != 1 or character.shape[-1] not in (3, 4):
            raise ValueError("Use one authoritative reference image")
        if not torch.isfinite(character).all():
            raise ValueError("Reference image contains invalid pixels")
        arr = (character[0, ..., :3].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        ref = Image.fromarray(arr, "RGB")
        model = os.environ.get("H3_SWAP_PROMPT_MODEL", "qwen3-vl:8b")
        try:
            model_record = json.loads((data_root() / "h3-swap/prompt-model.json").read_text())
            digest = (model_record.get("record") or {}).get("digest", "unrecorded")
        except (OSError, ValueError):
            digest = "unrecorded"
        key = prompts.cache_key(p, arr.tobytes(), base_start + start_seconds, frames, target_person, wardrobe, model, digest)
        folder = data_root() / "output/h3_character_swap/prompt_drafts"
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / (key + ".json")
        images, times, report = media.sample_frames(p, start_seconds, frames, base_start, base_duration)
        if reuse_cached_draft and file.is_file():
            analysis = prompts.parse_analysis(json.dumps(json.loads(file.read_text())["analysis"]))
            report["cached_draft"] = True
        else:
            mm.throw_exception_if_processing_interrupted()
            mm.unload_all_models()
            mm.soft_empty_cache()
            analysis = prompts.query([ref] + images, times, target_person, mm.throw_exception_if_processing_interrupted)
            report["cached_draft"] = False
        akatz = prompts.build_akatz(analysis, target_person, wardrobe)
        details = analysis["character_details"]
        report.update(analysis)
        report["review_required"] = True
        report["bfs_wardrobe_note"] = "BFS's trained template replaces body and clothing with the reference character; source-clothing mode applies only to the Akatz draft."
        atomic_json(file, {"analysis": analysis, "report": report, "akatz_prompt": akatz, "bfs_character_details": details})
        (folder / (key + "_akatz.txt")).write_text(akatz + "\n")
        (folder / (key + "_bfs_details.txt")).write_text(details + "\n")
        sheet = prompts.contact_sheet(images, times)
        sheet.save(folder / (key + "_source_samples.jpg"), quality=95)
        return akatz, details, json.dumps(report, indent=2), torch.from_numpy(np.asarray(sheet).copy().astype(np.float32) / 255.0)[None]


class H3SwapBFSPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"character_details": ("STRING", {"default": "person with the face, hair, physique and clothing visible in the character reference", "multiline": True})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "compose"
    CATEGORY = "H3 Character Replacement"

    def compose(self, character_details):
        return (prompts.bfs_prompt(character_details),)


class H3SwapComparison:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("IMAGE",), "generated": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compare"
    CATEGORY = "H3 Character Replacement"

    def compare(self, source, generated):
        import torch
        if tuple(source.shape) != tuple(generated.shape):
            raise ValueError("Comparison requires exactly aligned dimensions and frame counts")
        return (torch.cat([source, generated.to(source.device)], dim=2),)


class H3SwapSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",), "filename_prefix": ("STRING", {"default": "h3_character_swap/bfs"}), "crf": ("INT", {"default": 16, "min": 0, "max": 40})}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "H3 Character Replacement"

    def save(self, video, filename_prefix, crf):
        import folder_paths
        from comfy_api.latest import Types, ui, io
        from pathlib import PurePosixPath
        import uuid
        if "\\" in filename_prefix or ".." in PurePosixPath(filename_prefix).parts or PurePosixPath(filename_prefix).is_absolute():
            raise ValueError("Export prefix must remain inside the output directory")
        width, height = video.get_dimensions()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), width, height)
        filename = f"{name}_{counter:05d}_{uuid.uuid4().hex[:8]}.mp4"
        video.save_to(str(Path(folder) / filename), format=Types.VideoContainer("mp4"), codec=Types.VideoCodec("h264"), crf=int(crf), metadata=None)
        return {"ui": ui.PreviewVideo([ui.SavedResult(filename, subfolder, io.FolderType.output)]).as_dict()}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3SwapClipWindow, H3SwapAnalyzeClip, H3SwapBFSPrompt, H3SwapComparison, H3SwapSaveVideo)}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SwapClipWindow": "H3 Swap - Select exact source window", "H3SwapAnalyzeClip": "H3 Swap - Local prompt draft (review before rendering)", "H3SwapBFSPrompt": "H3 Swap - BFS v1.1 trained prompt template", "H3SwapComparison": "H3 Swap - Source / generated comparison", "H3SwapSaveVideo": "H3 Swap - Save source-audio MP4"}
