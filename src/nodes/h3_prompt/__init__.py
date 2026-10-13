"""ComfyUI nodes for local vision-assisted H3 prompting. No server-side routes."""
from __future__ import annotations
import json
from pathlib import Path
import threading
import time
import uuid

from .client import SYSTEM_PROMPT, settings, generate, validate_tags

LOCK = threading.Lock()
MODES = ["Generate with Ollama", "Use edited prompt (no Ollama)"]


class EverydayOllamaH3Prompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_1": ("IMAGE",),
            "mode": (MODES, {"default": MODES[0]}),
            "system_prompt": ("STRING", {"default": SYSTEM_PROMPT, "multiline": True}),
            "user_prompt": ("STRING", {"default": "I want the person from image 1 to hug the person from image 2. Preserve both subjects' appearance and clothing. One continuous shot, no dialogue, no music.", "multiline": True}),
            "edited_prompt": ("STRING", {"default": "", "multiline": True}),
            "length": ("INT", {"default": 124, "min": 124, "max": 362, "step": 17}),
            "variation_seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
            "save_prompt": ("BOOLEAN", {"default": True}),
        }, "optional": {"image_2": ("IMAGE",)}}

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("h3_prompt", "picture_1", "picture_2", "length")
    FUNCTION = "build_prompt"
    CATEGORY = "Everyday/H3 Prompt Assistant"
    DESCRIPTION = "Writes an H3 prompt from one or two still references, then unloads Ollama before returning. Image outputs preserve the original reference order. Use edited mode to render an already-reviewed prompt without another LLM call."

    def build_prompt(self, image_1, mode, system_prompt, user_prompt, edited_prompt,
                     length, variation_seed, temperature, save_prompt, image_2=None):
        if mode not in MODES:
            raise ValueError("Unknown prompt mode")
        if length < 124 or length > 362 or length % 17 != 5:
            raise ValueError("Use an H3 length on the 17k+5 grid from 124 to 362 frames")
        images = [image_1] + ([image_2] if image_2 is not None else [])
        for image in images:
            if image.ndim != 4 or image.shape[0] != 1:
                raise ValueError("Connect one still image per reference input")
        if mode == MODES[1]:
            prompt = edited_prompt.strip()
            if not prompt:
                raise ValueError("Paste the reviewed H3 text into edited_prompt, or choose Generate with Ollama")
            validate_tags(prompt, len(images))
            metadata = {"mode": "edited", "frame_count": length, "references": len(images)}
        else:
            import comfy.model_management as mm
            with LOCK:
                mm.throw_exception_if_processing_interrupted()
                # Clear previously cached H3/Krea GPU weights before a new vision request.
                mm.unload_all_models()
                mm.soft_empty_cache()
                prompt, metadata = generate(settings(), system_prompt, user_prompt, images, length,
                                            variation_seed, temperature, mm.throw_exception_if_processing_interrupted)
            print("[OllamaH3] Prompt ready; selected vision model no longer listed in Ollama /api/ps.", flush=True)
        if save_prompt:
            import folder_paths
            root = Path(folder_paths.get_output_directory()) / "prompt_assistant"
            root.mkdir(parents=True, exist_ok=True)
            stem = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            (root / (stem + ".txt")).write_text(prompt + "\n", encoding="utf-8")
            (root / (stem + ".json")).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"[OllamaH3] Saved prompt_assistant/{stem}.txt", flush=True)
        return prompt, image_1, image_2, length


class EverydayH3FilesAfterPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        # Use strings rather than dynamic dropdowns so a CPU build smoke-test needs no weights.
        return {"required": {
            "prompt_ready": ("STRING", {"forceInput": True}),
            "diffusion_model": ("STRING", {"default": "minimax_h3_ref2va_bf16.safetensors"}),
            "text_encoder": ("STRING", {"default": "qwen3vl_32b_minimax_h3_bf16.safetensors"}),
            "video_vae": ("STRING", {"default": "minimax_h3_video_vae_fp16.safetensors"}),
            "audio_vae": ("STRING", {"default": "minimax_h3_audio_vae_fp32.safetensors"}),
        }}

    RETURN_TYPES = ("COMBO", "COMBO", "COMBO", "COMBO")
    RETURN_NAMES = ("diffusion_model", "text_encoder", "video_vae", "audio_vae")
    FUNCTION = "after_prompt"
    CATEGORY = "Everyday/H3 Prompt Assistant"
    DESCRIPTION = "Execution dependency only: filenames reach native H3 loaders after prompting/unloading completes. This does not load or modify any model. Keep these connections in the integrated workflow."

    def after_prompt(self, prompt_ready, diffusion_model, text_encoder, video_vae, audio_vae):
        if not prompt_ready.strip():
            raise ValueError("An H3 prompt is required before the model loaders run")
        return diffusion_model, text_encoder, video_vae, audio_vae


NODE_CLASS_MAPPINGS = {"EverydayOllamaH3Prompt": EverydayOllamaH3Prompt,
                       "EverydayH3FilesAfterPrompt": EverydayH3FilesAfterPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"EverydayOllamaH3Prompt": "H3 Prompt Builder (Local Ollama)",
                              "EverydayH3FilesAfterPrompt": "H3 Models AFTER Prompt + Unload"}
