"""Small, local-only helpers for the quality-first workflows.

No remote service calls, face recognition, training, or code generation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid
import zipfile

WEB_DIRECTORY = "./web"
ARCHIVE_RE = re.compile(r"refmods-[0-9]{8}-[0-9]{6}-[0-9a-f]{12}\.zip\Z")


def reference_name(image, label: str) -> str:
    import torch
    if image is None or image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError("Connect exactly one reference image to each RefMod creator. The two-subject workflow needs BOTH images.")
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_")[:48]
    if not clean:
        raise ValueError("Use a short subject label containing letters or digits.")
    # Stable name identifies the supplied pixels, not a person's identity.
    rgb = image[0, ..., :3].detach().to(device="cpu", dtype=torch.float32)
    data = (rgb.clamp(0, 1).mul(255).round().to(torch.uint8)).contiguous().numpy().tobytes()
    h = hashlib.sha256(str(tuple(rgb.shape)).encode() + data).hexdigest()[:16]
    return f"{clean}_{h}"


class QualityRefModName:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "label": ("STRING", {"default": "subject"})}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "name"
    CATEGORY = "Quality/RefMod"
    DESCRIPTION = "Creates a stable filename from a label and image checksum. Does not identify a real person."
    def name(self, image, label):
        return (reference_name(image, label),)


class QualityCheckRefModBundle:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mods": ("H3_REF_MODS",)}}
    RETURN_TYPES = ("H3_REF_MODS",)
    FUNCTION = "check"
    CATEGORY = "Quality/RefMod"
    DESCRIPTION = "Stops a saved-reference render when no reference was selected."
    def check(self, mods):
        if not mods:
            raise ValueError("Select at least one saved RefMod in Load H3 RefMods before rendering.")
        return (mods,)


def make_archive(ref_root: Path, output_root: Path, max_bytes: int = 2 * 1024**3) -> tuple[str, int]:
    """Archive RefMod files only; reject symlinks and bound the total size."""
    root = ref_root.resolve()
    if not root.is_dir():
        raise ValueError("No RefMod directory exists yet. Run the creator first.")
    candidates = []
    total = 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(folder) / d).is_symlink() and d not in {".git", "__pycache__"})
        for name in sorted(files):
            path = Path(folder) / name
            if path.suffix not in {".safetensors", ".json"} or path.is_symlink():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError("RefMod path escapes the selected reference directory.")
            total += resolved.stat().st_size
            if total > max_bytes:
                raise ValueError("RefMod archive exceeds 2 GiB. Export a smaller library using the Pod file tools.")
            candidates.append((resolved, resolved.relative_to(root).as_posix()))
    if not candidates:
        raise ValueError("No saved RefMod files found. Enable save on the creator and run it first.")
    output_root.mkdir(parents=True, exist_ok=True)
    filename = "refmods-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12] + ".zip"
    destination = output_root / filename
    part = destination.with_suffix(".part")
    try:
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED) as z:
            manifest = []
            for path, relative in candidates:
                z.write(path, "refmods/" + relative)
                manifest.append({"path": relative, "bytes": path.stat().st_size})
            z.writestr("RESTORE_README.txt", "Place refmods/ contents in ComfyUI/models/refmods/. These are private reference data, not LoRAs. Do not publish without appropriate rights.\n")
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
        os.replace(part, destination)
    finally:
        part.unlink(missing_ok=True)
    return filename, len(candidates)


class QualityExportRefMods:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"after": ("H3_REF_MODS",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "Quality/RefMod"
    DESCRIPTION = "Backs up ALL saved RefMods in the primary RefMod folder to a downloadable ZIP. Temporary Pod files must be downloaded before termination."
    def export(self, after):
        import folder_paths
        roots = folder_paths.get_folder_paths("refmods")
        root = Path(roots[0]) if roots else Path(folder_paths.models_dir) / "refmods"
        directory = Path(folder_paths.get_output_directory()) / "refmod_exports"
        filename, count = make_archive(root, directory)
        relative_url = "/quality/refmod-exports/" + filename
        return {"ui": {"text": [f"{count} files. Download before terminating the Pod."], "refmod_download": [relative_url]},
                "result": (relative_url,)}


def register_download_route():
    try:
        import folder_paths
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return
    instance = getattr(PromptServer, "instance", None)
    if instance is None or getattr(instance, "_quality_refmod_export_route", False):
        return
    @instance.routes.get("/quality/refmod-exports/{filename}")
    async def download(request):
        filename = request.match_info["filename"]
        if not ARCHIVE_RE.fullmatch(filename):
            raise web.HTTPNotFound()
        root = (Path(folder_paths.get_output_directory()) / "refmod_exports").resolve()
        path = root / filename
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{filename}"',
                                               "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    instance._quality_refmod_export_route = True


NODE_CLASS_MAPPINGS = {"QualityRefModName": QualityRefModName,
                       "QualityCheckRefModBundle": QualityCheckRefModBundle,
                       "QualityExportRefMods": QualityExportRefMods}
NODE_DISPLAY_NAME_MAPPINGS = {"QualityRefModName": "Reference Filename (Image Checksum)",
                              "QualityCheckRefModBundle": "Check Saved RefMods Selected",
                              "QualityExportRefMods": "Download RefMod Library ZIP"}
register_download_route()
