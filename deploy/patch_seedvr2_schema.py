#!/usr/bin/env python3
"""Make pinned SeedVR2 node schema registration safe on GPU-less Docker builders.

SeedVR2's DiT and VAE loader schemas currently call get_device_list() and then
use devices[0]. During optional ComfyUI --cpu diagnostics there is no CUDA
or MPS device, so that list is empty and schema registration raises IndexError.

This patch only adds a CPU fallback when the list is empty. On a RunPod GPU the
upstream list is non-empty, so the runtime default remains cuda:0.
"""
from __future__ import annotations

from pathlib import Path
import argparse

FILES = (
    "src/interfaces/dit_model_loader.py",
    "src/interfaces/vae_model_loader.py",
)
OLD = "        devices = get_device_list()\n"
NEW = "        devices = get_device_list()\n        if not devices:\n            devices = [\"cpu\"]  # schema-only fallback for GPU-less build hosts\n"


def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        return
    count = text.count(OLD)
    if count != 1:
        raise RuntimeError(f"Expected exactly one SeedVR2 device-list assignment in {path}, found {count}")
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seedvr2-dir",
        type=Path,
        default=Path("/opt/ComfyUI/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler"),
    )
    args = parser.parse_args()
    for rel in FILES:
        path = args.seedvr2_dir / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        patch_file(path)
        print(f"SeedVR2 CPU-schema fallback verified: {path}")


if __name__ == "__main__":
    main()
