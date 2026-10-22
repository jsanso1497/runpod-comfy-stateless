"""Remove an unused base-image developer tool during Docker assembly only.

The application venv deliberately inherits the pinned PyTorch/CUDA installation.
It also inherits spin 0.17 unless the image is cleaned first. That developer tool
requires Click <8.4 while the current Hugging Face runtime needs Click >=8.4.2.
Do not downgrade runtime libraries or suppress pip check to hide this conflict.

This is NOT a host installer. The only protected-system pip operation permitted
here is uninstalling spin in the disposable image, after reverse-dependency and
pinned-version checks. All application installs still run in the app venv.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys

TARGET = "spin"
EXPECTED_VERSION = "0.17"
PROTECTED_PACKAGES = ("torch", "torchvision", "torchaudio")


def active_dependents(distributions=None) -> list[str]:
    """Return installed packages with an active runtime dependency on spin."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    rows = md.distributions() if distributions is None else distributions
    found = set()
    for dist in rows:
        name = dist.metadata.get("Name", "")
        if not name or canonicalize_name(name) == TARGET:
            continue
        for raw in dist.requires or ():
            requirement = Requirement(raw)
            if canonicalize_name(requirement.name) != TARGET:
                continue
            # Dev/test extras are not runtime dependencies of the base image.
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                found.add(name)
    return sorted(found)


def protected_snapshot() -> dict[str, dict[str, str]]:
    result = {}
    for name in PROTECTED_PACKAGES:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        result[name] = {"version": dist.version, "location": str(dist.locate_file(""))}
    return result


def remove_unused_spin() -> dict:
    """Narrow cleanup operation; main additionally restricts it to image builds."""
    before = protected_snapshot()
    try:
        dist = md.distribution(TARGET)
    except md.PackageNotFoundError:
        print("[H3 BASE CLEANUP] spin is absent; no removal needed", flush=True)
        return {"removed": [], "protected_packages": before}
    if dist.version != EXPECTED_VERSION:
        raise RuntimeError("Unexpected base-image spin version: " + dist.version +
                           ". Review the base-image pin before removing it.")
    dependents = active_dependents()
    if dependents:
        raise RuntimeError("Refusing to remove spin: active runtime dependents: " +
                           ", ".join(dependents))
    print("[H3 BASE CLEANUP] Removing unused spin " + dist.version +
          "; retaining PyTorch/CUDA and all runtime dependency checks", flush=True)
    # Scope the PEP 668 exception to this one uninstall, never pip install.
    # No global PIP_BREAK_SYSTEM_PACKAGES setting or marker deletion is used.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes",
                    "--break-system-packages", TARGET], check=True, timeout=120)
    importlib.invalidate_caches()
    try:
        md.distribution(TARGET)
    except md.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("spin is still visible after removal; refusing to build")
    after = protected_snapshot()
    if before != after:
        raise RuntimeError("Base cleanup changed the protected PyTorch installation")
    return {"removed": [{"name": TARGET, "version": EXPECTED_VERSION}],
            "protected_packages": after}


def require_image_build() -> None:
    if (os.environ.get("H3_SWAP_BUILD_CLEANUP") != "1"
            or os.environ.get("H3_SWAP_ROOT") != "/opt/h3-swap"
            or sys.platform != "linux"
            or sys.prefix != sys.base_prefix
            or getattr(os, "geteuid", lambda: -1)() != 0):
        raise RuntimeError("Base cleanup is only permitted by the Dockerfile during image assembly")
    if md.version("torch").split("+")[0] != "2.11.0":
        raise RuntimeError("Base cleanup requires the pinned PyTorch 2.11.0 image")


def main() -> None:
    require_image_build()
    record = remove_unused_spin()
    Path("/opt/h3-base-cleanup.json").write_text(json.dumps(record, indent=2) + "\n")
    print("[H3 BASE CLEANUP] Complete", flush=True)


if __name__ == "__main__":
    main()
