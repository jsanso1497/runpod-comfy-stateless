"""Build pinned software and resolve model metadata, never download model weights."""
from __future__ import annotations
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .config import ROOT, atomic_json


def run(*args):
    subprocess.run(args, check=True)


def checkout(spec: dict, dest: Path):
    if not re.fullmatch(r"[a-f0-9]{40}", spec["commit"]):
        raise ValueError("Source commits must be immutable full hashes")
    dest.mkdir(parents=True, exist_ok=True)
    run("git", "init", "-q", str(dest))
    run("git", "-C", str(dest), "remote", "add", "origin", spec["url"])
    run("git", "-C", str(dest), "fetch", "--depth", "1", "origin", spec["commit"])
    run("git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if actual != spec["commit"]:
        raise ValueError("Source checkout mismatch")
    # Keep source and license files in the image; only Git metadata is removed.
    shutil.rmtree(dest / ".git")


def metadata_lock(specs, api):
    cache, result = {}, []
    for row in specs:
        revision = row["revision"]
        if re.fullmatch(r"[a-f0-9]{7,39}", revision):
            # Resolve the inspected short source revision before any file lookup.
            commits = api.list_repo_commits(row["repo"], token=False)
            matches = [c.commit_id for c in commits if c.commit_id.startswith(revision)]
            if len(matches) != 1:
                raise ValueError("Inspected model revision cannot be resolved uniquely: " + row["id"])
            revision = matches[0]
        key = (row["repo"], revision)
        if key not in cache:
            cache[key] = api.model_info(row["repo"], revision=revision, files_metadata=True, token=False)
        info = cache[key]
        if not re.fullmatch(r"[a-f0-9]{40}", info.sha):
            raise ValueError("Provider did not supply an immutable model revision")
        matches = [f for f in info.siblings if f.rfilename == row["file"]]
        if len(matches) != 1 or not matches[0].lfs:
            raise ValueError("Missing LFS asset: " + row["id"])
        f = matches[0]
        lfs = f.lfs
        sha = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
        size = lfs.get("size") if isinstance(lfs, dict) else lfs.size
        if not re.fullmatch(r"[a-f0-9]{64}", sha or "") or int(size) <= 0:
            raise ValueError("Incomplete provider hash/size")
        if row.get("expected_sha256") and sha != row["expected_sha256"]:
            raise ValueError("Published release hash changed: " + row["id"])
        result.append({**row, "revision": info.sha, "sha256": sha, "bytes": int(size)})
    return result


def extract_bfs_template(text: str) -> str:
    candidates = re.findall(r"`(body_swap: subject_definitions:.*?)`", text, flags=re.S)
    if not candidates:
        # Also accept a Markdown fenced code block without changing its content.
        candidates = re.findall(r"```[^\n]*\n(body_swap:.*?)```", text, flags=re.S)
    candidates = [x for x in candidates if x.count("{details}") == 1 and "<Picture 1>" in x]
    if not candidates:
        raise ValueError("BFS documented v1.1 prompt contract changed")
    template = candidates[0].replace("\r\n", "\n").replace("\n", " ").strip()
    if not template.startswith("body_swap: subject_definitions:") or len(template) < 500:
        raise ValueError("Incomplete BFS template")
    return template


def download_public_text(repo: str, revision: str, filename: str) -> str:
    import requests
    from urllib.parse import quote
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(filename, safe='/')}"
    with requests.Session() as s:
        s.trust_env = False
        r = s.get(url, timeout=(15, 120))
        r.raise_for_status()
        if len(r.content) > 2_000_000:
            raise ValueError("Unexpectedly large source document")
        return r.content.decode("utf-8")


def require_virtualenv() -> dict[str, str]:
    """Fail before any checkout or pip write when the image selects system Python."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "H3 build selected system Python instead of the application virtual "
            "environment. Create /opt/h3-swap-venv with --system-site-packages "
            "and put its bin directory first on PATH."
        )
    environment = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    }
    print("[H3 BUILD PYTHON] " + json.dumps(environment, sort_keys=True), flush=True)
    return environment


def main():
    python_environment = require_virtualenv()
    pins = json.loads((ROOT / "pins.json").read_text())
    if sys.version_info[:2] != (3, 12):
        raise ValueError("This isolated image must use the qualified Python 3.12 runtime")
    if md.version("torch").split("+")[0] != pins["torch"]:
        raise ValueError("Base-image PyTorch version does not match the qualified runtime")
    comfy = Path("/opt/ComfyUI")
    checkout(pins["comfy"], comfy)
    constraints = Path("/tmp/swap-constraints.txt")
    constraints.write_text("\n".join(f"{n}=={md.version(n)}" for n in ("torch", "torchvision", "torchaudio") if _installed(n)) + "\n")
    run(sys.executable, "-m", "pip", "install", "--constraint", str(constraints), "-r", str(comfy / "requirements.txt"))
    run(sys.executable, "-m", "pip", "install", "--constraint", str(constraints), "-r", str(ROOT / "requirements.txt"))
    if md.version("comfyui-frontend-package") != pins["frontend"] or md.version("comfy-kitchen") != pins["kitchen"]:
        raise ValueError("Core dependency pin mismatch")
    relay = comfy / "custom_nodes/h3-relay"
    checkout(pins["relay"], relay)
    (comfy / "custom_nodes/H3SwapTools").symlink_to(ROOT / "custom_nodes/H3SwapTools", target_is_directory=True)
    from huggingface_hub import HfApi
    specs = json.loads((ROOT / "assets.spec.json").read_text())["assets"]
    locked = metadata_lock(specs, HfApi())
    atomic_json(ROOT / "assets.lock.json", {"assets": locked, "scope": "metadata only; weights not downloaded"})
    bfs_revision = next(r["revision"] for r in locked if r["id"] == "bfs-v11")
    docs = download_public_text("Alissonerdx/Minimax-H3-ComfyUI", bfs_revision, "docs/character-swap.md")
    (ROOT / "bfs-template.txt").write_text(extract_bfs_template(docs) + "\n")
    licenses = ROOT / "licenses"
    licenses.mkdir(exist_ok=True)
    (licenses / "BFS_CHARACTER_SWAP.md").write_text(download_public_text("Alissonerdx/Minimax-H3-ComfyUI", bfs_revision, "LICENSE_CHARACTER_SWAP.md"))
    (licenses / "BFS_CHARACTER_SWAP_GUIDE.md").write_text(docs)
    # Preserve upstream source, licenses and exact provenance for the supplied recipe.
    for filename in ("LICENSE", "LICENSE-Apache-2.0", "NOTICE.md"):
        shutil.copy2(relay / filename, licenses / ("RELAY_" + filename))
    from .graphs import build_all
    build_all(relay)
    run(sys.executable, "-m", "pip", "check")
    record = {"pins": pins, "asset_bytes_by_group": {g: sum(a["bytes"] for a in locked if a["group"] == g) for g in ("common", "akatz", "bfs")}, "python": sys.version, "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(), "gpu_tested": False}
    record["python_environment"] = python_environment
    atomic_json(ROOT / "build-provenance.json", record)


def _installed(name):
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
