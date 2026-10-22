"""Download only build-locked public weights into the existing persistent model tree."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import time
from .config import ROOT, atomic_json, data_root, file_sha, safe_path


def prepare(selected, status=lambda **kw: None):
    from huggingface_hub import hf_hub_download
    from filelock import FileLock
    rows = json.loads((ROOT / "assets.lock.json").read_text())["assets"]
    rows = [r for r in rows if r["group"] in {"common"} | set(selected)]
    root = data_root()
    state = root / "h3-swap"
    state.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    if token and token.lstrip().startswith("{{"):
        raise ValueError("HF_TOKEN contains an unresolved RunPod secret")
    record_path = state / "verified-models.json"
    try:
        verified = json.loads(record_path.read_text())
    except (OSError, ValueError):
        verified = {}
    with FileLock(str(state / "model-download.lock"), timeout=0):
        for index, row in enumerate(rows, 1):
            target = safe_path(root / "models", row["destination"])
            target.parent.mkdir(parents=True, exist_ok=True)
            status(stage="models", asset=row["id"], asset_number=index, asset_count=len(rows))
            print(f"[H3 SWAP PREP] {index}/{len(rows)} {row['id']}", flush=True)
            saved = verified.get(row["id"], {})
            st = target.stat() if target.is_file() else None
            fingerprint = {"sha256": row["sha256"], "bytes": st.st_size, "mtime_ns": st.st_mtime_ns} if st else {}
            if st and saved == fingerprint and st.st_size == row["bytes"]:
                continue
            if st and st.st_size == row["bytes"] and file_sha(target) == row["sha256"]:
                verified[row["id"]] = fingerprint
                atomic_json(record_path, verified)
                continue
            if target.exists():
                raise ValueError("An existing model differs from the pinned file: " + row["destination"] + ". It was not overwritten; move it aside manually before retrying.")
            reserve = 12 * 1024**3
            if shutil.disk_usage(root).free < row["bytes"] + reserve:
                raise ValueError("Insufficient persistent storage for " + row["id"] + " plus 12 GiB working headroom")
            staging = state / "downloads" / row["id"]
            staging.mkdir(parents=True, exist_ok=True)
            try:
                downloaded = Path(hf_hub_download(repo_id=row["repo"], filename=row["file"], revision=row["revision"], local_dir=str(staging), token=token))
            except Exception as exc:
                raise ValueError("Download failed for " + row["id"] + " (" + type(exc).__name__ + "). Check HF access, network and free volume space.") from None
            if downloaded.stat().st_size != row["bytes"] or file_sha(downloaded) != row["sha256"]:
                raise ValueError("Model verification failed: " + row["id"])
            os.replace(downloaded, target)
            st = target.stat()
            verified[row["id"]] = {"sha256": row["sha256"], "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
            atomic_json(record_path, verified)
            print("[H3 SWAP PREP] verified " + row["id"], flush=True)
    return {"assets": len(rows), "model_bytes": sum(r["bytes"] for r in rows)}
