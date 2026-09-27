#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
from huggingface_hub import hf_hub_download


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, item: dict):
    if not path.exists() or path.stat().st_size == 0:
        return False
    expected = item.get("sha256")
    if expected and sha256(path).lower() != expected.lower():
        return False
    return True


def with_civitai_token(url: str):
    token = os.getenv("CIVITAI_TOKEN", "")
    if not token:
        return url
    parsed = urlparse(url)
    q = dict(parse_qsl(parsed.query))
    q.setdefault("token", token)
    return urlunparse(parsed._replace(query=urlencode(q)))


def download_url(url: str, dest: Path, headers=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)

    cmd = [
        "aria2c",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--continue=true",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=8M",
        "--summary-interval=5",
        "--file-allocation=none",
        "--dir", str(dest.parent),
        "--out", tmp.name,
    ]
    for key, value in (headers or {}).items():
        cmd += ["--header", f"{key}: {value}"]
    cmd.append(url)
    subprocess.run(cmd, check=True)
    tmp.replace(dest)


def download_one(item: dict, comfy_home: Path):
    if not item.get("enabled", True):
        return f"SKIP disabled: {item.get('name', 'unnamed')}"

    name = item.get("name", "unnamed")
    destination = item["destination"]
    dest = comfy_home / destination
    required = item.get("required", True)

    try:
        if verify(dest, item):
            return f"OK cached: {name} -> {destination}"

        dest.unlink(missing_ok=True)
        source = item.get("source", "url")

        if source == "huggingface":
            repo_id = item["repo_id"]
            filename = item["filename"]
            revision = item.get("revision")
            token = os.getenv("HF_TOKEN") or None
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=str(dest.parent)) as td:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    token=token,
                    local_dir=td,
                )
                shutil.move(downloaded, dest)

        elif source == "civitai":
            if "url" in item:
                url = with_civitai_token(item["url"])
            else:
                url = with_civitai_token(
                    f"https://civitai.com/api/download/models/{item['model_version_id']}"
                )
            download_url(url, dest)

        elif source == "url":
            url = item["url"]
            headers = {}
            if item.get("use_hf_token") and os.getenv("HF_TOKEN"):
                headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
            download_url(url, dest, headers=headers)

        else:
            raise ValueError(f"Unsupported source: {source}")

        if not verify(dest, item):
            raise RuntimeError("downloaded file failed verification")

        return f"DONE: {name} -> {destination} ({dest.stat().st_size / 1024**3:.2f} GB)"
    except Exception as exc:
        if required:
            raise
        return f"WARN optional asset failed: {name}: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--comfy-home", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"No asset manifest: {manifest}")
        return

    items = json.loads(manifest.read_text())
    if not items:
        print(f"No enabled assets in {manifest.name}")
        return

    workers = max(1, int(os.getenv("ASSET_DOWNLOAD_WORKERS", "4")))
    comfy_home = Path(args.comfy_home)

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, item, comfy_home): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failures.append((item.get("name", "unnamed"), str(exc)))
                print(f"ERROR: {item.get('name', 'unnamed')}: {exc}", flush=True)

    if failures:
        raise SystemExit(f"Required asset downloads failed: {failures}")


if __name__ == "__main__":
    main()
