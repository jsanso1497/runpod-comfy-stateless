#!/usr/bin/env python3
"""Validated, profile-aware model and LoRA downloads. No model weights in GitHub."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

SHA = re.compile(r"[0-9a-fA-F]{64}\Z")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def selected_profiles(config: Path, override: str | None = None) -> set[str]:
    runtime = read_json(config / "runtime.json")
    value = override if override is not None else os.environ.get("MODEL_PROFILES", "")
    names = {s.strip() for s in value.split(",") if s.strip()} if value else set(runtime["default_profiles"])
    unknown = names - set(runtime["profiles"])
    if unknown or not names:
        raise ValueError("MODEL_PROFILES must contain known names: " + ", ".join(runtime["profiles"]))
    return names


def safe_destination(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("Invalid model destination")
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or len(part.parts) < 3 or part.parts[0] != "models":
        raise ValueError("Model destination must be beneath models/, with no traversal")
    root = root.resolve()
    target = (root / relative).resolve()
    allowed = (root / "models").resolve()
    if not allowed.is_relative_to(root):
        raise ValueError("The models directory must not point outside the ComfyUI root")
    if not target.is_relative_to(allowed) or target == allowed:
        raise ValueError("Model destination escapes the models directory")
    if target.suffix not in {".safetensors", ".gguf", ".pt", ".pth", ".bin"}:
        raise ValueError("Unexpected model file extension")
    return target


def asset_url(item: dict) -> str:
    source = item.get("source", "huggingface")
    if source == "huggingface":
        repo = item.get("repo_id", "")
        filename = item.get("filename", "")
        revision = item.get("revision", "main")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("Invalid Hugging Face repo_id")
        if not filename or filename.startswith("/") or ".." in PurePosixPath(filename).parts:
            raise ValueError("Invalid Hugging Face filename")
        return f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{quote(filename, safe='/')}"
    if source == "url":
        url = item.get("url", "")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("Direct model URLs must use HTTPS without embedded credentials")
        if any(x in parsed.query.lower() for x in ("token=", "api_key=", "apikey=", "authorization=")):
            raise ValueError("Keep credentials out of model URLs; use a RunPod Secret")
        return url
    raise ValueError("Supported sources are huggingface and url")


def catalog(config: Path, root: Path, profiles: set[str]) -> list[dict]:
    items, destinations = [], {}
    for filename in ("models.json", "loras.json"):
        path = config / filename
        if not path.exists():
            if filename == "loras.json":
                continue
            raise FileNotFoundError(path)
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{filename} must contain a JSON list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid entry in {filename}")
            if not row.get("enabled", True) or row.get("profile", "shared") not in profiles | {"shared"}:
                continue
            if not SHA.fullmatch(str(row.get("sha256", ""))):
                raise ValueError(f"A complete SHA-256 is required for {row.get('name', 'unnamed asset')}")
            target = safe_destination(root, row.get("destination", ""))
            if filename == "loras.json" and not target.is_relative_to((root / "models/loras").resolve()):
                raise ValueError("LoRA destinations must be inside models/loras/")
            asset_url(row)
            key = str(target)
            if key in destinations:
                if destinations[key] != row["sha256"].lower():
                    raise ValueError("Two different models use the same destination")
                continue
            destinations[key] = row["sha256"].lower()
            items.append(row)
    return items


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def headers_for(item: dict, url: str) -> dict[str, str]:
    headers = {"User-Agent": "ComfyUI-Everyday/2.0", "Accept-Encoding": "identity"}
    host = urlsplit(url).hostname
    # Requests strips Authorization when redirecting to a different host.
    env_name = "HF_TOKEN" if host == "huggingface.co" else "CIVITAI_TOKEN" if host in {"civitai.com", "www.civitai.com"} else None
    if env_name and os.environ.get(env_name):
        headers["Authorization"] = "Bearer " + os.environ[env_name]
    return headers


def download_one(item: dict, root: Path) -> dict:
    import requests
    dest = safe_destination(root, item["destination"])
    expected = item["sha256"].lower()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and digest(dest) == expected:
        print(f"VERIFIED: {item['name']} (already present)", flush=True)
        return {"name": item["name"], "destination": item["destination"], "sha256": expected, "bytes": dest.stat().st_size}
    part = dest.with_name(dest.name + ".part")
    if part.is_symlink():
        raise ValueError("Refusing a symlinked partial download")
    url = asset_url(item)
    last_error = "download incomplete"
    for attempt in range(1, 4):
        offset = part.stat().st_size if part.exists() else 0
        if offset and digest(part) == expected:
            os.replace(part, dest)
            return {"name": item["name"], "destination": item["destination"], "sha256": expected, "bytes": dest.stat().st_size}
        headers = headers_for(item, url)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as response:
                if response.status_code == 416:
                    part.unlink(missing_ok=True)
                    raise RuntimeError("Remote file range changed; restarting download")
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {response.status_code}; check access, license acceptance, or source path")
                if response.status_code == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Unexpected server byte range")
                    mode = "ab"
                else:
                    mode, offset = "wb", 0
                total = offset + int(response.headers.get("Content-Length", "0"))
                count, last_log = offset, 0.0
                with part.open(mode) as handle:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        count += len(chunk)
                        if time.monotonic() - last_log > 10:
                            suffix = f" / {total / 2**30:.1f} GiB" if total else " GiB"
                            print(f"DOWNLOAD: {item['name']}: {count / 2**30:.1f}{suffix}", flush=True)
                            last_log = time.monotonic()
            print(f"HASH CHECK: {item['name']}", flush=True)
            if digest(part) != expected:
                part.unlink(missing_ok=True)
                raise RuntimeError("SHA-256 mismatch; unverified bytes discarded")
            os.replace(part, dest)
            print(f"READY: {item['name']}", flush=True)
            return {"name": item["name"], "destination": item["destination"], "sha256": expected, "bytes": dest.stat().st_size}
        except (requests.RequestException, OSError, RuntimeError) as exc:
            # Never log signed redirect URLs or authentication credentials.
            last_error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
            print(f"RETRY {attempt}/3: {item['name']}: {last_error}", flush=True)
            if attempt < 3:
                time.sleep(3 * attempt)
    raise RuntimeError(f"Could not prepare {item['name']}: {last_error}")


def run_downloads(config: Path, root: Path, profiles: set[str]) -> None:
    items = catalog(config, root, profiles)
    root.mkdir(parents=True, exist_ok=True)
    missing_estimate = 0
    for item in items:
        target = safe_destination(root, item["destination"])
        # Count a full replacement unless an existing destination verifies.
        if not target.exists() or digest(target) != item["sha256"].lower():
            missing_estimate += int(item.get("estimated_bytes", 0))
    free = shutil.disk_usage(root).free
    if free < missing_estimate + 8 * 2**30:
        raise RuntimeError(f"Insufficient free disk for selected profiles. Need about {(missing_estimate + 8*2**30)/2**30:.0f} GiB free, have {free/2**30:.0f}. Increase container disk or remove h3 from MODEL_PROFILES.")
    workers = max(1, min(4, int(os.environ.get("ASSET_DOWNLOAD_WORKERS", "2"))))
    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, item, root): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"ASSET ERROR: {item['name']}: {exc}", file=sys.stderr, flush=True)
                if item.get("required", True):
                    failures.append(item["name"])
    report = {"profiles": sorted(profiles), "assets": sorted(results, key=lambda x: x["destination"]), "required_failures": failures}
    (root / "asset-download-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise RuntimeError("Required downloads failed: " + ", ".join(failures))


def workflow_sources(config: Path) -> list[Path]:
    """Exclude only explicitly retired recipe names, including leftovers after a ZIP merge."""
    runtime = read_json(config / "runtime.json")
    retired = set(runtime.get("retired_workflows", []))
    return [p for p in sorted((config / "workflows").glob("*.json")) if p.name not in retired]


def install_workflows(config: Path, root: Path, profiles: set[str]) -> list[Path]:
    runtime = read_json(config / "runtime.json")
    destination = root / "user/default/workflows"
    destination.mkdir(parents=True, exist_ok=True)
    installed = []
    for source in workflow_sources(config):
        profile = runtime.get("workflow_profiles", {}).get(source.name, "shared")
        if profile not in profiles | {"shared"}:
            continue
        target = destination / source.name
        # Copy recipe workflows on a new Pod. Preserve manually edited copies on a process restart.
        if not target.exists():
            shutil.copy2(source, target)
        installed.append(target)
    (root / "active-recipe-workflows.json").write_text(json.dumps([str(s) for s in installed], indent=2) + "\n")
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "download", "workflows", "profiles", "comfy-ref"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--comfy-home", type=Path, default=Path("/workspace/ComfyUI"))
    args = parser.parse_args()
    if args.command == "comfy-ref":
        value = read_json(args.config / "runtime.json")["comfy_ref"]
        if not re.fullmatch(r"[a-f0-9]{40}", value):
            raise ValueError("runtime.json comfy_ref must be a complete Git commit SHA")
        print(value)
        return
    profiles = selected_profiles(args.config)
    if args.command == "profiles":
        print(",".join(sorted(profiles)))
    elif args.command == "validate":
        rows = catalog(args.config, args.comfy_home, set(read_json(args.config / "runtime.json")["profiles"]))
        print(f"Catalog validation PASS: {len(rows)} enabled entries across all profiles")
    elif args.command == "download":
        run_downloads(args.config, args.comfy_home, profiles)
    else:
        paths = install_workflows(args.config, args.comfy_home, profiles)
        print("Installed recipe workflows: " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"CATALOG ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
