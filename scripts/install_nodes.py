#!/usr/bin/env python3
"""Install only explicitly enabled, commit-pinned HTTPS custom-node repositories."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


def validate(row: dict) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row.get("name", "")):
        raise ValueError("Invalid custom-node directory name")
    parsed = urlsplit(row.get("repo", ""))
    if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Custom-node repos must be public GitHub HTTPS URLs without credentials")
    if not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", parsed.path):
        raise ValueError("Invalid custom-node repository path")
    if not re.fullmatch(r"[a-f0-9]{40}", row.get("ref", "")):
        raise ValueError("Pin custom nodes to a complete Git commit SHA, not main/master")


def install(manifest: Path, comfy_home: Path, constraints: Path) -> None:
    rows = json.loads(manifest.read_text())
    if not isinstance(rows, list):
        raise ValueError("custom_nodes.json must be a list")
    names = set()
    for row in rows:
        if not row.get("enabled", True):
            continue
        validate(row)
        if row["name"] in names:
            raise ValueError("Duplicate custom-node directory")
        names.add(row["name"])
        target = comfy_home / "custom_nodes" / row["name"]
        if not target.resolve().is_relative_to((comfy_home / "custom_nodes").resolve()):
            raise ValueError("Custom-node directory escapes custom_nodes")
        target.mkdir(parents=True, exist_ok=True)
        def git(*args):
            return subprocess.run(["git", "-C", str(target), *args], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
        if not (target / ".git").exists():
            git("init", "-q")
            git("remote", "add", "origin", row["repo"])
        else:
            git("remote", "set-url", "origin", row["repo"])
        git("fetch", "--depth", "1", "origin", row["ref"])
        git("checkout", "--force", "--detach", "FETCH_HEAD")
        if git("rev-parse", "HEAD") != row["ref"]:
            raise RuntimeError("Custom-node revision verification failed")
        print(f"PINNED NODE: {row['name']} @ {row['ref']}", flush=True)
        req = target / "requirements.txt"
        if row.get("install_requirements", True) and req.is_file():
            subprocess.run([os.sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-c", str(constraints), "-r", str(req)], check=True)
        # Arbitrary install.py/install.sh hooks are deliberately NOT auto-executed.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--comfy-home", type=Path, required=True)
    parser.add_argument("--constraints", type=Path, required=True)
    args = parser.parse_args()
    install(args.manifest, args.comfy_home, args.constraints)
