#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def run(cmd, cwd=None, env=None):
    printable_parts = []
    token = os.getenv("GITHUB_TOKEN", "")
    for value in cmd:
        text = str(value)
        if token:
            text = text.replace(token, "***")
        printable_parts.append(shlex.quote(text))
    print(f"+ {' '.join(printable_parts)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def git_cmd(repo):
    cmd = ["git"]
    token = os.getenv("GITHUB_TOKEN", "")
    if token and urlparse(repo).hostname == "github.com":
        cmd += ["-c", f"http.extraHeader=Authorization: Bearer {token}"]
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--comfy-home", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"No custom node manifest: {manifest_path}")
        return

    nodes = json.loads(manifest_path.read_text())
    root = Path(args.comfy_home) / "custom_nodes"
    root.mkdir(parents=True, exist_ok=True)

    for node in nodes:
        if not node.get("enabled", True):
            continue

        repo = node["repo"]
        name = node.get("name") or Path(urlparse(repo).path).stem
        dest = root / name
        ref = node.get("ref")

        print(f"\n== Custom node: {name} ==")
        if dest.exists():
            run(["rm", "-rf", str(dest)])

        cmd = git_cmd(repo) + ["clone", "--filter=blob:none", repo, str(dest)]
        run(cmd)

        if ref:
            run(git_cmd(repo) + ["-C", str(dest), "fetch", "--depth", "1", "origin", ref])
            run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"])

        requirements = dest / "requirements.txt"
        if requirements.exists():
            run(["python", "-m", "pip", "install", "-r", str(requirements)])

        if node.get("run_install_py", False) and (dest / "install.py").exists():
            run(["python", "install.py"], cwd=dest)

        for command in node.get("post_install", []):
            print(f"+ [post_install] {command}")
            subprocess.run(command, cwd=dest, shell=True, check=True)


if __name__ == "__main__":
    main()
