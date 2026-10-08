#!/usr/bin/env python3
"""Conservatively register an already-installed, verified unified Qwen release.

Only paths in PACKAGE_MANIFEST.json plus the explicit repair helper are added.
Unrelated hashes are never changed. The existing snapshot verifier must pass.
By default this is a dry run; --write is the only mutation path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
REPAIR_FILES = {
    ".github/workflows/repair-qwen21-unified-snapshot.yml",
    "qwen21_unified/scripts/repair_source_snapshot.py",
    "qwen21_unified/tests/test_repair_snapshot.py",
}
GENERATED_FILE = "qwen21_photo_edit/Dockerfile.unified"
PREFERRED_MAP_KEYS = ("files", "entries", "inventory", "source_files", "file_hashes", "sha256")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel:
        raise RuntimeError(f"Unsafe release path: {rel!r}")
    path = Path(rel)
    if any(part in ("", ".", "..") for part in path.parts):
        raise RuntimeError(f"Unsafe release path: {rel!r}")
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"Path escapes repository: {rel}")
    return target


def load_release_files(root: Path) -> dict[str, str]:
    package = root / "PACKAGE_MANIFEST.json"
    if not package.is_file():
        raise RuntimeError("PACKAGE_MANIFEST.json missing. Upload the COMPLETE unified release, not just selected folders.")
    obj = json.loads(package.read_text(encoding="utf-8"))
    rows = obj.get("files")
    if not isinstance(rows, dict) or len(rows) < 100:
        raise RuntimeError("Unrecognized or incomplete unified PACKAGE_MANIFEST.json")
    expected = {}
    for rel, record in rows.items():
        path = checked_path(root, rel)
        if not path.is_file():
            raise RuntimeError(f"Unified release is incomplete: missing {rel}")
        declared = record if isinstance(record, str) else record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(declared, str) or not HEX64.fullmatch(declared):
            raise RuntimeError(f"Malformed packaged hash: {rel}")
        actual = digest(path)
        if actual.lower() != declared.lower():
            raise RuntimeError(f"Unified file differs from reviewed package: {rel}; expected {declared}, actual {actual}. Do not blindly rehash modified code.")
        if isinstance(record, dict) and "bytes" in record and path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Unified file byte length differs: {rel}")
        expected[rel] = actual
    if not (root / GENERATED_FILE).is_file():
        raise RuntimeError(f"Missing {GENERATED_FILE}. Reapply the unified overlay correctly.")
    scripts = root / "qwen21_unified/scripts"
    sys.path.insert(0, str(scripts))
    try:
        from generate_dockerfile import generate
        generated = generate((root / "qwen21_photo_edit/Dockerfile").read_text(encoding="utf-8"))
    finally:
        sys.path.pop(0)
    if (root / GENERATED_FILE).read_text(encoding="utf-8") != generated:
        raise RuntimeError("Generated unified Dockerfile differs from the existing Photo source; refusing to snapshot an unreviewed image.")
    expected[GENERATED_FILE] = digest(root / GENERATED_FILE)
    for rel in REPAIR_FILES:
        path = checked_path(root, rel)
        if not path.is_file():
            raise RuntimeError(f"Repair action was uploaded incompletely: missing {rel}")
        expected[rel] = digest(path)
    # The package manifest is a generated inventory and cannot hash itself.
    expected["PACKAGE_MANIFEST.json"] = digest(package)
    return expected


def extract_map(snapshot: dict):
    if not isinstance(snapshot, dict):
        raise RuntimeError("SOURCE_SNAPSHOT.json is not a JSON object")
    for key in PREFERRED_MAP_KEYS:
        value = snapshot.get(key)
        if isinstance(value, dict) and len(value) >= 10:
            if all(
                (isinstance(v, str) and bool(HEX64.fullmatch(v))) or
                (isinstance(v, dict) and isinstance(v.get("sha256"), str) and bool(HEX64.fullmatch(v["sha256"])))
                for v in value.values()
            ):
                return key, "dict", value
        if isinstance(value, list) and len(value) >= 10 and all(
            isinstance(row, dict) and isinstance(row.get("path"), str)
            and isinstance(row.get("sha256"), str) and bool(HEX64.fullmatch(row["sha256"]))
            for row in value
        ):
            paths = [row["path"] for row in value]
            if len(paths) != len(set(paths)):
                raise RuntimeError("Duplicate paths in SOURCE_SNAPSHOT.json; refusing repair")
            return key, "list", value
    raise RuntimeError("Unsupported SOURCE_SNAPSHOT.json schema. Existing verifier and manifest should be inspected manually.")


def reconcile(root: Path, write: bool = False, run_verifier: bool = True):
    root = root.resolve()
    manifest = root / "SOURCE_SNAPSHOT.json"
    if not manifest.is_file():
        raise RuntimeError("SOURCE_SNAPSHOT.json missing from repository")
    verifier = root / "tools/verify_snapshot.py"
    if not verifier.is_file():
        raise RuntimeError("tools/verify_snapshot.py missing; no safe validation of a revised manifest is possible")
    expected = load_release_files(root)
    original = manifest.read_bytes()
    obj = json.loads(original)
    key, mode, mapping = extract_map(obj)
    added, refreshed, unchanged = [], [], []
    old_count = len(mapping)
    if mode == "dict":
        template = next(iter(mapping.values()))
        for rel, sha in sorted(expected.items()):
            current = mapping.get(rel)
            if current is not None:
                old_sha = current if isinstance(current, str) else current["sha256"]
                if old_sha.lower() == sha.lower():
                    unchanged.append(rel)
                    continue
                # Only reviewed, release-verified files can have hashes refreshed.
                if isinstance(current, str):
                    mapping[rel] = sha
                else:
                    row = dict(current)
                    row["sha256"] = sha
                    if "bytes" in row:
                        row["bytes"] = checked_path(root, rel).stat().st_size
                    if "size_bytes" in row:
                        row["size_bytes"] = checked_path(root, rel).stat().st_size
                    mapping[rel] = row
                refreshed.append(rel)
                continue
            if isinstance(template, str):
                mapping[rel] = sha
            else:
                record = {"sha256": sha}
                if "bytes" in template:
                    record["bytes"] = checked_path(root, rel).stat().st_size
                if "size_bytes" in template:
                    record["size_bytes"] = checked_path(root, rel).stat().st_size
                mapping[rel] = record
            added.append(rel)
    else:
        bypath = {row["path"]: row for row in mapping}
        template = mapping[0]
        for rel, sha in sorted(expected.items()):
            if rel in bypath:
                old = bypath[rel]
                if old["sha256"].lower() == sha.lower():
                    unchanged.append(rel)
                    continue
                old["sha256"] = sha
                if "bytes" in old:
                    old["bytes"] = checked_path(root, rel).stat().st_size
                if "size_bytes" in old:
                    old["size_bytes"] = checked_path(root, rel).stat().st_size
                refreshed.append(rel)
                continue
            row = {"path": rel, "sha256": sha}
            if "bytes" in template:
                row["bytes"] = checked_path(root, rel).stat().st_size
            if "size_bytes" in template:
                row["size_bytes"] = checked_path(root, rel).stat().st_size
            mapping.append(row)
            added.append(rel)
        mapping.sort(key=lambda v: v["path"])
    for count_key in ("count", "file_count", "files_count", "total_files"):
        if isinstance(obj.get(count_key), int):
            obj[count_key] = len(mapping)
    result = {
        "snapshot_mapping": key,
        "previous_entries": old_count,
        "final_entries": len(mapping),
        "added": added,
        "refreshed_release_files": refreshed,
        "already_registered": len(unchanged),
        "mode": "write" if write else "dry-run",
    }
    print(json.dumps(result, indent=2), flush=True)
    if not write:
        print("DRY RUN: SOURCE_SNAPSHOT.json unchanged. Use --write to validate and save.", flush=True)
        return result
    if not added and not refreshed:
        print("No manifest changes required.", flush=True)
        if run_verifier:
            subprocess.run([sys.executable, str(verifier)], cwd=root, check=True)
        return result
    manifest.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        if run_verifier:
            run = subprocess.run([sys.executable, str(verifier)], cwd=root, text=True, capture_output=True)
            if run.returncode != 0:
                raise RuntimeError("Repository verifier rejected repaired snapshot; original restored. Details:\n" + (run.stdout + "\n" + run.stderr)[-4500:])
            print("PASS: repository snapshot verifier accepted all paths and digests.", flush=True)
    except BaseException:
        manifest.write_bytes(original)
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--write", action="store_true", help="Register reviewed release files and verify; otherwise dry-run")
    args = parser.parse_args()
    try:
        reconcile(args.repo, args.write)
    except Exception as exc:
        print("SNAPSHOT REPAIR FAILED: " + str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
