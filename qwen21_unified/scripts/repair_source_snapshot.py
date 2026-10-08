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
PREFERRED_MAP_KEYS = ("files", "entries", "inventory", "source_files", "file_hashes", "sha256", "tracked", "tracked_files", "trackedFiles", "paths", "manifest", "snapshot")
HASH_KEYS = ("sha256", "hash", "digest", "checksum", "sha256sum", "sha")
PATH_KEYS = ("path", "file", "filename", "relative_path", "filepath", "name", "source")
SIZE_KEYS = ("bytes", "size_bytes", "size", "file_size")
HEX_WITH_PREFIX = re.compile(r"^(sha256:)([0-9a-fA-F]{64})$", re.IGNORECASE)


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


def parse_hash(value):
    """Return (lowercase hex, format prefix), or None for non-SHA256 data."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if HEX64.fullmatch(value):
        return (value.lower(), "")
    match = HEX_WITH_PREFIX.fullmatch(value)
    if match:
        return (match.group(2).lower(), match.group(1))
    return None


def hash_field(record):
    if isinstance(record, str):
        return ("", *parse_hash(record)) if parse_hash(record) else None
    if isinstance(record, dict):
        for k in HASH_KEYS:
            parsed = parse_hash(record.get(k))
            if parsed:
                return (k, *parsed)
    return None


def shape_summary(obj):
    if not isinstance(obj, dict):
        return {"type": type(obj).__name__}
    return {"top_level_keys": list(obj)[:25],
            "value_shapes": {k: type(v).__name__ for k, v in list(obj.items())[:25]}}


def find_candidates(obj, root: Path):
    """Discover a path->sha256 inventory by verifying existing hashes on disk.

    A candidate must contain at least ten path+SHA256 entries and a large
    majority of its listed existing files must match. We never select a map
    merely because a field happens to be called `files`.
    """
    result = []

    def score(items, key_path, mode, hashkey=None, pathkey=None):
        total = len(items)
        if total < 10:
            return
        safe = matching = exists = mismatched = 0
        for rel, sha in items:
            if not isinstance(rel, str) or not rel or rel.startswith('/') or '\\' in rel:
                continue
            try:
                target = checked_path(root, rel)
            except RuntimeError:
                continue
            safe += 1
            if target.is_file():
                exists += 1
                if digest(target).lower() == sha:
                    matching += 1
                else:
                    mismatched += 1
        # The release upload introduces unexpected paths, but previously
        # tracked files should still exist and match the snapshot. Require
        # strong positive evidence that this is the real snapshot mapping.
        if safe < 10 or exists < 8 or matching < 8 or matching < max(8, int(exists * 0.80)):
            return
        if safe < int(total * 0.80):
            return
        label = '.'.join(str(x) for x in key_path) or '(root)'
        preference = 1 if any(str(p) in PREFERRED_MAP_KEYS for p in key_path) else 0
        result.append({"container_path": key_path, "mode": mode, "hashkey": hashkey,
                       "pathkey": pathkey, "matches": matching, "entries": total,
                       "existing": exists, "mismatches": mismatched,
                       "preference": preference, "label": label})

    def visit(node, loc=(), depth=0):
        if depth > 5:
            return
        if isinstance(node, dict):
            rows = [(k, hash_field(v)[1]) for k, v in node.items() if hash_field(v)]
            if len(rows) >= 10 and len(rows) >= max(10, int(len(node) * 0.80)):
                score(rows, loc, 'dict')
            for k, val in node.items():
                if isinstance(val, (dict, list)):
                    visit(val, loc + (k,), depth + 1)
        elif isinstance(node, list):
            possibilities = []
            for pathkey in PATH_KEYS:
                for hashkey in HASH_KEYS:
                    rows = []
                    for row in node:
                        if not isinstance(row, dict):
                            continue
                        val = parse_hash(row.get(hashkey))
                        if isinstance(row.get(pathkey), str) and val:
                            rows.append((row[pathkey], val[0]))
                    if len(rows) >= 10 and len(rows) >= max(10, int(len(node)*0.80)):
                        possibilities.append((pathkey, hashkey, rows))
            if possibilities:
                pathkey, hashkey, rows = max(possibilities, key=lambda p: len(p[2]))
                score(rows, loc, 'list', hashkey, pathkey)
            # Lists of containers are uncommon but supported for nested schemas.
            if len(node) < 15:
                for i, val in enumerate(node):
                    if isinstance(val, (dict, list)):
                        visit(val, loc + (i,), depth + 1)

    visit(obj)
    return sorted(result, key=lambda x: (x['matches'], x['preference'], x['entries']), reverse=True)


def get_mapping(obj, path):
    value = obj
    for key in path:
        value = value[key]
    return value


def extract_map(snapshot: dict, root: Path):
    if not isinstance(snapshot, dict):
        raise RuntimeError('SOURCE_SNAPSHOT.json must be an object; existing verifier and manifest should be inspected')
    candidates = find_candidates(snapshot, root)
    if not candidates:
        summary = json.dumps(shape_summary(snapshot), sort_keys=True)
        raise RuntimeError('Could not identify a verified path/SHA256 inventory in SOURCE_SNAPSHOT.json. '
                           'No files changed. Actual top-level structure: ' + summary)
    best = candidates[0]
    if len(candidates) > 1 and (candidates[1]['matches'], candidates[1]['entries']) == (best['matches'], best['entries']):
        raise RuntimeError('Ambiguous snapshot inventories: ' + ', '.join(c['label'] for c in candidates[:5]))
    mapping = get_mapping(snapshot, best['container_path'])
    print('Detected SOURCE_SNAPSHOT inventory:', json.dumps({k: v for k, v in best.items() if k not in ('preference',)}, sort_keys=True), flush=True)
    return best, mapping


def set_record_hash(record, sha: str, kind: str = ''):
    original = hash_field(record)
    if original is None:
        raise RuntimeError('Snapshot hash record format changed during repair')
    field, _, prefix = original
    newval = prefix + sha
    if isinstance(record, str):
        return newval
    result = dict(record)
    result[field] = newval
    return result


def new_record(template, sha, target):
    if isinstance(template, str):
        parsed = parse_hash(template)
        return (parsed[1] if parsed else '') + sha
    old = hash_field(template)
    if not old:
        raise RuntimeError('Snapshot record has no recognized SHA256 field')
    hash_key, _, prefix = old
    # Copy the schema keys that a legacy verifier may require. Only known
    # size information is refreshed. The original verifier remains the gate.
    obj = dict(template)
    obj[hash_key] = prefix + sha
    for key in SIZE_KEYS:
        if key in obj:
            obj[key] = target.stat().st_size
    return obj


def reconcile(root: Path, write: bool = False, run_verifier: bool = True):
    root = root.resolve()
    manifest = root / 'SOURCE_SNAPSHOT.json'
    if not manifest.is_file():
        raise RuntimeError('SOURCE_SNAPSHOT.json missing from repository')
    verifier = root / 'tools/verify_snapshot.py'
    if not verifier.is_file():
        raise RuntimeError('tools/verify_snapshot.py missing; cannot validate repaired manifest')
    expected = load_release_files(root)
    original = manifest.read_bytes()
    obj = json.loads(original)
    desc, mapping = extract_map(obj, root)
    mode = desc['mode']
    added, refreshed, unchanged = [], [], []
    old_count = len(mapping)
    if mode == 'dict':
        valid_records = [(k, v) for k, v in mapping.items() if hash_field(v)]
        template = valid_records[0][1]
        for rel, sha in sorted(expected.items()):
            current = mapping.get(rel)
            if current is not None:
                actual = hash_field(current)
                if not actual:
                    raise RuntimeError(f'Existing snapshot record cannot be read: {rel}')
                if actual[1] == sha.lower():
                    unchanged.append(rel)
                    continue
                mapping[rel] = set_record_hash(current, sha)
                if isinstance(mapping[rel], dict):
                    for k in SIZE_KEYS:
                        if k in mapping[rel]:
                            mapping[rel][k] = checked_path(root, rel).stat().st_size
                refreshed.append(rel)
            else:
                mapping[rel] = new_record(template, sha, checked_path(root, rel))
                added.append(rel)
    else:
        pathkey, hashkey = desc['pathkey'], desc['hashkey']
        bypath = {row[pathkey]: row for row in mapping if isinstance(row, dict) and pathkey in row and hashkey in row}
        if len(bypath) != len(mapping):
            raise RuntimeError('Snapshot file list contains duplicate or malformed entries; refusing repair')
        template = mapping[0]
        for rel, sha in sorted(expected.items()):
            if rel in bypath:
                old = bypath[rel]
                parsed = parse_hash(old[hashkey])
                if parsed and parsed[0] == sha.lower():
                    unchanged.append(rel)
                    continue
                old[hashkey] = (parsed[1] if parsed else '') + sha
                for k in SIZE_KEYS:
                    if k in old:
                        old[k] = checked_path(root, rel).stat().st_size
                refreshed.append(rel)
            else:
                row = dict(template)
                row[pathkey] = rel
                original_format = parse_hash(row[hashkey])
                row[hashkey] = (original_format[1] if original_format else '') + sha
                for k in SIZE_KEYS:
                    if k in row:
                        row[k] = checked_path(root, rel).stat().st_size
                mapping.append(row)
                added.append(rel)
        mapping.sort(key=lambda row: row[pathkey])
    # Update counts only if they previously counted this very inventory.
    parents = [obj]
    parent = obj
    for key in desc['container_path'][:-1]:
        parent = parent[key]
        if isinstance(parent, dict):
            parents.append(parent)
    for parent in parents:
        if not isinstance(parent, dict):
            continue
        for count_key in ('count', 'file_count', 'files_count', 'total_files', 'entry_count'):
            if parent.get(count_key) == old_count and type(parent.get(count_key)) is int:
                parent[count_key] = len(mapping)
    result = {'snapshot_mapping': desc['label'], 'schema_mode': mode, 'previous_entries': old_count,
              'final_entries': len(mapping), 'added': added,
              'refreshed_release_files': refreshed, 'already_registered': len(unchanged),
              'mode': 'write' if write else 'dry-run'}
    print(json.dumps(result, indent=2), flush=True)
    if not write:
        print('DRY RUN: SOURCE_SNAPSHOT.json unchanged. Use --write to validate and save.', flush=True)
        return result
    if not added and not refreshed:
        if run_verifier:
            subprocess.run([sys.executable, str(verifier)], cwd=root, check=True)
        print('Snapshot already registered and verified.', flush=True)
        return result
    manifest.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    try:
        if run_verifier:
            run = subprocess.run([sys.executable, str(verifier)], cwd=root, text=True, capture_output=True)
            if run.returncode:
                raise RuntimeError('Repository verifier rejected repaired snapshot; original restored. Details:\n'
                                   + (run.stdout + '\n' + run.stderr)[-6000:])
            print('PASS: repository snapshot verifier accepted all paths and digests.', flush=True)
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
