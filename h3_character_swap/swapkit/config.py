from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

ROOT = Path(os.environ.get("H3_SWAP_ROOT", Path(__file__).resolve().parents[1]))
RELEASE = "2026-10-27.1"


def data_root() -> Path:
    p = Path(os.environ.get("H3_SWAP_DATA", "/workspace"))
    if not p.is_absolute():
        raise ValueError("H3_SWAP_DATA must be an absolute path")
    return p


def boolean(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "1" if default else "0")
    if v not in ("0", "1"):
        raise ValueError(name + " must be 0 or 1")
    return v == "1"


def variants() -> set[str]:
    result = set(x.strip() for x in os.environ.get("H3_SWAP_VARIANTS", "akatz").split(",") if x.strip())
    if not result or result - {"akatz", "bfs"}:
        raise ValueError("H3_SWAP_VARIANTS must be akatz, bfs, or akatz,bfs")
    if "bfs" in result and not boolean("H3_SWAP_BFS_LICENSE_REVIEWED"):
        raise ValueError("Review licenses/BFS_CHARACTER_SWAP.md, then explicitly set H3_SWAP_BFS_LICENSE_REVIEWED=1 to prepare the optional BFS comparison.")
    return result


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("Invalid relative path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(x in ("..", ".") for x in rel.parts):
        raise ValueError("Path escapes root")
    root = root.resolve()
    p = root / relative
    if p.is_symlink() or not p.resolve().is_relative_to(root) or p.resolve() == root:
        raise ValueError("Path escapes root or is a symlink")
    return p


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refusing symlink")
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".swap-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def public_child_env() -> dict:
    env = dict(os.environ)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CIVITAI_TOKEN", "WB_PASSWORD", "hf_token", "civit_token"):
        env.pop(name, None)
    return env


def upload_mb() -> int:
    raw = os.environ.get("H3_SWAP_MAX_UPLOAD_MB", "1024")
    if not re.fullmatch(r"[0-9]+", raw) or not 10 <= int(raw) <= 2048:
        raise ValueError("H3_SWAP_MAX_UPLOAD_MB must be 10..2048")
    return int(raw)
