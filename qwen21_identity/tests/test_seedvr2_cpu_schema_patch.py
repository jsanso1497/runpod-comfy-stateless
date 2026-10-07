from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "patch_seedvr2_cpu_schema.py"

spec = importlib.util.spec_from_file_location("patch_seedvr2_cpu_schema", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "seedvr2"
    for rel in mod.FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "class X:\n"
            "    @classmethod\n"
            "    def define_schema(cls):\n"
            "        devices = get_device_list()\n"
            "        return devices[0]\n",
            encoding="utf-8",
        )
    return root


def test_patch_adds_empty_device_fallback(tmp_path):
    root = _make_tree(tmp_path)
    for rel in mod.FILES:
        mod.patch_file(root / rel)
        text = (root / rel).read_text(encoding="utf-8")
        assert 'if not devices:' in text
        assert 'devices = ["cpu"]' in text


def test_patch_is_idempotent(tmp_path):
    root = _make_tree(tmp_path)
    for rel in mod.FILES:
        path = root / rel
        mod.patch_file(path)
        once = path.read_text(encoding="utf-8")
        mod.patch_file(path)
        assert path.read_text(encoding="utf-8") == once


def test_patch_fails_closed_if_upstream_shape_changes(tmp_path):
    path = tmp_path / "changed.py"
    path.write_text("devices = something_else()\n", encoding="utf-8")
    try:
        mod.patch_file(path)
    except RuntimeError as exc:
        assert "Expected exactly one" in str(exc)
    else:
        raise AssertionError("patch should fail closed when the pinned source pattern is absent")
