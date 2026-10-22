"""Regression tests for inherited spin / Click conflict in the PyTorch image.

The pip integration test uses minimal locally-built fixture wheels with the
reported dependency metadata. It does not claim to test real H3 inference.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import packaging
import pytest

from swapkit import base_cleanup as cleanup
from swapkit import build

ROOT = Path(__file__).resolve().parents[1]


def dist(name, version="1", requires=()):
    return SimpleNamespace(metadata={"Name": name}, version=version, requires=list(requires),
                           locate_file=lambda _: Path("/usr/local/lib/python3.12/dist-packages"))


def absent(name):
    raise cleanup.md.PackageNotFoundError(name)


def test_cleanup_is_one_allowlisted_developer_tool():
    assert cleanup.TARGET == "spin"
    assert cleanup.EXPECTED_VERSION == "0.17"
    assert cleanup.PROTECTED_PACKAGES == ("torch", "torchvision", "torchaudio")


def test_reverse_dependencies_include_active_runtime_requirement():
    rows = [dist("runtime", requires=["spin>=0.17"]),
            dist("other", requires=["Spin; python_version >= '3.10'"]),
            dist("spin", requires=["click<8.4"])]
    assert cleanup.active_dependents(rows) == ["other", "runtime"]


def test_reverse_dependencies_ignore_inactive_dev_extras():
    rows = [dist("numpy", requires=['spin; extra == "dev"']),
            dist("other", requires=['spin; python_version < "2"'])]
    assert cleanup.active_dependents(rows) == []


def test_cleanup_refuses_a_runtime_dependent(monkeypatch):
    monkeypatch.setattr(cleanup, "protected_snapshot", lambda: {})
    monkeypatch.setattr(cleanup.md, "distribution", lambda _: dist("spin", "0.17"))
    monkeypatch.setattr(cleanup, "active_dependents", lambda: ["runtime-package"])
    calls=[]
    monkeypatch.setattr(cleanup.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="runtime-package"):
        cleanup.remove_unused_spin()
    assert not calls


def test_cleanup_refuses_an_unreviewed_base_version(monkeypatch):
    monkeypatch.setattr(cleanup, "protected_snapshot", lambda: {})
    monkeypatch.setattr(cleanup.md, "distribution", lambda _: dist("spin", "0.18"))
    with pytest.raises(RuntimeError, match="Unexpected base-image spin version"):
        cleanup.remove_unused_spin()


def test_cleanup_skips_if_tool_is_already_absent(monkeypatch):
    snapshot = {"torch": {"version": "2.11.0", "location": "/base"}}
    monkeypatch.setattr(cleanup, "protected_snapshot", lambda: snapshot)
    monkeypatch.setattr(cleanup.md, "distribution", absent)
    monkeypatch.setattr(cleanup.subprocess, "run", lambda *a, **k: pytest.fail("unexpected removal"))
    assert cleanup.remove_unused_spin() == {"removed": [], "protected_packages": snapshot}


def test_cleanup_uninstall_is_narrow_and_preserves_torch(monkeypatch):
    snapshot = {"torch": {"version": "2.11.0", "location": "/base"}}
    monkeypatch.setattr(cleanup, "protected_snapshot", lambda: snapshot)
    installed = [True]
    def distribution(name):
        return dist("spin", "0.17") if installed[0] else absent(name)
    monkeypatch.setattr(cleanup.md, "distribution", distribution)
    monkeypatch.setattr(cleanup, "active_dependents", lambda: [])
    calls=[]
    def run(args, **kw):
        calls.append((args, kw)); installed[0] = False
    monkeypatch.setattr(cleanup.subprocess, "run", run)
    record=cleanup.remove_unused_spin()
    assert calls == [([sys.executable, "-m", "pip", "uninstall", "--yes",
                      "--break-system-packages", "spin"], {"check": True, "timeout": 120})]
    assert record["protected_packages"] == snapshot
    assert record["removed"] == [{"name": "spin", "version": "0.17"}]


def test_cleanup_requires_actual_removal(monkeypatch):
    monkeypatch.setattr(cleanup, "protected_snapshot", lambda: {})
    monkeypatch.setattr(cleanup.md, "distribution", lambda _: dist("spin", "0.17"))
    monkeypatch.setattr(cleanup, "active_dependents", lambda: [])
    monkeypatch.setattr(cleanup.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="still visible"):
        cleanup.remove_unused_spin()


def test_cleanup_fails_on_pip_error(monkeypatch):
    monkeypatch.setattr(cleanup, "protected_snapshot", lambda: {})
    monkeypatch.setattr(cleanup.md, "distribution", lambda _: dist("spin", "0.17"))
    monkeypatch.setattr(cleanup, "active_dependents", lambda: [])
    def fail(*a, **k):
        raise subprocess.CalledProcessError(1, a[0])
    monkeypatch.setattr(cleanup.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        cleanup.remove_unused_spin()


def test_cli_refuses_normal_host_invocation(monkeypatch):
    monkeypatch.delenv("H3_SWAP_BUILD_CLEANUP", raising=False)
    with pytest.raises(RuntimeError, match="only permitted by the Dockerfile"):
        cleanup.require_image_build()


def test_cleanup_order_and_no_global_pep668_override():
    text = (ROOT / "Dockerfile").read_text()
    assert text.index("COPY swapkit/base_cleanup.py") < text.index("H3_SWAP_BUILD_CLEANUP=1")
    assert text.index("H3_SWAP_BUILD_CLEANUP=1") < text.index("RUN python -m venv")
    assert "ENV PIP_BREAK_SYSTEM_PACKAGES" not in text
    assert "python -m swapkit.build" in text
    assert "--system-site-packages" in text


def test_single_requirement_transaction_and_immediate_strict_check(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(build, "run", lambda *a: calls.append(a))
    monkeypatch.setattr(build, "ROOT", tmp_path / "kit")
    comfy = tmp_path / "ComfyUI"
    constraints = tmp_path / "constraints.txt"
    build.install_application_requirements(comfy, constraints)
    assert calls == [
        (sys.executable, "-m", "pip", "install", "--constraint", str(constraints),
         "-r", str(comfy / "requirements.txt"), "-r", str(tmp_path / "kit/requirements.txt")),
        (sys.executable, "-m", "pip", "check"),
    ]


def _wheel(folder: Path, name: str, version: str, requirements=()) -> Path:
    """Fixture metadata only; never ship these as actual third-party packages."""
    normalized = name.replace("-", "_")
    info = f"{normalized}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {req}\n" for req in requirements)
    members = {
        normalized + "/__init__.py": f"__version__ = {version!r}\n".encode(),
        info + "/METADATA": metadata.encode(),
        info + "/WHEEL": b"Wheel-Version: 1.0\nGenerator: h3-offline-regression-fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = io.StringIO()
    rows = csv.writer(record, lineterminator="\n")
    for path, data in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        rows.writerow([path, "sha256=" + digest, str(len(data))])
    rows.writerow([info + "/RECORD", "", ""])
    members[info + "/RECORD"] = record.getvalue().encode()
    dest = folder / f"{normalized}-{version}-py3-none-any.whl"
    with ZipFile(dest, "w") as z:
        for path, data in members.items():
            z.writestr(path, data)
    return dest


def test_real_pip_reproduces_inherited_conflict_then_cleanup_passes(tmp_path):
    """Real pip + synthetic tiny packages: reproduce, repair, and check unchanged torch."""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_TARGET", "PIP_PREFIX", "PIP_USER",
                "PIP_REQUIRE_VIRTUALENV", "PIP_BREAK_SYSTEM_PACKAGES"):
        env.pop(key, None)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INDEX="1")
    def run(args, **kw):
        return subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True,
                              timeout=60, **kw)
    def make_env(name):
        path=tmp_path/name
        base=getattr(sys, "_base_executable", sys.executable)
        run([base,"-m","venv",path],check=True)
        py=path/("Scripts/python.exe" if os.name=="nt" else "bin/python")
        site=Path(run([py,"-c","import sysconfig;print(sysconfig.get_path('purelib'))"],check=True).stdout.strip())
        return py,site
    base,base_site=make_env("base-image-fixture")
    app,app_site=make_env("app-venv-fixture")
    # Reproduce the visibility of --system-site-packages without modifying the host.
    (app_site/"base-image-fixture.pth").write_text(str(base_site)+"\n")
    # The production base supplies packaging; copy it only inside this temp fixture.
    shutil.copytree(Path(packaging.__file__).parent,base_site/"packaging")
    wheelhouse=tmp_path/"wheels";wheelhouse.mkdir()
    old_click=_wheel(wheelhouse,"click","8.3.1")
    spin=_wheel(wheelhouse,"spin","0.17",["click>=8,!=8.3.0,<8.4"])
    torch=_wheel(wheelhouse,"torch","2.11.0")
    new_click=_wheel(wheelhouse,"click","8.5.0")
    hub=_wheel(wheelhouse,"huggingface-hub","1.33.0",["click>=8.4.2,<9.0.0"])
    run([base,"-m","pip","install","--no-index",old_click,spin,torch],check=True)
    run([app,"-m","pip","install","--no-index",new_click,hub],check=True)
    broken=run([app,"-m","pip","check"])
    assert broken.returncode==1
    assert "spin 0.17" in broken.stdout and "click 8.5.0" in broken.stdout
    code=("import sys;sys.path.insert(0,"+repr(str(ROOT))+");"
          "from swapkit.base_cleanup import remove_unused_spin;"
          "import json;print(json.dumps(remove_unused_spin()))")
    removal=run([base,"-c",code],check=True)
    assert 'Successfully uninstalled spin-0.17' in removal.stdout
    record=json.loads(removal.stdout.strip().splitlines()[-1])
    assert record["protected_packages"]["torch"]["version"]=="2.11.0"
    check=run([app,"-m","pip","check"],check=True)
    assert "No broken requirements found" in check.stdout
    again=run([base,"-c",code],check=True)
    assert json.loads(again.stdout.strip().splitlines()[-1])["removed"]==[]
    unchanged=run([base,"-c","import importlib.metadata as m;print(m.version('click'),m.version('torch'))"],check=True)
    assert unchanged.stdout.strip()=="8.3.1 2.11.0"
