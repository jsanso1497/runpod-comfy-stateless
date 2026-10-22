"""Regression coverage for the 2026-10-22 PEP 668 Docker build failure."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from swapkit import build

ROOT = Path(__file__).resolve().parents[1]


def test_build_rejects_system_python_before_any_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "sys", SimpleNamespace(
        prefix="/usr", base_prefix="/usr", executable="/usr/bin/python"))
    monkeypatch.setattr(build, "ROOT", tmp_path / "not-opened")
    with pytest.raises(RuntimeError, match="application virtual environment"):
        build.main()


def test_build_reports_the_selected_virtual_environment(monkeypatch, capsys):
    monkeypatch.setattr(build, "sys", SimpleNamespace(
        prefix="/opt/h3-swap-venv", base_prefix="/usr",
        executable="/opt/h3-swap-venv/bin/python"))
    record = build.require_virtualenv()
    assert record == {
        "executable": "/opt/h3-swap-venv/bin/python",
        "prefix": "/opt/h3-swap-venv", "base_prefix": "/usr"}
    assert json.loads(capsys.readouterr().out.split("] ", 1)[1]) == record


def test_docker_creates_and_selects_venv_before_build():
    text = (ROOT / "Dockerfile").read_text()
    assert "python3-venv" in text
    create = text.index("RUN python -m venv --system-site-packages /opt/h3-swap-venv")
    path = text.index('ENV PATH="/opt/h3-swap-venv/bin:${PATH}"')
    assert text.index("python3-venv") < create < path < text.index("RUN python -m swapkit.build")
    assert "ENV VIRTUAL_ENV=/opt/h3-swap-venv" in text


def test_docker_keeps_application_installs_protected():
    text = (ROOT / "Dockerfile").read_text()
    assert "--break-system-packages" not in text
    assert "PIP_BREAK_SYSTEM_PACKAGES" not in text
    assert "EXTERNALLY-MANAGED" not in text
    # The only scoped exception lives in the checked base-image spin uninstall.
    # Package installs still use the protected application virtual environment.
    assert "H3_SWAP_BUILD_CLEANUP=1 python /tmp/h3-base-cleanup.py" in text
    cleanup = (ROOT / "swapkit/base_cleanup.py").read_text()
    assert '\"uninstall\", \"--yes\"' in cleanup
    assert '\"--break-system-packages\", TARGET' in cleanup
    assert 'TARGET = \"spin\"' in cleanup


def test_qualified_runtime_and_ollama_pins_are_preserved():
    text = (ROOT / "Dockerfile").read_text()
    pins = json.loads((ROOT / "pins.json").read_text())
    assert "FROM " + pins["base_image"] in text
    assert pins["ollama"]["url"] in text
    assert pins["ollama"]["sha256"] + "  /tmp/ollama.tar.zst" in text
    assert "sha256sum -c -" in text


def test_build_download_and_static_check_are_separate_steps():
    text = (ROOT / "Dockerfile").read_text()
    assert "RUN python -m swapkit.build\n" in text
    assert "RUN curl --fail --location --retry 3" in text
    assert "RUN python -m compileall -q swapkit custom_nodes" in text
    assert "&& python -m swapkit.check --static" in text


def test_runtime_keeps_the_interpreter_selected_by_the_image():
    text = (ROOT / "Dockerfile").read_text()
    assert 'ENTRYPOINT ["/usr/bin/tini", "-s", "--", "python", "-m", "swapkit.runtime"]' in text
    runtime = (ROOT / "swapkit/runtime.py").read_text()
    assert 'args=[sys.executable,"/opt/ComfyUI/main.py"' in runtime


def test_build_keeps_dependency_constraints_and_records_python():
    text = (ROOT / "swapkit/build.py").read_text()
    assert 'for n in ("torch", "torchvision", "torchaudio")' in text
    assert '"--constraint", str(constraints)' in text
    assert 'record["python_environment"] = python_environment' in text
    assert 'run(sys.executable, "-m", "pip", "check")' in text


def _offline_test_wheel(folder: Path) -> Path:
    """Build a tiny pure-Python wheel without a package index or build backend."""
    name = "h3_venv_regression_probe"
    dist = name + "-0.0.0.dist-info"
    files = {
        name + "/__init__.py": b"VALUE = 'installed in the application venv'\n",
        dist + "/METADATA": b"Metadata-Version: 2.1\nName: h3-venv-regression-probe\nVersion: 0.0.0\n",
        dist + "/WHEEL": b"Wheel-Version: 1.0\nGenerator: h3-build-regression\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([path, "sha256=" + digest, str(len(data))])
    writer.writerow([dist + "/RECORD", "", ""])
    files[dist + "/RECORD"] = record.getvalue().encode()
    wheel = folder / (name + "-0.0.0-py3-none-any.whl")
    with ZipFile(wheel, "w") as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return wheel


def test_real_venv_installs_offline_and_inherits_base_site_packages(tmp_path):
    """Exercise real venv/pip, not a mocked installer. No network or GPU required."""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(key, None)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INDEX="1")
    dest = tmp_path / "application-venv"
    base = getattr(sys, "_base_executable", sys.executable)
    subprocess.run([base, "-m", "venv", "--system-site-packages", str(dest)],
                   env=env, check=True, capture_output=True, text=True, timeout=60)
    py = dest / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wheel = _offline_test_wheel(tmp_path)
    subprocess.run([str(py), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
                   env=env, check=True, capture_output=True, text=True, timeout=60)
    code = (
        "import json,sys,h3_venv_regression_probe as probe;"
        "from pip._internal.utils.misc import check_externally_managed;"
        "check_externally_managed();"
        "print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,"
        "'path':probe.__file__,'value':probe.VALUE,'sys_path':sys.path}))"
    )
    result = subprocess.run([str(py), "-c", code], env=env, check=True,
                            capture_output=True, text=True, timeout=30)
    info = json.loads(result.stdout)
    assert Path(info["prefix"]).resolve() == dest.resolve()
    assert info["prefix"] != info["base_prefix"]
    assert Path(info["path"]).resolve().is_relative_to(dest.resolve())
    assert info["value"] == "installed in the application venv"
    assert "include-system-site-packages = true" in (dest / "pyvenv.cfg").read_text().lower()
    base_result = subprocess.run([base, "-c", "import site,json;print(json.dumps(site.getsitepackages()))"],
                                 env=env, check=True, capture_output=True, text=True, timeout=30)
    existing_base_paths = [p for p in json.loads(base_result.stdout) if Path(p).is_dir()]
    assert existing_base_paths
    assert set(existing_base_paths).issubset(set(info["sys_path"]))
