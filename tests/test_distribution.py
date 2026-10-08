"""Offline guarantees for the same-repository, no-CLI distribution."""
from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("name", [
    ".github/workflows/build.yml", ".github/workflows/ci.yml",
    ".github/workflows/pages.yml", ".gitignore", ".dockerignore", ".env.example",
])
def test_dotfiles_are_present(name):
    assert (ROOT / name).is_file()


def test_existing_repository_default():
    release = json.loads((ROOT / "catalog/release.json").read_text())
    assert release["repository_default"] == "jsanso1497/runpod-comfy-stateless"
    assert release["version"] == "0.1.1"


def test_new_repository_publish_scripts_removed():
    assert not (ROOT / "scripts/publish.sh").exists()
    assert not (ROOT / "scripts/publish.ps1").exists()
    assert "publish.sh" not in (ROOT / ".github/workflows/ci.yml").read_text()


def test_mac_installation_is_documented():
    readme = (ROOT / "README.md").read_text()
    guide = (ROOT / "docs/mac-install.md").read_text()
    assert "docs/mac-install.md" in readme
    assert "Browser only" in guide
    assert "`.git`" in guide


def test_source_and_generated_site_use_existing_repository():
    template = (ROOT / "site/index.template.html").read_text()
    page = (ROOT / "site/index.html").read_text()
    assert "Create your repository" not in template
    assert "scripts/publish.sh" not in page
    assert "jsanso1497/runpod-comfy-stateless" in page
    assert "RUNPOD_SECRET_hf_token" in page
    assert "RUNPOD_SECRET_civit_token" in page
