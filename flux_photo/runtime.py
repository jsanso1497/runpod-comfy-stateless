"""Install bundled FLUX workflows and verified models; optional shared LoRA links."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

from assets import download_models

HERE = Path(__file__).resolve().parent


def flag(name, default='1'):
    value = os.environ.get(name, default)
    if value not in ('0', '1'):
        raise ValueError(name + ' must be 0 or 1.')
    return value == '1'


def install_workflows(home):
    target = Path(home) / 'user/default/workflows/FLUX_Photo'
    target.mkdir(parents=True, exist_ok=True)
    for source in sorted((HERE / 'workflows').glob('*.json')):
        try:
            with (target / source.name).open('x', encoding='utf-8') as out:
                out.write(source.read_text())
        except FileExistsError:
            pass  # A process restart must never overwrite the user's edited copy.
    return target


def lora_links():
    repo = os.environ.get('CONFIG_REPO', '')
    if not repo:
        return HERE / 'config/lora_links.txt'
    url = urlsplit(repo)
    if (url.scheme != 'https' or url.hostname != 'github.com' or url.username or url.password
            or url.query or url.fragment or not re.fullmatch(r'/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', url.path)):
        raise ValueError('CONFIG_REPO must be a plain GitHub HTTPS repository URL, without credentials.')
    ref = os.environ.get('CONFIG_REF', 'main')
    if ref.startswith('-') or any(c.isspace() for c in ref):
        raise ValueError('Invalid CONFIG_REF.')
    # Fixed separate path; live configuration only supplies links, never executable code.
    directory = Path('/workspace/flux-photo-config')
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    if not (directory / '.git').exists():
        subprocess.run(['git', 'init', '-q', str(directory)], check=True)
    with tempfile.TemporaryDirectory(prefix='flux-git-auth-') as tmp:
        askpass = Path(tmp) / 'askpass.sh'
        askpass.write_text('#!/bin/sh\ncase "$1" in\n *Username*) printf "%s\\n" x-access-token ;;\n'
                           ' *Password*) printf "%s\\n" "$GITHUB_TOKEN" ;;\nesac\n')
        askpass.chmod(0o700)
        if env.get('GITHUB_TOKEN'):
            env['GIT_ASKPASS'] = str(askpass)
        subprocess.run(['git', '-C', str(directory), 'fetch', '--depth', '1', repo, ref], env=env, check=True)
    subprocess.run(['git', '-C', str(directory), 'checkout', '--force', '--detach', 'FETCH_HEAD'], check=True)
    target = directory / 'config/lora_links.txt'
    if not target.is_file():
        raise FileNotFoundError('CONFIG_REPO is missing config/lora_links.txt.')
    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--comfy-home', type=Path, default=Path('/workspace/ComfyUI'))
    args = p.parse_args()
    home = args.comfy_home.resolve()
    if not home.is_relative_to(Path('/workspace')) or home == Path('/workspace'):
        raise ValueError('ComfyUI home must be a subdirectory of /workspace.')
    target = install_workflows(home)
    if flag('FLUX_PHOTO_DOWNLOAD_MODELS'):
        download_models(home)
    else:
        print('Model downloads disabled. Place all three exact model files in models/ before running.')
    if flag('SYNC_SHARED_LORAS'):
        subprocess.run([sys.executable, '/opt/shared-loras/sync.py', '--links', str(lora_links()),
                        '--comfy-home', str(home)], check=True)
    print('FLUX Photo workflows installed at', target, flush=True)

if __name__ == '__main__': main()
