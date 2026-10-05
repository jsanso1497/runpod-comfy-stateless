#!/usr/bin/env python3
"""Fail before a costly Docker pull when the source upload is incomplete."""
import argparse
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(target):
    wanted = ['.dockerignore', 'shared_loras/sync.py', 'shared_loras/tests/test_links.py',
              'config/lora_links.txt']
    if target == 'portrait':
        wanted += ['h3_portrait/Dockerfile', 'h3_portrait/verify_release.py',
                   '.github/workflows/build-h3-portrait.yml']
    else:
        wanted += ['Dockerfile', '.github/workflows/build-image.yml', 'scripts/prepare_image.sh',
                   'scripts/check_rebalance_runtime.py', 'config/runtime.json']
    for name in wanted:
        if not (ROOT/name).is_file():
            raise ValueError('Missing repository file: ' + name)
    test = ROOT/'shared_loras/tests/test_links.py'
    ast.parse(test.read_text(), filename=str(test))
    if 'self.disk_usage = disk_patch.start()' not in test.read_text():
        raise ValueError('The corrected shared-LoRA disk-fixture test file is missing.')
    if target == 'portrait':
        subprocess.run([sys.executable, str(ROOT/'h3_portrait/verify_release.py'),
                        '--source', str(ROOT/'h3_portrait')], check=True)
    else:
        for name in ('test_update', 'test_krea_rebalance', 'test_ollama_h3',
                     'test_quality', 'test_single_person', 'test_user_directed_h3'):
            f=ROOT/'tests'/f'{name}.py'
            ast.parse(f.read_text(), filename=str(f))
    print('REPOSITORY PREFLIGHT PASS: ' + target)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--target', choices=('portrait', 'general'), required=True)
    check(ap.parse_args().target)
