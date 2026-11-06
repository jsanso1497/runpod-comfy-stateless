"""Resolve only immutable source commits. Run during the dedicated Docker build."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import sys
from workflow_tools import blob_sha, build_workflows

PACKAGE = Path(__file__).resolve().parent


def checkout(row: dict, dest: Path) -> None:
    if dest.exists():
        raise RuntimeError('Build destination already exists: ' + str(dest))
    dest.mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(dest), *args], text=True).strip()
    git('init', '-q')
    git('remote', 'add', 'origin', row['url'])
    git('fetch', '--depth', '1', 'origin', row['commit'])
    git('checkout', '--detach', 'FETCH_HEAD')
    if git('rev-parse', 'HEAD') != row['commit']:
        raise RuntimeError('Source pin mismatch.')
    for relative, expected in row.get('blobs', {}).items():
        if blob_sha((dest / relative).read_bytes()) != expected:
            raise RuntimeError('Source blob mismatch: ' + relative)
    shutil.rmtree(dest / '.git')


def main():
    lock = json.loads((PACKAGE / 'sources.lock.json').read_text())
    sources = lock['sources']
    comfy = Path('/opt/ComfyUI')
    fizgig = comfy / 'custom_nodes' / 'ComfyUI-Fizgig-H3-Still'
    rmbg = Path('/opt/vendor/ComfyUI-RMBG')
    checkout(sources['comfy'], comfy)
    checkout(sources['fizgig'], fizgig)
    checkout(sources['rmbg'], rmbg)
    shutil.copytree(PACKAGE / 'nodes/rmbg_resize', comfy / 'custom_nodes/H3_Still_RMBG_Resize')
    original = PACKAGE / 'upstream_workflows'
    shutil.copytree(fizgig / 'example_workflows', original)
    licenses = PACKAGE / 'licenses'
    licenses.mkdir()
    for name, path in [('Fizgig', fizgig), ('RMBG', rmbg), ('ComfyUI', comfy)]:
        for candidate in ('LICENSE', 'LICENSE.txt', 'LICENSE.md'):
            if (path / candidate).is_file():
                shutil.copy2(path / candidate, licenses / (name + '-' + candidate))
                break
        else:
            raise RuntimeError('Missing upstream license for ' + name)
    shutil.copy2(fizgig / 'README.md', PACKAGE / 'UPSTREAM_FIZGIG_README.md')
    models = json.loads((PACKAGE / 'models.lock.json').read_text())['models']
    report = build_workflows(original, PACKAGE / 'workflows', models)
    (PACKAGE / 'workflow-provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    (PACKAGE / 'build-provenance.json').write_text(json.dumps({
        'source_lock': lock, 'python': sys.version,
        'status': 'Source assembly complete; CPU schema and GPU inference are separate tests.'
    }, indent=2)+'\n')
    print('Built four portable presets from three verified developer files.')


if __name__ == '__main__':
    main()
