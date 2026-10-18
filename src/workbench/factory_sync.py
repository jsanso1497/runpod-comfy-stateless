"""Versioned factory workflows: new image sources without overwriting user edits."""
from pathlib import Path
import hashlib
import json
import os
import tempfile


def sync_factory(source_root, data_root, tasks, workspace):
    source_root, data_root = Path(source_root), Path(data_root)
    rows = []
    bundle = hashlib.sha256()
    for task in tasks:
        if workspace not in task['workspaces']:
            continue
        for variant in task['variants']:
            relative = variant['file']
            path = source_root/relative
            if not path.resolve().is_relative_to(source_root.resolve()):
                raise ValueError('Factory source escapes repository.')
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            rows.append((task['id'], path.name, raw, digest))
            bundle.update((task['id']+'/'+relative+'\0'+digest+'\n').encode())
    folder = data_root/'comfy-user/default/workflows/Workbench Factory'/('source-'+bundle.hexdigest()[:12])
    records = []
    for task, name, raw, digest in rows:
        dest = folder/task/name
        if not dest.resolve().is_relative_to(data_root.resolve()) or dest.is_symlink():
            raise ValueError('Factory destination escapes the data volume.')
        if dest.exists() and dest.read_bytes() != raw:
            # A user edited the factory copy: keep it and add a distinct original.
            dest = dest.with_name(dest.stem+'__factory_'+digest[:12]+dest.suffix)
        if not dest.resolve().is_relative_to(data_root.resolve()) or dest.is_symlink():
            raise ValueError('Factory alternate destination escapes the data volume.')
        if dest.exists() and dest.read_bytes() != raw:
            raise ValueError('Both factory workflow copies were edited; preserve them and choose a separate user filename.')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix='.factory-')
            try:
                with os.fdopen(fd, 'wb') as handle:
                    handle.write(raw)
                os.chmod(tmp, 0o644)
                os.replace(tmp, dest)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        records.append({'task': task, 'path': str(dest.relative_to(data_root)), 'source_sha256': digest})
    return {'directory': str(folder.relative_to(data_root)), 'workflows': records}
