#!/usr/bin/env python3
"""Local H3 Image controls: status, idle memory release, supervised restart, ZIP outputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
from assets import data_root

URL = 'http://127.0.0.1:8189'


def request(path, data=None):
    encoded = json.dumps(data).encode() if data is not None else None
    headers = {'Content-Type': 'application/json'} if encoded is not None else {}
    with urllib.request.urlopen(urllib.request.Request(URL + path, data=encoded, headers=headers), timeout=8) as r:
        body = r.read()
        return json.loads(body) if body else {}


def assert_idle(queue, interrupt=False):
    """Pending jobs matter too: never silently discard a user's queue."""
    if (queue.get('queue_running') or queue.get('queue_pending')) and not interrupt:
        raise RuntimeError('Jobs are running or queued. Finish/clear the queue, or add --interrupt to cancel them explicitly.')


def memory_status():
    for name in ('memory.current', 'memory.max', 'memory.events'):
        file = Path('/sys/fs/cgroup') / name
        if file.exists():
            print('Container ' + name + ': ' + file.read_text().strip())


def zip_outputs(root):
    """Stream files to a unique ZIP64 archive outside output; do not touch originals."""
    root = Path(root).resolve()
    source = root / 'output'
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError('Expected a real output folder: ' + str(source))
    entries = []
    total = 0
    for folder, dirs, files in os.walk(source, followlinks=False):
        folder = Path(folder)
        dirs[:] = sorted(d for d in dirs if not (folder / d).is_symlink())
        for name in sorted(files):
            path = folder / name
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode):
                continue
            entries.append((path, st.st_size, st.st_mtime_ns))
            total += st.st_size
    if not entries:
        raise RuntimeError('No regular output files exist yet.')
    destination = root / 'exports'
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError('Refusing a symlinked export directory.')
    # PNGs usually compress poorly. Store without recompression to minimize CPU/RAM pressure.
    if shutil.disk_usage(destination).free < total + len(entries) * 1024 + 256 * 1024 ** 2:
        raise RuntimeError('Not enough disk space for an additional copy of the outputs plus headroom.')
    stem = 'H3_Image_Outputs_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    final = destination / (stem + '.zip')
    partial = destination / (stem + '.zip.part')
    try:
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path, size, modified in entries:
                flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                fd = os.open(path, flags)
                with os.fdopen(fd, 'rb') as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode) or (before.st_size, before.st_mtime_ns) != (size, modified):
                        raise RuntimeError('Output changed during export. Finish generation, then retry: ' + path.name)
                    info = zipfile.ZipInfo(path.relative_to(root).as_posix(), time.localtime(before.st_mtime)[:6])
                    info.compress_type = zipfile.ZIP_STORED
                    with archive.open(info, 'w', force_zip64=True) as zipped:
                        shutil.copyfileobj(stream, zipped, length=1024*1024)
                    after = os.fstat(stream.fileno())
                    if (after.st_size, after.st_mtime_ns) != (size, modified):
                        raise RuntimeError('Output changed while being read. Retry after generation finishes.')
        # Refuse overwrites even in the unlikely event of a name collision.
        os.link(partial, final)
        partial.unlink()
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print('Created:', final)
    print('Files:', len(entries), '| Original bytes:', total)
    print('Download from HTTP 8888 > Files > h3-image > exports. Originals are unchanged.')
    return final


def supervisor_running(root):
    state = Path('/run/h3-image-v1/supervisor.json')
    try:
        info = json.loads(state.read_text())
        if Path(info['root']).resolve() != Path(root).resolve():
            return False
        pid = int(info['pid'])
        cmd = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
        return b'/opt/h3-image/runtime/service_launcher.py' in cmd
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'free', 'restart', 'zip'])
    parser.add_argument('--interrupt', action='store_true', help='Explicitly cancel running and queued jobs.')
    args = parser.parse_args()
    root = data_root()
    if args.action == 'status':
        try:
            info = request('/object_info')
            print('Native H3 Image node:', 'MiniMaxH3ImageConditioning' in info)
            print('Workspace helpers:', all(n in info for n in ('H3ImageManualConditioning', 'H3ImageSixStepSchedule')))
            q = request('/queue')
            print('Running:', len(q.get('queue_running', [])), 'Queued:', len(q.get('queue_pending', [])))
            for device in request('/system_stats').get('devices', []):
                print('GPU:', device.get('name'), 'VRAM free / total:', device.get('vram_free'), '/', device.get('vram_total'))
        except Exception as e:
            print('ComfyUI not reachable:', type(e).__name__)
        memory_status()
        print('Supervisor running:', supervisor_running(root))
        print('Logs:', root / 'logs/comfy.log')
        print('Persistent storage free: %.1f GB' % (shutil.disk_usage(root).free/1e9))
        return
    try:
        queue = request('/queue')
        assert_idle(queue, args.interrupt and args.action != 'zip')
    except OSError:
        if args.action == 'free':
            raise RuntimeError('ComfyUI is not responding. Use restart instead.') from None
        if args.action == 'restart' and not args.interrupt:
            raise RuntimeError('Cannot determine queue state. Use restart --interrupt to explicitly allow interruption, or restart the Pod.') from None
        # ZIP is allowed while the backend is stopped; per-file consistency is checked.
    if args.action == 'zip':
        zip_outputs(root)
        return
    if args.action == 'restart':
        if not supervisor_running(root):
            raise RuntimeError('The H3 Image supervisor is not running. Restart the Pod; do not launch a duplicate backend.')
        run = root / 'run'
        run.mkdir(parents=True, exist_ok=True)
        temporary = run / ('restart.%s.tmp' % os.getpid())
        temporary.write_text(str(time.time()))
        temporary.replace(run / 'restart.request')
        print('Restart requested from the supervisor. Saved images/models are untouched; queue entries will be cleared.')
        return
    if args.interrupt:
        request('/queue', {'clear': True})
        request('/interrupt', {})
        print('Cancellation requested. Memory release will be handled by the running backend.')
    request('/free', {'unload_models': True, 'free_memory': True})
    print('Memory release requested inside ComfyUI. It does not delete files or stop GPU billing.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('STOPPED:', str(e), file=sys.stderr)
        sys.exit(1)
