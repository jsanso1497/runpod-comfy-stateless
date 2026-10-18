#!/usr/bin/env python3
"""Read-only Workbench diagnostic; never prints credentials or private URLs."""
from pathlib import Path
import base64
import json
import os
import shutil
import subprocess
from urllib.request import Request, build_opener, ProxyHandler


def main():
    print('WORKBENCH_DIAGNOSTIC_STARTED', flush=True)
    root = Path(os.getenv('WB_DATA_ROOT', '/workspace'))
    result = {'tasks': os.getenv('WB_TASKS', '(defaults)'), 'data_root': str(root)}
    try:
        disk = shutil.disk_usage(root)
        result['disk_free_gb'] = round(disk.free/1e9, 2)
    except OSError:
        result['disk'] = 'unavailable'
    for name in ('memory.max', 'memory.current', 'memory.events'):
        p = Path('/sys/fs/cgroup')/name
        if p.is_file():
            result['cgroup_'+name] = p.read_text().strip()
    try:
        result['mount'] = subprocess.check_output(
            ['findmnt','-T',str(root),'-n','-o','TARGET,FSTYPE'], text=True, timeout=5).strip()
        if result['mount'].split()[0] == '/':
            result['storage_warning'] = 'Data root is on the container filesystem. Back up before stop/recreate; do not assume volume persistence.'
    except (OSError, subprocess.SubprocessError, IndexError):
        result['mount'] = 'not verified'
    try:
        result['nvidia'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
            '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        result['nvidia'] = 'not available'
    headers = {}
    if os.getenv('WB_AUTH_MODE','basic') == 'basic':
        password = os.getenv('WB_PASSWORD','')
        headers['Authorization'] = 'Basic '+base64.b64encode(('workbench:'+password).encode()).decode()
    try:
        req = Request('http://127.0.0.1:8188/_workbench/status', headers=headers)
        with build_opener(ProxyHandler({})).open(req, timeout=20) as response:
            data = json.load(response)
        result['preparation'] = data.get('preparation')
        result['comfy_running'] = data.get('comfy_running')
        result['important_tasks'] = [t for t in data.get('tasks',[]) if t.get('id') in ('QKEEP1','QMOSS','QMOSS_I2I','QNATIVE','QOVERLAY')]
    except Exception as exc:
        result['status_error'] = type(exc).__name__
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
