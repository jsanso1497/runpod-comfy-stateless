#!/usr/bin/env python3
"""Do not expose ComfyUI until the owned daemon has all required prompt models."""
import argparse
import json
import os
from pathlib import Path
import time
from node import ollama_client as oc
from node.logic import settings


def wait_ready(pid, root=Path('/workspace/h3-portrait')):
    cfg = settings()
    end = time.monotonic() + 120 + cfg['pull_timeout_seconds']*len(set((cfg['analysis_model'], cfg['ollama_model'])))
    print('H3 STARTUP: waiting for prompt-model downloads. File browser remains available on 8888.', flush=True)
    while time.monotonic() < end:
        os.kill(pid, 0)  # Stop startup if the supervised service exited.
        proc_stat=Path(f'/proc/{pid}/stat')
        if proc_stat.is_file() and proc_stat.read_text().rsplit(')',1)[1].strip().split()[0]=='Z':
            raise RuntimeError('Ollama supervisor exited before readiness; inspect the startup log.')
        path = root/'ollama-ready.json'
        if path.is_file():
            report = json.loads(path.read_text())
            if report.get('pid') == pid and report.get('version') == '1.5.3':
                with oc.session() as s:
                    oc.pipeline_info(s, cfg)
                print('H3 STARTUP GATE PASSED: Ollama model files and capabilities verified.', flush=True)
                return
        status = root/'ollama-status.json'
        if status.is_file():
            report = json.loads(status.read_text())
            if report.get('pid') == pid and report.get('phase') == 'failed':
                raise RuntimeError(report.get('error', 'Ollama startup failed.'))
        time.sleep(1)
    raise RuntimeError('Prompt-model startup exceeded its time budget. See ollama-status.json; partial weights were retained.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--pid', type=int, required=True)
    wait_ready(p.parse_args().pid)
