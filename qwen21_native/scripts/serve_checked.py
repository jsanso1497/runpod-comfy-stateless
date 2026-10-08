#!/usr/bin/env python3
"""Supervise ComfyUI and validate node schemas on the actual runtime, not BuildKit."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
from validate_workflows import selected_catalog,check_schema,check_graph,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=['identity','upscale'],required=True)
    p.add_argument('--include-bfs',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
    cmd=args.command[1:] if args.command[:1]==['--'] else args.command
    if not cmd:p.error('Missing ComfyUI server command')
    child=subprocess.Popen(cmd)
    def terminate(signum,frame):
        if child.poll() is None:child.send_signal(signum)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        deadline=time.monotonic()+240;info=None
        url=f"http://127.0.0.1:{os.environ.get('COMFY_PORT','8188')}/object_info"
        while time.monotonic()<deadline:
            if child.poll() is not None:raise RuntimeError(f'ComfyUI exited during startup with code {child.returncode}')
            try:
                with urllib.request.urlopen(url,timeout=10) as response:info=json.load(response)
                break
            except (OSError,ValueError):time.sleep(2)
        if info is None:raise RuntimeError('ComfyUI did not provide node schemas within 240 seconds; inspect startup log above.')
        rows=selected_catalog(args.profile,args.include_bfs)
        for row in rows:
            path=ROOT/'workflows'/row['file'];ui=json.loads(path.read_text());api=json.loads((ROOT/'api_workflows'/(path.stem+'.api.json')).read_text())
            try:check_graph(ui,api);check_schema(ui,api,info)
            except Exception as exc:raise RuntimeError(f"Runtime schema validation failed for {row['file']}: {exc}") from exc
        logs=Path(os.environ.get('DATA_ROOT','/workspace/qwen21_native'))/'logs';logs.mkdir(parents=True,exist_ok=True)
        # TQ21_ADDON_SCHEMA_CHECK: real runtime only, never a BuildKit GPU probe.
        torso_validator=ROOT/'addons/torso_lock/scripts/validate_workflows.py'
        if not torso_validator.is_file():
            raise RuntimeError('Torso addon validator is missing from the image.')
        result=subprocess.run([sys.executable,str(torso_validator),'--server',url.rsplit('/object_info',1)[0]],
            capture_output=True,text=True,timeout=150)
        (logs/'torso_runtime_schema_check.txt').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError('Torso runtime schema check failed: '+result.stdout+result.stderr)
        print('TORSO LOCK READY: 4 addon workflow schemas passed; no inference performed.',flush=True)
        report={'status':'PASS','profile':args.profile,'workflows':len(rows),'note':'Node schemas checked; no GPU inference performed by this check.'}
        (logs/'runtime_schema_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f"NATIVE SUITE READY: {len(rows)} workflow schemas passed on the runtime host. No generation quality claim.",flush=True)
        return child.wait()
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':
    try:sys.exit(main())
    except (RuntimeError,OSError) as exc:print('STARTUP FAILED: '+str(exc),file=sys.stderr);sys.exit(1)
