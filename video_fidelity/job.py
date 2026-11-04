"""CPU supervisor: prepare, run GPU child, wait for exit, then encode."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback
from fv_common import ROOT, MODEL, VAE, atomic_json, child_env, parent_death_signal, terminate
from media import make_plan, prepare, detect_cuts, finish


def run(job: Path) -> None:
    request=json.loads((job/'request.json').read_text())
    state={'state':'running','phase':'preflight','pid':os.getpid()}
    def record(**fields):
        state.update(fields)
        atomic_json(job/'status.json',state)
    record()
    source=Path(request['source'])
    plan=make_plan(source,request)
    # Worst-case RGB16 PNGs + source RGB intermediate + encoding reserve.
    required=int(plan['count']*(plan['output_size'][0]*plan['output_size'][1]*6+
                               plan['source_size'][0]*plan['source_size'][1]*4)*1.2+3*1024**3)
    if shutil.disk_usage(job).free < required:
        raise OSError(f'Need approximately {required/1024**3:.1f} GiB free for this trim. Shorten the trim or provide storage.')
    if request['strength']:
        for filename in (MODEL,VAE):
            if not (Path(request['model_dir'])/filename).is_file():
                raise FileNotFoundError('Missing verified model: '+filename)
    atomic_json(job/'plan.json',plan)
    record(phase='source_preparation')
    prepared=prepare(source,job,plan)
    record(phase='scene_detection')
    plan['cuts']=detect_cuts(prepared,job,plan['count'])
    atomic_json(job/'plan.json',plan)
    record(phase='inference',frames=plan['count'],cuts=plan['cuts'])
    env=child_env()
    env['FAITHFUL_PARENT_PID']=str(os.getpid())
    with (job/'inference.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py'),'--job',str(job)],
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               cwd=ROOT,env=env)
        try:
            rc=child.wait()
        finally:
            terminate(child)
    worker=json.loads((job/'worker_status.json').read_text()) if (job/'worker_status.json').is_file() else {}
    if rc or worker.get('state')!='complete':
        detail=worker.get('error',f'Worker exited {rc}; inspect inference.log and system RAM.')
        raise RuntimeError(detail+' No automatic precision, resolution or temporal-quality fallback was applied.')
    record(phase='encode',gpu_worker_exited=True)
    output=Path(request['output'])
    finish(job,output,plan)
    atomic_json(output.parent/'settings.json',{'request':request,'plan':plan,
                 'quality_status':'GPU visual acceptance is user-reviewed, not certified by CPU tests.'})
    if not request['keep_frames']:
        shutil.rmtree(job/'frames')
        prepared.unlink()
    record(state='complete',phase='complete',output=str(output),frames=plan['count'])


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    parent_death_signal(int(os.environ.get('FAITHFUL_PARENT_PID',os.getppid())))
    def stop(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    try:
        run(args.job)
        return 0
    except BaseException as error:
        status=args.job/'status.json'
        old=json.loads(status.read_text()) if status.exists() else {}
        old.update(state='cancelled' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed',
                   error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        atomic_json(status,old)
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
