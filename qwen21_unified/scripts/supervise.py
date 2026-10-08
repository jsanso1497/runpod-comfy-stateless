#!/usr/bin/env python3
"""Start original Photo entrypoint and verify Native, Torso, and SAM nodes live."""
from __future__ import annotations
import json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path

BASE=Path('/opt/qwen21_native')
OUTPUT=Path('/workspace/qwen21_unified/logs')
REQUIRED={'TextEncodeQwenImage21','QwenImage21Cache','SAM3_Detect',
          'AUSBOSS_NODES_CropForInpaint','AUSBOSS_NODES_StitchInpaint'}

def main():
    args=sys.argv[1:]
    cmd=args[1:] if args[:1]==['--'] else args
    if not cmd:raise RuntimeError('Missing original Photo server command')
    if Path(cmd[0]).name=='start.sh':cmd=['bash',*cmd]
    child=subprocess.Popen(cmd)
    def forward(sig,_):
        if child.poll() is None:child.send_signal(sig)
    signal.signal(signal.SIGTERM,forward);signal.signal(signal.SIGINT,forward)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    try:
        # Model weights are downloaded by the original Photo startup before the HTTP server starts.
        deadline=time.monotonic()+5400
        port=os.getenv('COMFY_PORT','8188')
        url=f'http://127.0.0.1:{port}'
        info=None
        while time.monotonic()<deadline:
            if child.poll() is not None:raise RuntimeError(f'Original Photo server exited before readiness: {child.returncode}')
            try:
                with urllib.request.urlopen(url+'/object_info',timeout=12) as r:info=json.load(r)
                if isinstance(info,dict):break
            except (OSError,ValueError):pass
            time.sleep(5)
        if not isinstance(info,dict):raise RuntimeError('Photo server /object_info did not become available; check model downloads and GPU startup logs.')
        missing=sorted(REQUIRED-set(info))
        if missing:raise RuntimeError('Missing required unified runtime node schemas: '+', '.join(missing))
        # Photo may supply --user-directory differing from both default locations.
        # Discover actual server command line and ensure all 40 UI files land there.
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                argv=[x for x in (proc/'cmdline').read_bytes().decode().strip('\0').split('\0') if x]
                if not any(x=='main.py' or x.endswith('/main.py') for x in argv):continue
                cwd=(proc/'cwd').resolve()
                name=next(x for x in argv if x=='main.py' or x.endswith('/main.py'))
                comfy=(cwd/Path(name)).resolve().parent
                if not (comfy/'comfy_extras/nodes_qwen.py').is_file():continue
                user=comfy/'user'
                for i,arg in enumerate(argv):
                    if arg=='--user-directory' and i+1<len(argv):user=Path(argv[i+1]);break
                    if arg.startswith('--user-directory='):user=Path(arg.split('=',1)[1]);break
                if not user.is_absolute():user=(cwd/user).resolve()
                prepared=subprocess.run([sys.executable,'/opt/qwen21_unified/scripts/runtime_prepare.py',
                    '--comfy-dir',str(comfy),'--user-dir',str(user)],capture_output=True,text=True,timeout=180)
                (OUTPUT/'active_user_dir_check.log').write_text(prepared.stdout+prepared.stderr)
                if prepared.returncode:raise RuntimeError('Cannot install workflows into actual ComfyUI user directory: '+prepared.stderr)
                print('ACTIVE COMFYUI USER DIR:',str(user),flush=True)
                break
            except (OSError,UnicodeError):continue
        else:
            raise RuntimeError('Could not locate the actual ComfyUI main.py process to verify where user workflows are installed.')
        native=subprocess.run([sys.executable,str(BASE/'scripts/validate_workflows.py'),'--profile','identity','--server',url],capture_output=True,text=True,timeout=180)
        torso=subprocess.run([sys.executable,str(BASE/'addons/torso_lock/scripts/validate_workflows.py'),'--server',url],capture_output=True,text=True,timeout=180)
        photo=subprocess.run([sys.executable,'/opt/qwen21_photo_edit/preflight.py','--comfy-home','/opt/ComfyUI','--sam','--server',url],capture_output=True,text=True,timeout=180)
        for title,result in [('native',native),('torso',torso),('photo',photo)]:
            (OUTPUT/f'{title}_schema_check.log').write_text(result.stdout+'\n'+result.stderr)
            if result.returncode:raise RuntimeError(title+' runtime schema validation failed. '+(result.stdout+result.stderr)[-2500:])
        report={'status':'PASS','photo':'Qwen 2.1 2K / SAM 3.1 BF16 + INT8 graphs',
                'native_graphs':34,'torso_graphs':4,'photo_graphs':2,
                'note':'Live node schemas verified. This does not prove GPU image generation or protect landmarks inside generated regions.'}
        (OUTPUT/'runtime_schema_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print('QWEN UNIFIED READY: Photo 2K/SAM3 + Native(34) + Torso Lock(4) verified in live ComfyUI.',flush=True)
        return child.wait()
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=25)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as ex:print('QWEN UNIFIED STARTUP CHECK FAILED: '+str(ex),file=sys.stderr);sys.exit(1)
