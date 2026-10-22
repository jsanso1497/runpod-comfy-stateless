"""Managed ComfyUI + optional local Ollama; only authenticated 8188 is public."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from aiohttp import web, ClientSession, ClientTimeout
from .config import ROOT, RELEASE, atomic_json, boolean, data_root, file_sha, public_child_env, upload_mb, variants
from .gateway import make_app


def sync_workflows(root):
    folder=root/"comfy-user-h3-swap/default/workflows"/("H3_Character_Replacement_"+RELEASE)
    folder.mkdir(parents=True,exist_ok=True)
    for src in (ROOT/"workflows").glob("*.json"):
        if src.name.endswith(".api.json"): continue
        dest=folder/src.name
        if dest.exists() and dest.read_bytes()!=src.read_bytes():
            dest=folder/(src.stem+"_factory_"+file_sha(src)[:10]+src.suffix)
        if not dest.exists(): shutil.copy2(src,dest)
    return folder


def configure():
    root=data_root();root.mkdir(parents=True,exist_ok=True)
    for name in ("input","output","models","h3-swap","ollama","comfy-user-h3-swap"):
        (root/name).mkdir(parents=True,exist_ok=True)
    comfy=Path("/opt/ComfyUI")
    for name in ("diffusion_models","text_encoders","vae","loras","checkpoints"):
        dest=root/"models"/name;dest.mkdir(parents=True,exist_ok=True)
        target=comfy/"models"/name
        if target.is_symlink():
            if target.resolve()!=dest.resolve(): raise ValueError("Unexpected model symlink")
        elif target.exists():
            if any(p.is_file() and not p.name.startswith("put_") for p in target.iterdir()):
                raise ValueError("Image contains unexpected model files")
            shutil.rmtree(target);target.symlink_to(dest,target_is_directory=True)
        else: target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(dest,target_is_directory=True)
    sync_workflows(root)
    env=public_child_env()
    env.update({"H3_SWAP_ROOT":str(ROOT),"PYTHONPATH":str(ROOT),"OLLAMA_HOST":"127.0.0.1:11434", "OLLAMA_MODELS":str(root/"ollama"),"OLLAMA_CONTEXT_LENGTH":"16384","OLLAMA_NUM_PARALLEL":"1","OLLAMA_MAX_LOADED_MODELS":"1","OLLAMA_KEEP_ALIVE":"0","OLLAMA_NO_CLOUD":"1"})
    return env


async def prepare_ollama(env,status):
    async with ClientSession(timeout=ClientTimeout(total=20),trust_env=False) as session:
        for _ in range(120):
            try:
                async with session.get("http://127.0.0.1:11434/api/tags") as r:
                    if r.status==200: break
            except Exception: pass
            await asyncio.sleep(1)
        else: raise ValueError("Private Ollama did not start")
        model=env.get("H3_SWAP_PROMPT_MODEL","qwen3-vl:8b")
        if model.endswith("-cloud"): raise ValueError("Cloud prompt models are not allowed")
        status.update(stage="prompt model",asset=model)
        print("[H3 SWAP PREP] Preparing local vision helper "+model,flush=True)
        success=False
        async with session.post("http://127.0.0.1:11434/api/pull",json={"model":model,"stream":True},timeout=ClientTimeout(total=14400,sock_read=180)) as r:
            r.raise_for_status()
            async for line in r.content:
                if not line.strip(): continue
                row=json.loads(line)
                if row.get("error"): raise ValueError("Local vision-helper preparation failed")
                if row.get("status")=="success": success=True
                status["prompt_download_bytes"]=row.get("completed",0)
                status["prompt_download_total"]=row.get("total",0)
        if not success: raise ValueError("Local vision-helper pull ended without success")
        async with session.get("http://127.0.0.1:11434/api/tags") as r:
            r.raise_for_status();tags=await r.json()
        record=next((x for x in tags.get("models",[]) if x.get("name")==model),None)
        atomic_json(data_root()/"h3-swap/prompt-model.json",{"model":model,"record":record,"context":16384,"downloaded_at":time.time()})


async def main_async():
    if os.environ.get("WB_AUTH_MODE","basic")!="basic":
        raise ValueError("This image requires WB_AUTH_MODE=basic; only port 8188 should be exposed")
    selected=variants();env=configure();root=data_root()
    status={"ready":False,"stage":"starting","release":RELEASE,"variants":sorted(selected),"gpu_validation":"not certified"}
    app=make_app(os.environ.get("WB_PASSWORD",""),status)
    runner=web.AppRunner(app);await runner.setup();await web.TCPSite(runner,"0.0.0.0",8188).start()
    logdir=root/"h3-swap/logs";logdir.mkdir(exist_ok=True)
    logs=[];children=[]
    stop=asyncio.Event()
    loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,stop.set)
    try:
        comfy_log=(logdir/"comfy.log").open("a",buffering=1);logs.append(comfy_log)
        args=[sys.executable,"/opt/ComfyUI/main.py","--listen","127.0.0.1","--port","8189","--input-directory",str(root/"input"),"--output-directory",str(root/"output"),"--user-directory",str(root/"comfy-user-h3-swap"),"--disable-auto-launch","--disable-metadata","--max-upload-size",str(upload_mb())]
        # Deliberately retain native normal offloading and the qualified kernels.
        comfy=subprocess.Popen(args,cwd="/opt/ComfyUI",env=env,stdout=comfy_log,stderr=subprocess.STDOUT);children.append(comfy)
        use_assistant=boolean("H3_SWAP_PROMPT_ASSIST",True)
        if use_assistant:
            ollama_log=(logdir/"ollama.log").open("a",buffering=1);logs.append(ollama_log)
            children.append(subprocess.Popen(["ollama","serve"],env=env,stdout=ollama_log,stderr=subprocess.STDOUT))
        def update(**kwargs):
            status.update(kwargs);atomic_json(root/"h3-swap/status.json",dict(status))
        from .assets import prepare
        result=await asyncio.to_thread(prepare,selected,update)
        if use_assistant: await prepare_ollama(env,status)
        # A real source/schema smoke check runs on the Pod after models are visible.
        from .check import live_report
        for _ in range(180):
            if comfy.poll() is not None: raise ValueError("ComfyUI exited; inspect /workspace/h3-swap/logs/comfy.log")
            try:
                report=await asyncio.to_thread(live_report,selected)
                if report["ok"]: break
            except Exception: pass
            await asyncio.sleep(1)
        else: raise ValueError("Required native/custom nodes or prepared models are missing; inspect logs and run python -m swapkit.check --live")
        update(ready=True,stage="ready",asset=None,preparation=result)
        print("[H3 SWAP READY] Open HTTP 8188. Username: workbench. Workflows sidebar: H3_Character_Replacement_"+RELEASE,flush=True)
        while not stop.is_set():
            if any(p.poll() is not None for p in children):
                raise ValueError("A managed service exited; inspect /workspace/h3-swap/logs")
            try: await asyncio.wait_for(stop.wait(),timeout=5)
            except asyncio.TimeoutError: pass
    except Exception as exc:
        status.update(ready=False,stage="error",error=str(exc));atomic_json(root/"h3-swap/status.json",status)
        print("[H3 SWAP ERROR] "+str(exc),flush=True)
        # Keep the authenticated status page available for diagnosis; no auto-restart loop.
        await stop.wait()
    finally:
        for proc in children:
            if proc.poll() is None: proc.terminate()
        for proc in children:
            try: await asyncio.to_thread(proc.wait,timeout=20)
            except subprocess.TimeoutExpired:proc.kill()
        for handle in logs:handle.close()
        await runner.cleanup()


def main():
    asyncio.run(main_async())

if __name__=="__main__":main()
