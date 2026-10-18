"""One explicit ComfyUI installation, authenticated gateway, task-specific preparation."""
from __future__ import annotations
import asyncio, base64, contextlib, hashlib, hmac, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
from urllib.parse import urlsplit
from aiohttp import web, ClientSession, ClientTimeout, WSMsgType
from filelock import FileLock, Timeout as LockTimeout
from .config import (ROOT, atomic_json, child_environment, data_root, model_config, model_root,
 read_catalog, requested_groups, resolve_secret_aliases, safe_path, selected_tasks, state_root, workspace)
from .assets import prepare_private, prepare_standard
from . import http_gateway


def authorized(header: str, password: str) -> bool:
    try:
        scheme,value=header.split(' ',1)
        if scheme.lower()!='basic':return False
        user,secret=base64.b64decode(value,validate=True).decode().split(':',1)
        return hmac.compare_digest(user,'workbench') and hmac.compare_digest(secret.encode(),password.encode())
    except (ValueError,UnicodeError):return False

@web.middleware
async def security(request,handler):
    mode = request.app.get(http_gateway.AUTH_MODE_KEY, request.app.get('auth_mode', 'basic'))
    if request.path == '/healthz':
        return web.json_response({'gateway': 'running', 'workspace': workspace(),
                                  'auth_mode': mode, 'access_version': http_gateway.ACCESS_VERSION})
    if request.path == '/readyz':
        return await handler(request)
    if mode != 'none' and not authorized(request.headers.get('Authorization',''), request.app.get(http_gateway.PASSWORD_KEY, request.app.get('password', ''))):
        return web.Response(status=401, text='Workbench authentication required.',
            headers={'WWW-Authenticate': 'Basic realm="Comfy Workbench", charset="UTF-8"',
                     'Cache-Control': 'no-store'})
    if not http_gateway.same_origin(request):
        return web.Response(status=403, text='Cross-origin writes are not allowed.')
    try:
        return await handler(request)
    except (ValueError, KeyError, FileNotFoundError):
        return web.json_response({'error': 'Invalid request or unavailable local resource.'}, status=400)


def configure_files()->dict[str,str]:
    root=data_root();state=state_root();root.mkdir(parents=True,exist_ok=True)
    for name in ('input','output','comfy-user','models','workbench','ollama'):(root/name).mkdir(parents=True,exist_ok=True)
    # Keep one immutable code tree; only user data and assets live on mounted storage.
    comfy=Path('/opt/ComfyUI')
    for category in ('checkpoints','diffusion_models','text_encoders','vae','loras','SEEDVR2','clip_vision','upscale_models','refmods'):
        dest=root/'models'/category;dest.mkdir(parents=True,exist_ok=True)
        target=comfy/'models'/category
        if target.is_symlink():
            if target.resolve()!=dest.resolve():raise ValueError('Unexpected model directory link.')
        elif target.exists():
            if any(p.is_file() and not p.name.startswith('put_') for p in target.iterdir()):
                raise ValueError('Image unexpectedly contains model weights; refusing to replace them.')
            shutil.rmtree(target);target.symlink_to(dest,target_is_directory=True)
        else:target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(dest,target_is_directory=True)
    config=root/'config';config.mkdir(exist_ok=True)
    for source,target in [(ROOT/'config/ollama.json',config/'ollama.json'),(ROOT/'config/h3/settings.json',config/'h3/settings.json')]:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source,target)
    # Existing settings never silently downgrade precision or retain a Lite configuration.
    settings=json.loads((config/'h3/settings.json').read_text())
    if workspace()=='h3' and (settings.get('profile')!='full' or any('int8' in str(v).lower() for v in settings.get('model_files',{}).values())):
        raise ValueError('Persistent H3 configuration is not HQ. Move the old config aside and restart to create the HQ defaults.')
    env=child_environment();env.update({'H3_PORTRAIT_CONFIG':str(config/'h3'),'CONFIG_HOME':str(config),
       'OLLAMA_HOST':'127.0.0.1:11434','OLLAMA_MODELS':str(root/'ollama'),'OLLAMA_NUM_PARALLEL':'1','OLLAMA_MAX_LOADED_MODELS':'1',
       'HF_HUB_DISABLE_TELEMETRY':'1'})
    # Source helper policy paths are explicit instead of inherited from previous images.
    from .factory_sync import sync_factory
    factory_report = sync_factory(ROOT, root, read_catalog('tasks.json')['tasks'], workspace())
    atomic_json(state/'factory-workflows.json', factory_report)
    atomic_json(state/'workspace.json',{'workspace':workspace(),'version':'0.1.0','selected_tasks':selected_tasks(),'private_urls_in_status':False})
    return env

class Runtime:
    def __init__(self):
        self.comfy=None;self.ollama=None;self.job=None;self.job_state={'state':'idle'};self.session=None;self.env=None
    def start_comfy(self):
        from .runtime_recovery import guard_single_instance, launch_options
        guard_single_instance(self)
        self.env=configure_files()
        args=[sys.executable,'/opt/ComfyUI/main.py','--listen','127.0.0.1','--port','8189',
          '--input-directory',str(data_root()/'input'),'--output-directory',str(data_root()/'output'),
          '--user-directory',str(data_root()/'comfy-user'),'--disable-auto-launch']
        if os.environ.get('WB_SAVE_METADATA','0')!='1':args+=['--disable-metadata']
        args += launch_options(Path('/opt/ComfyUI'), workspace())
        # No arbitrary shell command string or forced reduced-precision flags.
        self.comfy=subprocess.Popen(args,cwd='/opt/ComfyUI',env=self.env)
    async def start_ollama(self):
        if self.ollama is None or self.ollama.poll() is not None:
            if not shutil.which('ollama'):raise ValueError('Ollama binary is missing from the image.')
            self.ollama=subprocess.Popen(['ollama','serve'],env=self.env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                async with self.session.get('http://127.0.0.1:11434/api/tags',timeout=ClientTimeout(total=3)) as r:
                    if r.status==200:return
            except (OSError,asyncio.TimeoutError):pass
            await asyncio.sleep(1)
        raise ValueError('Local prompt helper did not start.')
    async def prepare(self,groups:list[str],private:bool=False):
        from .audit_preparation import run_preparation
        await run_preparation(self, groups, private, globals())
    async def nodes(self):
        try:
            async with self.session.get('http://127.0.0.1:8189/object_info',timeout=ClientTimeout(total=10)) as r:
                if r.status==200:return await r.json()
        except Exception:pass
        return {}
    async def status(self,request):
        from .audit_preparation import preparation_view, _read_status
        schema=await self.nodes(); rows=[]
        asset_path=state_root()/'asset-status.json';asset=_read_status(asset_path)
        installed={r['id'] for r in asset.get('installed',[]) if (model_root()/r['destination']).is_file()}
        requirements=read_catalog('assets.json')['assets']
        ollama_path=state_root()/'ollama-status.json';helpers=_read_status(ollama_path).get('models',[])
        expected_helpers={name.casefold() for name in read_catalog('assets.json')['ollama']['models']}
        installed_helpers={m['name'].casefold() for m in helpers}
        for task in read_catalog('tasks.json')['tasks']:
            if workspace() not in task['workspaces']:continue
            needed=[a['id'] for a in requirements if a['group'] in task['requires']]
            missing_assets=[id for id in needed if id not in installed]
            if 'ollama' in task['requires']:
                missing_assets.extend('ollama:'+name for name in sorted(expected_helpers-installed_helpers))
            variants=[]
            for v in task['variants']:
                graph=json.loads((ROOT/v['file']).read_text())
                from .graph_support import required_node_types
                required_nodes=required_node_types(graph)|set(task.get('required_nodes',[]))
                missing_nodes=sorted(required_nodes-set(schema)-{'Note','MarkdownNote','Reroute','PrimitiveNode'}) if schema else ['ComfyUI is starting']
                variants.append({'file':v['file'],'missing_nodes':missing_nodes,'state':'ready' if not missing_nodes and not missing_assets else 'unavailable'})
            rows.append({'id':task['id'],'missing_assets':missing_assets,'variants':variants})
        return web.json_response({'workspace':workspace(),'comfy_running':bool(schema),'tasks':rows,'preparation':preparation_view(self),'startup_groups':requested_groups(),'gpu_validation':'not certified','release':read_catalog('release.json')})
    async def catalog(self,request):
        return web.json_response({'workspace':workspace(),'tasks':[t for t in read_catalog('tasks.json')['tasks'] if workspace() in t['workspaces']],'models':read_catalog('models.json')['models']})
    async def prepare_request(self,request):
        from .audit_preparation import request_preparation
        return await request_preparation(self, request, globals())
    async def graph(self,request):
        file=request.query.get('file','')
        allowed={v['file'] for t in read_catalog('tasks.json')['tasks'] if workspace() in t['workspaces'] for v in t['variants']}
        if file not in allowed:raise ValueError('Unknown workflow.')
        return web.FileResponse(safe_path(ROOT,file))
    async def files(self,request):
        kind=request.query.get('kind','output')
        if kind not in ('input','output'):raise ValueError('Only input and output media are exposed.')
        root=data_root()/kind
        if request.query.get('path'):
            p=safe_path(root,request.query['path'])
            if not p.is_file():raise FileNotFoundError()
            return web.FileResponse(p,headers={'Content-Disposition':'attachment'})
        rows=[]
        for p in root.rglob('*'):
            if len(rows)>=2000:break
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.mp4','.mov','.wav','.flac','.mp3'):
                rows.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size})
        return web.json_response({'kind':kind,'files':rows,'limit':2000})
    async def proxy(self,request):
        return await http_gateway.proxy(request, self.session)
    async def ready(self,request):
        ready = False
        try:
            async with self.session.get('http://127.0.0.1:8189/system_stats',
                                        timeout=ClientTimeout(total=5)) as response:
                ready = response.status == 200
        except Exception:
            pass
        return web.json_response({'gateway': 'running', 'comfy_ready': ready,
                                  'access_version': http_gateway.ACCESS_VERSION},
                                 status=200 if ready else 503)
    async def startup(self,app):
        self.session=http_gateway.session()
        self.start_comfy()
        self.job=asyncio.create_task(self.prepare(requested_groups(),True))
    async def cleanup(self,app):
        if self.job and not self.job.done():
            self.job.cancel()
            try:
                await self.job
            except asyncio.CancelledError:
                pass
        for proc in (self.comfy,self.ollama):
            if proc and proc.poll() is None:
                proc.terminate()
                try:await asyncio.to_thread(proc.wait,timeout=15)
                except subprocess.TimeoutExpired:proc.kill()
        if self.session:await self.session.close()

def make_app(password:str='',runtime:Runtime|None=None,auth_mode:str|None=None):
    mode = http_gateway.auth_mode(auth_mode)
    if mode == 'basic' and len(password)<16:
        raise ValueError('WB_PASSWORD must resolve to a secret of at least 16 characters. Username: workbench.')
    app=web.Application(middlewares=[security],client_max_size=2*1024**3,
                        handler_args={'auto_decompress': False})
    app[http_gateway.PASSWORD_KEY]=password if mode == 'basic' else ''
    app[http_gateway.AUTH_MODE_KEY]=mode
    app.on_response_prepare.append(http_gateway.on_prepare)
    r=runtime or Runtime()
    async def health(req):
        return web.json_response({'gateway': 'running'})
    app.router.add_get('/healthz',health)
    app.router.add_get('/readyz',r.ready)
    from .runtime_recovery import restart
    async def restart_comfy(req):return await restart(r,req)
    app.router.add_post('/_workbench/restart-comfy',restart_comfy)
    app.router.add_get('/_workbench/catalog',r.catalog);app.router.add_get('/_workbench/status',r.status)
    app.router.add_post('/_workbench/prepare',r.prepare_request);app.router.add_get('/_workbench/workflow',r.graph)
    app.router.add_get('/_workbench/files',r.files)
    async def index(req):return web.FileResponse(ROOT/'site/index.html')
    app.router.add_get('/_workbench/',index)
    if (ROOT/'site').exists():app.router.add_static('/_workbench/site/',ROOT/'site',show_index=False,follow_symlinks=False)
    app.router.add_route('*','/{path:.*}',r.proxy)
    app.on_startup.append(r.startup);app.on_cleanup.append(r.cleanup);return app

def main():
    mode = http_gateway.auth_mode()
    # A stale/removed RunPod password-secret binding must not block no-login mode.
    # This affects only the Workbench password, not HF/Civitai/private libraries.
    if mode == 'none':
        os.environ.pop('WB_PASSWORD', None)
        print('[WORKBENCH ACCESS 1.2.0] NO LOGIN on port 8188. Anyone with the URL can access this Pod.', flush=True)
    else:
        print('[WORKBENCH ACCESS 1.2.0] Password protection enabled on port 8188.', flush=True)
    resolve_secret_aliases()
    image_workspace=(ROOT/'BUILD_WORKSPACE').read_text().strip()
    if workspace()!=image_workspace:raise ValueError('Selected workspace does not match this image. Use the matching RunPod image.')
    selected_tasks();requested_groups()
    web.run_app(make_app(os.environ.get('WB_PASSWORD',''),auth_mode=mode),host='0.0.0.0',port=8188,access_log=None)

if __name__=='__main__':main()
