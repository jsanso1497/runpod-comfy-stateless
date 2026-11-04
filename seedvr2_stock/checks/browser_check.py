"""Build gate: actual Chromium against native ComfyUI through standard Nginx.

The delivered graph is copied with only the three GPU SeedVR2 nodes removed.
This tests the native UI, crop editor serialization, comparison, logs and export;
it is explicitly NOT a model/quality test or a RunPod public proxy test.
"""
import asyncio, copy, hashlib, json, os
from pathlib import Path
from urllib.parse import urlencode
from playwright.async_api import async_playwright
ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8188'
REPORT=ROOT/'reports'


async def frontend_state(page):
    """Read-only diagnostics. Never import a frontend shim to probe readiness."""
    return await page.evaluate("""() => {
        const app=window.comfyAPI?.app?.app;
        const api=window.comfyAPI?.api?.api;
        const overlay=document.querySelector('[data-testid="app-loading-overlay"]');
        return {
            url:location.href, title:document.title,
            hasWindowApp:!!window.app,
            hasExtensionManager:!!window.app?.extensionManager,
            hasExposedApp:!!app, hasGraph:!!app?.graph,
            canLoadGraph:typeof app?.loadGraphData==='function',
            canSerializeGraph:typeof app?.graphToPrompt==='function',
            hasApi:!!api, hasClientId:!!api?.clientId,
            loadingOverlayPresent:!!overlay,
            loadingOverlayVisible:!!overlay && !!(overlay.offsetWidth ||
                overlay.offsetHeight || overlay.getClientRects().length)
        };
    }""")


async def wait_for_frontend(page, timeout=120000):
    """Wait for the real app, not DOMContentLoaded or a guessed sleep.

    The pinned frontend's /scripts/app.js is a compatibility shim reading
    window.comfyAPI.app at module evaluation. Importing it too early throws
    and caches a failed module evaluation for the rest of that page session.
    Follow upstream browser readiness: extensionManager, then hidden loading
    overlay. Also require the exposed graph methods and real WebSocket clientId
    before this check serializes a workflow or subscribes to logs.
    """
    try:
        await page.wait_for_function(
            "() => !!window.app?.extensionManager", timeout=timeout)
        overlay=page.get_by_test_id('app-loading-overlay')
        await overlay.wait_for(state='attached', timeout=timeout)
        await overlay.wait_for(state='hidden', timeout=timeout)
        await page.wait_for_function("""() => {
            const app=window.comfyAPI?.app?.app;
            const api=window.comfyAPI?.api?.api;
            return !!(app?.graph &&
                typeof app.loadGraphData==='function' &&
                typeof app.graphToPrompt==='function' &&
                typeof api?.clientId==='string' && api.clientId.length>0);
        }""", timeout=timeout)
        await page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => resolve(true)))")
    except Exception as error:
        try:
            state=await frontend_state(page)
        except Exception as diagnostic_error:
            state={'diagnostic_error':str(diagnostic_error)}
        raise RuntimeError('ComfyUI frontend did not become ready: '+json.dumps(state)) from error

def cpu_graph(original,filename):
    g=copy.deepcopy(original)
    g['nodes']=[n for n in g['nodes'] if n['id'] not in (3,4,5)]
    nodes={n['id']:n for n in g['nodes']}
    g['links']=[l for l in g['links'] if l[1] not in (3,4,5) and l[3] not in (3,4,5)]
    # Replace only the model output with the decoded crop, explicitly for CI.
    for dst,name in [(6,'images'),(12,'image')]:
        slot=next(i for i,x in enumerate(nodes[dst]['inputs']) if x['name']==name)
        lid=max([l[0] for l in g['links']]+[0])+1
        g['links'].append([lid,2,0,dst,slot,'IMAGE'])
    for n in nodes.values():
        for inp in n['inputs']:inp['link']=None
        for out in n['outputs']:out['links']=[]
    for lid,src,slot,dst,islot,kind in g['links']:
        nodes[src]['outputs'][slot]['links'].append(lid);nodes[dst]['inputs'][islot]['link']=lid
    # Test-only image output forces native Autogrow/math serialization and execution.
    size_node={'id':90,'type':'EmptyImage','pos':[900,1200],'size':[280,160],'flags':{},'order':90,'mode':0,
      'inputs':[{'name':'width','type':'INT','link':None,'widget':{'name':'width'}}],
      'outputs':[{'name':'IMAGE','type':'IMAGE','links':[]}],
      'properties':{'Node name for S&R':'EmptyImage'},'widgets_values':[512,2,1,0]}
    saver={'id':91,'type':'SaveImage','pos':[1250,1200],'size':[280,200],'flags':{},'order':91,'mode':0,
      'inputs':[{'name':'images','type':'IMAGE','link':None}], 'outputs':[],
      'properties':{'Node name for S&R':'SaveImage'},'widgets_values':['SeedVR2/BROWSER_SIZE']}
    nodes[90]=size_node;nodes[91]=saver;g['nodes'] += [size_node,saver]
    for src,slot,dst,islot,kind in [(10,1,90,0,'INT'),(90,0,91,0,'IMAGE')]:
        lid=max(l[0] for l in g['links'])+1;g['links'].append([lid,src,slot,dst,islot,kind])
        nodes[src]['outputs'][slot]['links'].append(lid);nodes[dst]['inputs'][islot]['link']=lid
    g['last_node_id']=91
    g['last_link_id']=max(l[0] for l in g['links'])
    nodes[1]['widgets_values']=[filename]
    nodes[8]['widgets_values']=[{'crop':{'x':120,'y':0,'width':72,'height':108}}]
    nodes[7]['widgets_values']=['SeedVR2/BROWSER_TEST','mp4']
    nodes[14]['widgets_values']=['SeedVR2/BROWSER_COMPARE']
    return g

async def main():
    REPORT.mkdir(exist_ok=True)
    report=json.loads((REPORT/'native-report.json').read_text())
    sample=next(x for x in report['cases'] if x['name']=='full')
    graph=cpu_graph(json.loads((ROOT/'workflows/01_Video_Clarity.json').read_text()),sample['uploaded'])
    sockets=[];errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch()
        ctx=await browser.new_context(http_credentials={'username':'workbench','password':os.environ['COMFY_PASSWORD']},
            accept_downloads=True,viewport={'width':1700,'height':1050})
        page=await ctx.new_page()
        def on_socket(ws):
            row={'url':ws.url,'closed':False,'messages':0};sockets.append(row)
            ws.on('close',lambda:row.update(closed=True))
            ws.on('framereceived',lambda _:row.update(messages=row['messages']+1))
        page.on('websocket',on_socket);page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            await page.goto(BASE,wait_until='domcontentloaded',timeout=120000)
            await wait_for_frontend(page)
            print('PASS: frontend initialized with a WebSocket clientId.',flush=True)
            encoded=await page.evaluate("""async graph => {
                const app=window.comfyAPI.app.app;
                await app.loadGraphData(graph); return (await app.graphToPrompt()).output;
            }""",graph)
            (REPORT/'browser-serialized-prompt.json').write_text(json.dumps(encoded,indent=2)+'\n')
            assert encoded['8']['inputs']['crop']['crop']=={'x':120,'y':0,'width':72,'height':108},encoded['8']
            assert encoded['6']['inputs']['fps']==['2',2] and encoded['6']['inputs']['audio']==['2',1]
            assert encoded['13']['inputs']['direction']=='right' and not encoded['13']['inputs']['match_image_size']
            assert encoded['7']['inputs']['format']=='mp4',encoded['7']
            assert encoded['10']['inputs']['expression']=='min(a,b)'
            assert encoded['10']['inputs']['values.a']==['9',0] and encoded['10']['inputs']['values.b']==['9',1]
            response=await page.evaluate("""async prompt => {
                const api=window.comfyAPI.api.api;
                if (!api.clientId) throw new Error('WebSocket clientId is missing');
                const r=await fetch('/internal/logs/subscribe',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({clientId:api.clientId,enabled:true})});
                if (!r.ok) throw new Error('Native log subscription failed: '+r.status);
                const q=await fetch('/prompt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,client_id:api.clientId})});
                const data=await q.json(); if(!q.ok) throw new Error(JSON.stringify(data));return data;
            }""",encoded)
            pid=response['prompt_id'];item=None
            for _ in range(180):
                r=await ctx.request.get(BASE+'/history/'+pid);history=await r.json()
                if pid in history:item=history[pid];break
                await asyncio.sleep(.5)
            assert item and item['status']['status_str']=='success',item
            await asyncio.sleep(15)
            active=[s for s in sockets if '/ws' in s['url'] and not s['closed'] and s['messages']>0]
            assert active and len([s for s in sockets if '/ws' in s['url']])<=2,sockets
            # Download a real native exported MP4 in a browser, not a terminal client.
            url=BASE+'/view?'+urlencode(sample['download'])
            async with page.expect_download(timeout=30000) as event:
                await page.evaluate("""url => {const a=document.createElement('a');a.href=url;a.download='seedvr2-native-test.mp4';document.body.appendChild(a);a.click();a.remove();}""",url)
            download=await event.value;dest=REPORT/'browser-download.mp4';await download.save_as(dest)
            response=await ctx.request.get(url)
            assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(await response.body()).digest()
            assert dest.stat().st_size>1000
            await page.screenshot(path=str(REPORT/'native-comfyui.png'),full_page=True)
            (REPORT/'browser-report.json').write_text(json.dumps({'real_chromium':True,'nginx_auth':True,
                'crop_serialized':True,'native_cpu_execution':True,'stable_browser_websocket':True,
                'logs_subscribed':True,'downloaded_mp4':True,'gpu_tested':False,'page_errors':errors,'sockets':sockets},indent=2)+'\n')
            print('PASS: real browser crop, native CPU execution, logs, sustained websocket and MP4 download through Nginx.')
        except BaseException as error:
            details={'error':str(error),'errors':errors,'sockets':sockets}
            try:
                details['frontend']=await frontend_state(page)
            except Exception as diagnostic_error:
                details['frontend_probe_error']=str(diagnostic_error)
            try:
                await page.screenshot(path=str(REPORT/'browser-failure.png'),full_page=True)
            except Exception as screenshot_error:
                details['screenshot_error']=str(screenshot_error)
            (REPORT/'browser-errors.json').write_text(json.dumps(details,indent=2)+'\n')
            raise
        finally:await browser.close()

if __name__=='__main__':asyncio.run(main())
