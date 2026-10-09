#!/usr/bin/env python3
"""Optional local Chromium smoke test against a simulated ComfyUI upstream.

Requires playwright plus Chromium. It does not contact RunPod or a model API.
This is not an end-to-end test of the real ComfyUI frontend or generation.
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from test_http_gateway import Harness, PASSWORD


async def run(executable=None):
    from playwright.async_api import async_playwright
    results=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True, executable_path=executable,
                                       args=['--no-sandbox','--disable-dev-shm-usage'])
        try:
            for mode in ('none','basic'):
                harness=await Harness().start(mode)
                kwargs={'http_credentials':{'username':'workbench','password':PASSWORD}} if mode=='basic' else {}
                context=await browser.new_context(**kwargs)
                page=await context.new_page()
                errors=[]
                page.on('pageerror',lambda error: errors.append(str(error)))
                try:
                    response=await page.goto(str(harness.client.make_url('/')),wait_until='networkidle')
                    assert response.status==200
                    await page.wait_for_function('window.gatewayTestLoaded === true')
                    result=await page.evaluate('''async () => {
                        const data = new Uint8Array([0,1,2,3,255,128,64]);
                        const form = new FormData();
                        form.append('image', new Blob([data], {type:'image/png'}), 'test.png');
                        const uploaded = await fetch('/upload/image', {method:'POST',body:form});
                        const upload = await uploaded.json();
                        const prompt = await fetch('/prompt', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:{}})});
                        const catalog = await (await fetch('/_workbench/catalog')).json();
                        const health = await (await fetch('/healthz')).json();
                        const socket = await new Promise((resolve,reject) => {
                            const ws = new WebSocket(location.origin.replace(/^http/,'ws')+'/ws');
                            const timer=setTimeout(()=>{ws.close();reject(new Error('WebSocket timeout'));},5000);
                            ws.onopen=()=>ws.send('progress-ready');
                            ws.onmessage=(event)=>{clearTimeout(timer);ws.close();resolve(event.data);};
                            ws.onerror=()=>{clearTimeout(timer);reject(new Error('WebSocket error'));};
                        });
                        return {uploadStatus:uploaded.status, uploadBytes:upload.bytes,
                                promptStatus:prompt.status, workspace:catalog.workspace,
                                authMode:health.auth_mode, websocket:socket};
                    }''')
                    assert result=={'uploadStatus':200,'uploadBytes':7,'promptStatus':200,
                                    'workspace':'qwen','authMode':mode,'websocket':'progress-ready'},result
                    assert not errors,errors
                    results.append({'mode':mode,'passed':True,'checks':result})
                finally:
                    await context.close()
                    await harness.close()
        finally:
            await browser.close()
    return {'browser':'Chromium','results':results,
            'scope':'Real local browser; simulated ComfyUI upstream; no RunPod, GPU, or paid API call.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--executable',default=shutil.which('chromium'))
    parser.add_argument('--report')
    args=parser.parse_args()
    report=asyncio.run(run(args.executable))
    print(json.dumps(report,indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
