#!/usr/bin/env python3
"""Offline configurator smoke test. Install Playwright and Chromium to reproduce."""
from pathlib import Path
import json,shutil
from playwright.sync_api import sync_playwright
R=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    options={'headless':True,'args':['--no-sandbox']}
    if shutil.which('chromium'):options['executable_path']=shutil.which('chromium')
    browser=p.chromium.launch(**options)
    page=browser.new_page(viewport={'width':1480,'height':1050},device_scale_factor=1)
    errors=[];requests=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:requests.append(r.url))
    page.set_content((R/'site/index.html').read_text(),wait_until='load');page.wait_for_timeout(300)
    assert page.title().startswith('Comfy Workbench')
    assert page.locator('.task-row').count()>30
    page.screenshot(path=str(R/'reports/configurator-desktop.png'),full_page=True)
    page.locator('[data-model="h3"]').click()
    assert 'RUNPOD_SECRET_hf_token' in page.locator('#result').input_value()
    assert page.evaluate("window.WorkbenchApp.plan().asset_groups.includes('ollama')")
    page.locator('#private-loras').check()
    assert 'RUNPOD_SECRET_comfy_loras' in page.locator('#result').input_value()
    page.locator('[data-model="qwen"]').click()
    page.locator('#search').fill('shirt')
    page.locator('#search').fill('')
    with page.expect_download() as item:page.locator('#download-json').click()
    assert item.value.suggested_filename=='qwen-runpod-settings.json'
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(R/'reports/configurator-mobile.png'),full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert not errors
    assert not any(url.startswith(('http://','https://')) for url in requests)
    report={'browser':'headless Chromium','standalone_html_rendered_inline':True,'file_navigation':'restricted by the testing environment; inline HTML used','desktop':True,'mobile_no_horizontal_overflow':True,'model_selection':True,'private_secret_references':True,'json_download':True,'network_requests':0,'javascript_errors':errors}
    (R/'reports/browser-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));browser.close()
