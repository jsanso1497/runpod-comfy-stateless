import { app } from '../../scripts/app.js';

const TASK = 'QLOCAL';
function el(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}
async function read(path, body) {
  const response = await fetch('/_workbench/' + path, {
    method: body ? 'POST' : 'GET',
    headers: body ? {'Content-Type': 'application/json'} : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error('Workbench request failed: ' + response.status);
  return response.json();
}
async function render(container) {
  container.replaceChildren();
  const panel = el('div');
  panel.style.cssText = 'padding:16px;font:13px/1.5 system-ui;overflow:auto;height:100%';
  container.append(panel);
  panel.append(el('h2', 'Approved workflow'));
  panel.append(el('h3', '00 | SAM-Guided Local Edit'));
  panel.append(el('p', 'Qwen BF16 + SAM. No prompt enhancer or other model family.'));
  const status = el('p', 'Checking readiness...');
  status.style.whiteSpace = 'pre-wrap';
  panel.append(status);
  const open = el('button', 'Open approved workflow');
  const prepare = el('button', 'Verify / prepare Qwen + SAM');
  for (const button of [open, prepare]) {
    button.style.cssText = 'display:block;margin:10px 0;padding:9px;width:100%';
    panel.append(button);
  }
  panel.append(el('p', 'Also in Workflows > 00_Approved. Existing saved workflows are not deleted.'));
  const note = el('small', 'User-approved workflow. Approval does not certify GPU results.');
  panel.append(note);
  let busy = false;
  async function poll() {
    if (busy) return;
    busy = true;
    try {
      const data = await read('status');
      const task = data.tasks.find(t => t.id === TASK);
      const variant = task?.variants?.[0];
      const job = data.preparation || {};
      const progress = job.progress;
      const lines = [variant?.state === 'ready' ? 'Models and nodes are ready.' : 'Preparing / checking required models and nodes.'];
      if (task?.missing_assets?.length) lines.push('Missing assets: ' + task.missing_assets.join(', '));
      if (variant?.missing_nodes?.length) lines.push('Missing nodes: ' + variant.missing_nodes.join(', '));
      lines.push('Preparation: ' + (job.state || 'unknown'));
      if (job.failures) lines.push('Asset failures: ' + job.failures);
      if (progress) lines.push(progress.stage + (progress.asset_percent == null ? '' : ' ' + progress.asset_percent + '%'));
      status.textContent = lines.join('\n');
    } catch (error) { status.textContent = error.message; }
    finally { busy = false; }
  }
  open.onclick = async () => {
    open.disabled = true;
    try {
      const catalog = await read('catalog');
      const task = catalog.tasks.find(t => t.id === TASK);
      if (!task || catalog.tasks.length !== 1) throw new Error('Expected the approved-only catalog. Check the image and profile.');
      const graph = await read('workflow?file=' + encodeURIComponent(task.variants[0].file));
      await app.loadGraphData(graph);
      status.textContent = 'Opened SAM-Guided Local Edit. Upload your image before queuing.';
    } catch (error) { status.textContent = error.message; }
    finally { open.disabled = false; }
  };
  prepare.onclick = async () => {
    prepare.disabled = true;
    try {
      await read('prepare', {tasks: [TASK], features: [], private: false});
      await poll();
    } catch (error) { status.textContent = error.message; }
    finally { prepare.disabled = false; }
  };
  await poll();
  const timer = setInterval(() => {
    if (!panel.isConnected) { clearInterval(timer); return; }
    poll();
  }, 15000);
}
app.registerExtension({name: 'Workbench.ApprovedLocalEdit', async setup() {
  if (app.extensionManager?.registerSidebarTab) {
    app.extensionManager.registerSidebarTab({id: 'workbench-approved', icon: 'pi pi-check',
      title: 'Approved', tooltip: 'SAM-Guided Local Edit', type: 'custom', render});
  } else {
    const button = el('button', 'Approved workflow');
    button.style.cssText = 'position:fixed;right:10px;top:80px;z-index:1000';
    document.body.append(button);
    let panel;
    button.onclick = () => {
      if (panel) { panel.remove(); panel = null; return; }
      panel = el('div');
      panel.style.cssText = 'position:fixed;right:15px;top:120px;bottom:20px;width:min(400px,90vw);z-index:1000;background:var(--comfy-menu-bg,#202020)';
      document.body.append(panel); render(panel);
    };
  }
}});
