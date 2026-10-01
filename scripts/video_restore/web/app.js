'use strict';
const $ = (id) => document.getElementById(id);
let csrf = '', uploaded = null, jobId = null, currentReady = false, rendered = null, busy = false;
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf}, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function message(error) { $('error').textContent = error ? String(error.message || error) : ''; }
$('file').addEventListener('change', () => { $('upload').disabled = !$('file').files.length || !csrf; });
$('upload').addEventListener('click', () => {
  const file = $('file').files[0]; if (!file) return;
  $('upload').disabled = true; $('uploadStatus').textContent = 'Uploading to your Pod...';
  const xhr = new XMLHttpRequest();
  xhr.open('POST', `/api/upload?name=${encodeURIComponent(file.name)}`);
  xhr.setRequestHeader('Content-Type', 'application/octet-stream');
  xhr.setRequestHeader('X-CSRF-Token', csrf);
  xhr.upload.onprogress = e => { if (e.lengthComputable) $('progress').value = 100 * e.loaded / e.total; };
  xhr.onload = () => {
    $('upload').disabled = false;
    try {
      const data = JSON.parse(xhr.responseText);
      if (xhr.status >= 400) throw new Error(data.error);
      uploaded = data; $('progress').value = 100;
      $('uploadStatus').textContent = `${data.name} | ${data.duration.toFixed(2)} seconds | ${data.fps} fps. ${(data.warnings || []).join(' ')}`;
      $('run').disabled = !currentReady || busy;
    } catch(e) { $('uploadStatus').textContent = 'Upload or validation failed.'; message(e); }
  };
  xhr.onerror = () => { $('upload').disabled = false; message('Upload interrupted. Check the connection and retry.'); };
  xhr.send(file);
});
$('run').addEventListener('click', async () => {
  try {
    message(null);
    const seconds = Number($('duration').value);
    if (seconds > 15 && !confirm('This processes more than a short preview and may take substantial paid GPU time. Continue?')) return;
    $('run').disabled = true;
    const result = await api('/api/run', {upload_id: uploaded.id, options: {
      start: Number($('start').value), duration: seconds, mode: $('mode').value, shadow: $('shadow').value
    }});
    jobId = result.id; rendered = null; $('results').hidden = true; busy = true;
  } catch(e) { message(e); $('run').disabled = !currentReady || !uploaded; }
});
$('cancel').addEventListener('click', async () => { try { await api('/api/cancel', {}); } catch(e) { message(e); } });
$('delete').addEventListener('click', async () => {
  if (!jobId || !confirm('Permanently delete the local outputs and logs for this job? Download them first.')) return;
  try { await api('/api/delete', {id: jobId}); jobId = null; rendered = null; $('results').hidden = true; } catch(e) { message(e); }
});
$('playBoth').addEventListener('click', () => {
  for (const name of ['before', 'after']) { const video = $(name); video.currentTime = 0; video.play().catch(() => {}); }
  $('after').muted = true;
});
function showResults(job) {
  if (rendered === job.id) return;
  rendered = job.id;
  const names = job.outputs || [];
  $('results').hidden = false;
  const prefix = `/files/${job.id}/`;
  const afterName = names.find(n => n.startsWith('03_')) || names.find(n => n.startsWith('02_'));
  $('before').src = prefix + '01_Original_1080p.mp4';
  $('after').src = afterName ? prefix + afterName : '';
  $('downloads').replaceChildren();
  for (const name of names) {
    const a = document.createElement('a'); a.href = prefix + name + '?download=1'; a.textContent = name; $('downloads').append(a);
  }
  const detail = names.find(n => n.endsWith('.png'));
  $('detail').hidden = !detail; if (detail) $('detail').src = prefix + detail;
  $('reportSummary').textContent = job.report ? `Completed in ${Math.round(job.report.elapsed_seconds)} seconds. ${(job.report.warnings || []).join(' ')}` : 'Some outputs may be incomplete. Check the report and processing log.';
}
async function poll() {
  try {
    const state = await api('/api/state'); csrf = state.csrf; currentReady = state.ready;
    $('stage').textContent = state.stage; $('startupError').textContent = state.error || '';
    $('ready').textContent = state.error ? 'Error' : state.ready ? 'Ready' : 'Preparing';
    $('ready').className = `badge ${state.error ? 'bad' : state.ready ? 'good' : ''}`;
    const h = state.hardware;
    $('hardware').textContent = h.gpu ? `${h.gpu} | ${h.vram_gib} GiB VRAM | ${h.ram_gib} GiB RAM | ${state.model.toUpperCase()} FP16 | attention: ${h.attention}. ${(h.warnings || []).join(' ')}` : '';
    if (state.active) jobId = state.active.id;
    busy = !!state.active && state.active.status === 'running';
    $('cancel').disabled = !busy;
    $('run').disabled = !state.ready || !uploaded || busy;
    if ($('file').files.length) $('upload').disabled = false;
    if (jobId) {
      const job = await api(`/api/job/${jobId}`);
      $('log').textContent = job.log_tail || 'Preparing the job...';
      if (job.status !== 'running') { showResults(job); if (job.error) message(job.error); }
    }
  } catch(e) { message(e); }
  setTimeout(poll, 2500);
}
poll();
