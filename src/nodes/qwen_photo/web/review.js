import { app } from '/scripts/app.js';
import { api } from '/scripts/api.js';

app.registerExtension({
  name: 'Qwen21Photo.Review',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'Q21PhotoReviewSave') return;
    const previousCreate = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = previousCreate?.apply(this, arguments);
      const root = document.createElement('div');
      root.style.cssText = 'display:flex;flex-direction:column;gap:8px;padding:8px;box-sizing:border-box;width:100%;height:100%;color:var(--input-text,#ddd);';
      const title = document.createElement('div');
      title.textContent = 'ORIGINAL (left) / RESULT OR MASK (right)';
      const select = document.createElement('select');
      for (const [value, label] of [['full', 'Full photograph / mask preview'], ['crop', 'Edit crop comparison']]) {
        const option = document.createElement('option'); option.value = value; option.textContent = label; select.appendChild(option);
      }
      const frame = document.createElement('div');
      frame.style.cssText = 'position:relative;width:100%;flex:1;min-height:200px;overflow:hidden;background:#171717;';
      const original = new Image(), edited = new Image();
      for (const image of [edited, original]) {
        image.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain;';
        frame.appendChild(image);
      }
      const line = document.createElement('div');
      line.style.cssText = 'position:absolute;top:0;bottom:0;left:50%;width:1px;background:white;pointer-events:none;';
      frame.appendChild(line);
      const slider = document.createElement('input');
      slider.type = 'range'; slider.min = '0'; slider.max = '100'; slider.value = '50';
      slider.setAttribute('aria-label', 'Before and after divider');
      const update = () => { original.style.clipPath = `inset(0 ${100 - Number(slider.value)}% 0 0)`; line.style.left = `${slider.value}%`; };
      slider.addEventListener('input', update);
      for (const element of [slider, select]) element.addEventListener('pointerdown', event => event.stopPropagation());
      const report = document.createElement('textarea'); report.readOnly = true;
      report.style.cssText = 'width:100%;height:130px;box-sizing:border-box;resize:none;font:11px monospace;';
      report.value = 'Run with run_edit OFF to inspect the mask. Turn ON after review.';
      root.append(title, select, frame, slider, report);
      this.addDOMWidget('q21_review_view', 'custom', root, {
        serialize: false, hideOnZoom: true, getMinHeight: () => 430,
        getHeight: () => Math.max(430, (this.size?.[1] || 700) - 150),
      });
      const state = { original, edited, report, select, views: {} };
      const show = () => {
        const pair = state.views[select.value]; if (!pair || pair.length !== 2) return;
        const url = image => api.apiURL('/view?' + new URLSearchParams({filename:image.filename, subfolder:image.subfolder || '', type:image.type || 'temp'}));
        original.src = url(pair[0]); edited.src = url(pair[1]);
      };
      select.addEventListener('change', show);
      state.show = show;
      this._q21Review = state; this.setSize([710, 700]); update();
      return result;
    };
    const previousExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      previousExecuted?.apply(this, arguments);
      const state = this._q21Review;
      if (!state || !message.q21_full?.length) return;
      state.views = {full: message.q21_full, crop: message.q21_crop || []};
      state.select.options[1].disabled = state.views.crop.length !== 2;
      if (state.select.options[1].disabled) state.select.value = 'full';
      state.report.value = (message.q21_report || []).join('\n');
      state.show();
    };
  },
});
