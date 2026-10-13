import { app } from '/scripts/app.js';
import { api } from '/scripts/api.js';

app.registerExtension({
  name: 'FluxPhoto.ProtectedCompare',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'FluxPhotoCompare') return;
    const oldCreate = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = oldCreate?.apply(this, arguments);
      const root = document.createElement('div');
      root.style.cssText = 'display:flex;flex-direction:column;gap:8px;padding:8px;box-sizing:border-box;width:100%;height:100%;color:var(--input-text,#ddd);';
      const label = document.createElement('div');
      label.textContent = 'ORIGINAL (left) / EDITED (right)';
      const frame = document.createElement('div');
      frame.style.cssText = 'position:relative;width:100%;flex:1;min-height:180px;overflow:hidden;background:#161616;';
      const original = new Image();
      const edited = new Image();
      for (const image of [edited, original]) {
        image.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain;';
        frame.appendChild(image);
      }
      const divider = document.createElement('div');
      divider.style.cssText = 'position:absolute;top:0;bottom:0;left:50%;width:1px;background:white;pointer-events:none;';
      frame.appendChild(divider);
      const slider = document.createElement('input');
      slider.type = 'range'; slider.min = '0'; slider.max = '100'; slider.value = '50';
      slider.setAttribute('aria-label', 'Original and edited comparison divider');
      const update = () => {
        original.style.clipPath = `inset(0 ${100 - Number(slider.value)}% 0 0)`;
        divider.style.left = `${slider.value}%`;
      };
      slider.addEventListener('input', update);
      slider.addEventListener('pointerdown', event => event.stopPropagation());
      const report = document.createElement('textarea');
      report.readOnly = true;
      report.style.cssText = 'width:100%;height:105px;box-sizing:border-box;resize:none;font:11px monospace;';
      report.value = 'Run the workflow to compare. Previews may be reduced; saved output is full size.';
      root.append(label, frame, slider, report);
      this.addDOMWidget('photo_compare_view', 'custom', root, {
        serialize: false, hideOnZoom: true,
        getMinHeight: () => 360,
        getHeight: () => Math.max(360, (this.size?.[1] || 560) - 125),
      });
      this._fluxPhotoCompare = { original, edited, report };
      this.setSize([700, 600]);
      update();
      return result;
    };
    const oldExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      oldExecuted?.apply(this, arguments);
      if (!this._fluxPhotoCompare || !message.photo_compare?.length) return;
      const url = image => api.apiURL('/view?' + new URLSearchParams({
        filename: image.filename, subfolder: image.subfolder || '', type: image.type || 'temp',
      }).toString());
      this._fluxPhotoCompare.original.src = url(message.photo_compare[0]);
      this._fluxPhotoCompare.edited.src = url(message.photo_compare[1]);
      this._fluxPhotoCompare.report.value = (message.photo_report || []).join('\n');
    };
  },
});
