import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

app.registerExtension({
  name: 'h3.video-still-studio',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === 'H3VVideo') {
      const created = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        created?.apply(this, arguments);
        const chooser = document.createElement('input');
        chooser.type = 'file';
        chooser.accept = '.mp4,.mov,.mkv,.webm,.m4v,.avi';
        chooser.onchange = async () => {
          const file = chooser.files?.[0];
          if (!file) return;
          const form = new FormData();
          form.append('image', file); form.append('type', 'input');
          form.append('overwrite', 'false');
          try {
            const response = await api.fetchApi('/upload/image', { method: 'POST', body: form });
            if (!response.ok) throw new Error(await response.text());
            const uploaded = await response.json();
            const value = uploaded.subfolder ? `${uploaded.subfolder}/${uploaded.name}` : uploaded.name;
            const widget = this.widgets.find(w => w.name === 'file');
            if (!widget.options.values.includes(value)) widget.options.values.push(value);
            widget.value = value; widget.callback?.(value); app.graph.setDirtyCanvas(true, true);
          } catch (error) {
            alert(`Video upload failed: ${error.message}. Upload into the workspace input folder with Files/Jupyter, then refresh ComfyUI.`);
          } finally { chooser.value = ''; }
        };
        const button = this.addWidget('button', 'Upload reference video', null, () => chooser.click());
        button.options = { ...(button.options || {}), serialize: false };
      };
    }
    if (nodeData.name === 'H3VAudit') {
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        executed?.apply(this, arguments);
        let widget = this.widgets.find(w => w.name === 'audit_result');
        if (!widget) widget = this.addWidget('text', 'audit_result', '', null, { multiline: true, serialize: false });
        widget.value = (message.text || []).join('\n');
      };
    }
  }
});
