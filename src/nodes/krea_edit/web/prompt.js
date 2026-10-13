import { app } from '../../../scripts/app.js';
import { ComfyWidgets } from '../../../scripts/widgets.js';
app.registerExtension({
  name: 'workbench.krea.promptPreview',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'WBKreaPrompt') return;
    const old = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function(message) {
      old?.apply(this, arguments);
      let widget = this.widgets?.find(w => w.name === 'final_prompt_preview');
      if (!widget) {
        widget = ComfyWidgets.STRING(this, 'final_prompt_preview', ['STRING', {multiline: true}], app).widget;
        widget.options.serialize = false;
        if (widget.inputEl) widget.inputEl.readOnly = true;
      }
      widget.value = (message.text || []).join('\n');
      this.setDirtyCanvas(true, true);
    };
  }
});
