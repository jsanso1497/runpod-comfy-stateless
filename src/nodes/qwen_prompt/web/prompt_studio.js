import { app } from '../../scripts/app.js';
import { ComfyWidgets } from '../../scripts/widgets.js';

app.registerExtension({
  name: 'Workbench.QwenPromptStudio',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'WBQwenPromptStudio') return;
    const oldCreated = nodeType.prototype.onNodeCreated;
    const oldExecuted = nodeType.prototype.onExecuted;
    const oldStart = nodeType.prototype.onExecutionStart;
    function transient(widget) {
      widget.options ??= {};
      widget.options.serialize = false;
      widget.serializeValue = () => undefined;
      return widget;
    }
    function ensure(node) {
      if (node._studioPreview) return;
      node._studioPrompt = '';
      node._studioFilename = 'qwen_prompt.txt';
      node._studioPreview = transient(ComfyWidgets.STRING(node, 'FINAL PROMPT - copy or download',
        ['STRING', {multiline: true}], app).widget);
      if (node._studioPreview.inputEl) node._studioPreview.inputEl.readOnly = true;
      node._studioStatus = transient(ComfyWidgets.STRING(node, 'STATUS / IMAGE ROLES (not exported)',
        ['STRING', {multiline: true}], app).widget);
      if (node._studioStatus.inputEl) node._studioStatus.inputEl.readOnly = true;
      transient(node.addWidget('button', 'Copy final prompt', null, async () => {
        if (!node._studioPrompt) return;
        try {
          await navigator.clipboard.writeText(node._studioPrompt);
        } catch {
          const field = node._studioPreview.inputEl;
          if (field) { field.focus(); field.select(); document.execCommand('copy'); }
        }
      }));
      transient(node.addWidget('button', 'Download prompt .txt', null, () => {
        if (!node._studioPrompt) return;
        const url = URL.createObjectURL(new Blob([node._studioPrompt + '\n'], {type: 'text/plain;charset=utf-8'}));
        const link = document.createElement('a');
        link.href = url;
        link.download = node._studioFilename;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }));
    }
    nodeType.prototype.onNodeCreated = function() {
      const result = oldCreated?.apply(this, arguments);
      ensure(this);
      return result;
    };
    nodeType.prototype.onExecutionStart = function() {
      const result = oldStart?.apply(this, arguments);
      ensure(this);
      this._studioPrompt = '';
      this._studioPreview.value = '';
      this._studioStatus.value = 'Analyzing images and writing prompt...';
      return result;
    };
    nodeType.prototype.onExecuted = function(message) {
      const result = oldExecuted?.apply(this, arguments);
      ensure(this);
      const text = message?.text?.[0];
      if (typeof text !== 'string') return result;
      this._studioPrompt = text;
      this._studioFilename = message.studio_filename?.[0] || 'qwen_prompt.txt';
      this._studioPreview.value = text;
      this._studioStatus.value = (message.studio_status?.[0] || '') + '\n' +
        (message.studio_roles?.[0] || '') +
        (message.studio_saved?.[0] ? '\nSaved: ' + message.studio_saved[0] : '');
      const computed = this.computeSize();
      this.setSize([Math.max(this.size[0], 580), Math.max(this.size[1], computed[1], 1050)]);
      this.setDirtyCanvas(true, true);
      return result;
    };
  }
});
