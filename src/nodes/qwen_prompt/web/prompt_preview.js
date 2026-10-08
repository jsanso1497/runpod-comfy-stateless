import { app } from '../../scripts/app.js';
import { ComfyWidgets } from '../../scripts/widgets.js';

const supported = new Set(['Q21NEncodeReferences', 'Q21PhotoEncode', 'WBQwenPromptPreview']);
app.registerExtension({
  name: 'Workbench.QwenI2IPromptPreview',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!supported.has(nodeData.name)) return;
    const previous = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function(message) {
      const result = previous?.apply(this, arguments);
      const report = message?.qwen_prompt?.[0];
      if (!report) return result;
      if (!this._wbPromptPreview) {
        this._wbPromptPreview = ComfyWidgets.STRING(this, 'Prompt actually used (read-only)',
          ['STRING', {multiline: true}], app).widget;
        this._wbPromptPreview.inputEl.readOnly = true;
        this._wbPromptPreview.options ??= {};
        this._wbPromptPreview.options.serialize = false;
        this._wbPromptPreview.serializeValue = () => undefined;
        const size = this.computeSize();
        this.setSize([Math.max(this.size[0], 400), Math.max(this.size[1], size[1] + 160)]);
      }
      this._wbPromptPreview.value = report.status + (report.cache_hit ? ' | cached' : '') +
        '\n\nPROMPT USED\n' + report.effective_prompt + '\n\nORIGINAL\n' + report.original_prompt;
      this.setDirtyCanvas(true, true);
      return result;
    };
  }
});
