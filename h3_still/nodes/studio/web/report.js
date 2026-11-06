import { app } from '/scripts/app.js';
import { ComfyWidgets } from '/scripts/widgets.js';

// Read-only execution report. This does not modify graph inputs or queue jobs.
app.registerExtension({
  name: 'h3.still.studio.report',
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== 'H3StudioAudit') return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const widget = ComfyWidgets.STRING(this, 'last_run_report', ['STRING', {multiline: true}], app).widget;
      widget.options.serialize = false;
      if (widget.inputEl) widget.inputEl.readOnly = true;
      widget.value = 'Run to see the selected model, mode, output dimensions and Picture tags.';
      this.h3ReportWidget = widget;
      return result;
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      executed?.apply(this, arguments);
      if (this.h3ReportWidget) this.h3ReportWidget.value = (message.text || []).join('\n');
      this.setDirtyCanvas?.(true, true);
    };
  }
});
