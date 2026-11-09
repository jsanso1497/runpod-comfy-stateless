import { app } from "/scripts/app.js";
app.registerExtension({
  name: "H3StudioLab.Reports",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["H3LabReport", "H3LabRefModSave"].includes(nodeData.name)) return;
    const prior = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      prior?.apply(this, arguments);
      const value = (message.text || []).join("\n");
      if (!this._h3LabReportWidget) {
        const area = document.createElement("textarea");
        area.readOnly = true;
        area.style.cssText = "width:100%;height:100%;resize:none;box-sizing:border-box;font:12px monospace";
        this._h3LabReportWidget = this.addDOMWidget("report_preview", "text", area, {serialize:false});
        this._h3LabReportWidget.options.serialize = false;
        this._h3LabReportElement = area;
      }
      this._h3LabReportElement.value = value;
    };
  }
});
