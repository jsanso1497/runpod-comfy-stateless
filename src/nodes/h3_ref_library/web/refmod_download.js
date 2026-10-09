import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "quality.refmod.download",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "QualityExportRefMods") return;
        const old = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            old?.apply(this, arguments);
            const path = message?.refmod_download?.[0];
            if (typeof path !== "string" || !/^\/quality\/refmod-exports\/refmods-[0-9]{8}-[0-9]{6}-[0-9a-f]{12}\.zip$/.test(path)) return;
            if (!this._qualityDownloadLink) {
                const wrapper = document.createElement("div");
                wrapper.style.padding = "12px";
                const anchor = document.createElement("a");
                anchor.textContent = "Download saved RefMod library ZIP";
                anchor.rel = "noopener";
                anchor.download = "";
                wrapper.appendChild(anchor);
                this.addDOMWidget("refmod_download", "download", wrapper, { serialize: false });
                this._qualityDownloadLink = anchor;
            }
            this._qualityDownloadLink.href = new URL(path, window.location.origin).href;
            this.setDirtyCanvas(true, true);
        };
    }
});
