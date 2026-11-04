import { app } from "../../scripts/app.js";
app.registerExtension({
  name: "workbench.seedvr.streaming",
  async setup() {
    if (app.extensionManager?.registerSidebarTab) {
      app.extensionManager.registerSidebarTab({
        id: "wb-video-jobs", icon: "pi pi-folder-open", title: "Files and video jobs",
        tooltip: "Saved files, video diagnostics, and memory", type: "custom",
        render: (container) => {
          container.replaceChildren();
          const link = document.createElement("a");
          link.href = "/_workbench/repair";
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = "Open saved files, video jobs, and memory diagnostics";
          link.style.cssText = "display:block;padding:16px;line-height:1.5";
          container.append(link);
        }
      });
    } else {
      console.info("Workbench saved files and video jobs: /_workbench/repair");
    }
  }
});
