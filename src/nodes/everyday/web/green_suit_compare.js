import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Only local ComfyUI image URLs. No credentials or image uploads in this UI.
function imageURL(item) {
  const query = new URLSearchParams({
    filename: item.filename, subfolder: item.subfolder || "", type: item.type || "temp"
  });
  return api.apiURL(`/view?${query.toString()}`);
}
function element(tag, style, text) {
  const item = document.createElement(tag);
  if (style) Object.assign(item.style, style);
  if (text) item.textContent = text;
  return item;
}

function installViewer(node) {
  if (node._wbgsViewer || typeof node.addDOMWidget !== "function") return;
  const root = element("div", { display: "flex", flexDirection: "column", gap: "8px",
    padding: "10px", boxSizing: "border-box", background: "#171a20", color: "#f0f1f3",
    font: "13px sans-serif", height: "100%", overflow: "auto" });
  const toolbar = element("div", {display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px"});
  const a = element("select", {width: "100%", minWidth: "0", padding: "5px"});
  const b = element("select", {width: "100%", minWidth: "0", padding: "5px"});
  a.setAttribute("aria-label", "Image A"); b.setAttribute("aria-label", "Image B");
  toolbar.append(a, b);
  const modes = element("div", {display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center"});
  const mode = element("select", {padding: "5px"});
  for (const [value, text] of [["wipe", "A/B wipe"], ["overlay", "50% overlay"], ["A", "A only"], ["B", "B only"]]) {
    mode.add(new Option(text, value));
  }
  const nativeLabel = element("label", {display: "flex", gap: "5px", alignItems: "center"});
  const full = element("input"); full.type = "checkbox";
  nativeLabel.append(full, document.createTextNode("Full-resolution images"));
  modes.append(mode, nativeLabel);
  const stage = element("div", {position: "relative", flex: "1 0 380px", minHeight: "300px",
    aspectRatio: "2 / 3", width: "100%", maxHeight: "700px", background: "#090b0d", overflow: "hidden"});
  const imageStyle = {position: "absolute", width: "100%", height: "100%", objectFit: "contain", left: "0", top: "0", pointerEvents: "none"};
  const imgA = element("img", imageStyle), imgB = element("img", imageStyle);
  imgA.alt = "Comparison A"; imgB.alt = "Comparison B";
  stage.append(imgA, imgB);
  const slider = element("input", {width: "100%"});
  slider.type = "range"; slider.min = "0"; slider.max = "100"; slider.value = "50";
  slider.setAttribute("aria-label", "Comparison wipe position");
  const status = element("div", {lineHeight: "1.4"}, "Run mask preview to start. Comparisons do not run models or the API.");
  const links = element("div", {display: "flex", flexWrap: "wrap", gap: "12px"});
  const linkA = element("a", {color: "#b4d3ff"}, "Open A at full size");
  const linkB = element("a", {color: "#b4d3ff"}, "Open B at full size");
  for (const link of [linkA, linkB]) { link.target = "_blank"; link.rel = "noopener"; }
  links.append(linkA, linkB);
  const details = element("details");
  details.append(element("summary", {cursor: "pointer"}, "Geometry and API report"));
  const report = element("pre", {font: "11px monospace", whiteSpace: "pre-wrap", overflowWrap: "anywhere", maxHeight: "240px", overflow: "auto"});
  details.append(report);
  root.append(toolbar, modes, stage, slider, status, links, details);
  let previews = [], saved = [];
  function styleImages() {
    imgA.style.display = mode.value === "B" ? "none" : "block";
    imgB.style.display = mode.value === "A" ? "none" : "block";
    imgB.style.opacity = mode.value === "overlay" ? "0.5" : "1";
    imgB.style.clipPath = mode.value === "wipe" ? `inset(0 0 0 ${slider.value}%)` : "none";
    slider.disabled = mode.value !== "wipe";
  }
  function selectImages() {
    const items = full.checked ? saved : previews;
    const ia = Math.max(0, Number(a.value)), ib = Math.max(0, Number(b.value));
    if (!items[ia] || !items[ib]) return;
    imgA.src = imageURL(items[ia]); imgB.src = imageURL(items[ib]);
    linkA.href = imageURL(saved[ia] || items[ia]); linkB.href = imageURL(saved[ib] || items[ib]);
    status.textContent = `A: ${items[ia].label} | B: ${items[ib].label}. ${full.checked ? "Full-resolution sources" : "640 x 960 previews"}; both displayed on the same 2:3 canvas.`;
    styleImages();
  }
  for (const input of [a, b, full]) input.addEventListener("change", selectImages);
  mode.addEventListener("change", styleImages); slider.addEventListener("input", styleImages);
  // Keep viewer controls from dragging the graph node, without blocking defaults.
  for (const control of [toolbar, modes, slider, links, details]) {
    for (const event of ["pointerdown", "mousedown", "dblclick"]) control.addEventListener(event, e => e.stopPropagation());
  }
  const widget = node.addDOMWidget("green_suit_progression", "wbgs_compare", root, {serialize: false, hideOnZoom: false});
  widget.computeSize = width => [width, 880];
  node._wbgsViewer = {
    root,
    update(message) {
      const previousCount = previews.length;
      previews = message.wbgs_images || [];
      saved = message.wbgs_saved || previews;
      if (!previews.length) return;
      const oldA = a.options[a.selectedIndex]?.text, oldB = b.options[b.selectedIndex]?.text;
      a.replaceChildren(); b.replaceChildren();
      previews.forEach((item, i) => { a.add(new Option(item.label, String(i))); b.add(new Option(item.label, String(i))); });
      // Prefer the green reference versus the finished output after a full run.
      const indexA = previousCount === previews.length ? previews.findIndex(x => x.label === oldA) : -1;
      const indexB = previousCount === previews.length ? previews.findIndex(x => x.label === oldB) : -1;
      const defaultA = previews.length >= 7 ? 3 : previews.length >= 4 ? 2 : 0;
      a.value = String(indexA >= 0 ? indexA : defaultA);
      b.value = String(indexB >= 0 && indexB !== Number(a.value) ? indexB : previews.length - 1);
      report.textContent = (message.wbgs_report || []).join("\n");
      selectImages();
    }
  };
  node.setSize([Math.max(node.size[0], 530), Math.max(node.size[1], 1030)]);
}

app.registerExtension({
  name: "Workbench.GreenSuit.ProgressionCompare",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "WBGSReviewCompare") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() { const value = created?.apply(this, arguments); installViewer(this); return value; };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function(message) {
      const value = executed?.apply(this, arguments);
      installViewer(this); this._wbgsViewer?.update(message); return value;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function() { this._wbgsViewer?.root.remove(); return removed?.apply(this, arguments); };
  }
});
