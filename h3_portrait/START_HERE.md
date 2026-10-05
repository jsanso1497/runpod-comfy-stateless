# H3 Portrait setup

Follow **../START_HERE.md** at the repository root for this complete package.
It includes browser-only upload steps, both required HTTP ports (8188,8888), the
new FILEBROWSER_PASSWORD secret, all environment variables and the initial test.

The portrait implementation remains 1.3.0 (split-model-reference-v3). The added
file manager does not change prompt inputs, generation settings, LoRA selection,
model editions or the 9:16/2:3 output handling. Build H3 Portrait template and use
image ghcr.io/jsanso1497/runpod-comfy-stateless:h3-portrait-clean.

The file manager starts automatically before model downloads, with JupyterLab's
password login on port 8888. It browses /workspace and provides upload/download,
editing and a server-side terminal. Port 8188 still opens normal ComfyUI.
