# H3 Still Studio v2

Use [H3_STUDIO_V2.md](H3_STUDIO_V2.md) for the current setup.

This update adds video references, a person-replacement workflow, a password sign-in page, and an independent file browser. It keeps the INT8 model stack and the existing photo-only Studio workflows.

Apply `update_h3_studio_video_person.py` locally, review/commit/push in GitHub Desktop, then start a new **Build H3 Still Developer Workspace** Action. Use the new build digest and HTTP ports `8188,8888`.

On first access, use `/__h3/login` on the active Pod's HTTPS service URL. Both services take `comfy_password` only, with no username. Port 8888 opens Files/Jupyter; the file browser no longer depends on JupyterLab initializing.

Open **Workflows > 00_H3_Studio_v2**. Preserve `/workspace`. Do not apply the older padding/recovery/Studio installers afterward.
