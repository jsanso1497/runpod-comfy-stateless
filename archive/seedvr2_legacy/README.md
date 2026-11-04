# Retired SeedVR2 entrypoints

The installer moves reviewed obsolete video-build actions and old factory workflows
here without discarding their source. These are historical, not the current setup.
Use `SEEDVR2_START_HERE.md`, Build SeedVR2 Stock FP16, and 01_Video_Clarity.
Runtime still contains only upstream ComfyUI, the official SeedVR2 extension and the
existing standard Nginx authentication layer. Old Faithful worker/gateway code is not
installed by the new image. No saved media or model weights are migrated or deleted
by the local installer.
