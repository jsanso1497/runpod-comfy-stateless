# GitHub overlay for Qwen21 Native Suite v1.1.0

Merge this `qwen21_native` folder into the repository root. It assumes the
previously supplied Native Suite v1.1.0 is already present. It is an overlay,
not the full suite, and does not target the old `qwen21_identity` folder.

Changed existing files:
- qwen21_native/Dockerfile: run the addon tests inside the existing Torch test environment.
- qwen21_native/scripts/start.sh: install the addon before ComfyUI starts.
- qwen21_native/scripts/serve_checked.py: check the addon against the real runtime node schemas.

Added folder: qwen21_native/addons/torso_lock

No Action, model, upstream pin, precision, sampler default, or image tag changes.
Commit these files and run Build Qwen 2.1 Native Suite on that NEW commit. Do not
rerun an old commit and expect it to contain the new code. No image has been
built or published by this package.

For a modified native checkout, use APPLY_TORSO_TO_REPO.py from the standalone
addon download instead; it checks the integration points and makes backups.

After the new image starts, find the four new workflows under
Qwen21 Torso Lock v1.0.0. Start with 36_Sports_Bra_Torso_Landmark_Lock.

Read qwen21_native/addons/torso_lock/README.md for image roles and masking.
