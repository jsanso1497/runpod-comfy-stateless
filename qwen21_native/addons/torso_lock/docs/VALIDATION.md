# Torso Lock validation

## Executed locally

- 49 addon unit tests passed.
- 148 combined tests passed: the original suite's 99 plus 49 addon tests.
- All four editable UI/API graph pairs passed link, widget and configuration checks.
- Python compilation and patched startup shell syntax passed.
- Applying the GitHub integration twice leaves the three patched files identical.
- Original model allowlist, pinned upstream revisions and GitHub Action are byte-identical.

The tests cover restoration of protected source pixels despite a deliberately
corrupted full-frame candidate; zero changes outside the edit mask; inward-only
feathering; overlapping protection; soft masks; invalid masks; same-source checks;
secondary-pass original restoration; optional reference combinations and automatic
renumbering; native encoder argument delegation; final/audit/mask export; and
additive installation preserving existing files and user-edited workflows.

The PNG-export test checks protected output bytes against the same original RGB
rendered to 8-bit. It does not claim preservation of a source JPEG's compressed
bytes, high-bit-depth source encoding, EXIF or ICC metadata.

## Not executed

- No Qwen model or VAE weights were loaded.
- No GPU generation, likeness assessment or anatomical correspondence test was run.
- No Docker image was built or pushed.
- No running ComfyUI frontend or live `/object_info` endpoint was available.
- Native encoding is tested by delegation to a stub, not by sampling.
- No claim of an automatic multi-view registration or exact skin reconstruction is made.

The addon includes a live schema validator to run after installation. The GitHub
overlay integrates that check into actual Pod startup, not BuildKit.

Test environment versions and captured outputs are in `validation_logs/`.
