# GitHub overlay validation and limitations

- Source: user-provided root ZIP `New Compressed (zipped) Folder(5).zip`.
- New isolated image build: `.github/workflows/build-qwen21-photo-edit.yml`.
- Dedicated package build: `qwen21_photo_edit/Dockerfile` and `start.sh`, using the official pinned ComfyUI core and existing PyTorch base image.
- Legacy source files: unchanged except root `SOURCE_SNAPSHOT.json` to re-inventory the exact uploaded files plus overlay.
- Qwen workflows: BF16 and INT8, both with manual/SAM mask toggle, 2K-area crop, multi-reference conditioning, optional LoRA slots, lazy mask preview, and protected stitch.
- Validation: run `python qwen21_photo_edit/ci_validate.py`, `python -m unittest discover -s qwen21_photo_edit/tests -p 'test_*.py' -q`, and `python tools/verify_snapshot.py` after applying the overlay.
- Docker/GPU: not available in the packaging environment; validate separately with the new GitHub Action and a real RunPod GPU.
