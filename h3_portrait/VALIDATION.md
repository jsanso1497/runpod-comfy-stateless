# H3 Portrait 1.5 validation

See `../VALIDATION.md` for executed commands, results, and limitations.

Release-specific checks cover:

- role-aware H3 video routing;
- still safe-swap routing where pose/camera, expression, and scene guides are text-only;
- MiniMax H3 Ref2VA five-frame single-image recipe and one-frame selection;
- no separate still-image model assets in Lite;
- corrected ComfyUI seed/widget serialization;
- exact exported-video final-frame PNG;
- Full/Lite workflow installation and source/version verification.

GPU H3 inference is not performed by the local test suite.
