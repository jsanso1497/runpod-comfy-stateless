# Validation: Qwen Image 2.1 Photo Edit 2.0

## Completed

- 39 CPU unit tests passed. One test includes 81 area/shape/alignment combinations.
- Two separate full-size CPU stress cases passed: 7001 x 4003 and 4003 x 7001 source canvases, each with a 2500 x 2500 editable region and a protected 32 x 32 hole.
- The large cases sampled no model. A synthetic generated patch was fitted and stitched into the original source. Every zero-mask pixel was checked for exact tensor equality, in memory-efficient row blocks.
- Both large cases retained exact source/output dimensions, used a 2048 x 2048 working crop, and reported zero changed channels outside the effective mask.
- Real small-image PNG export tests passed, including rounded 8-bit output, the sRGB profile, mask export, and verification sidecar.
- Unit tests cover a missing manual mask, shape/value errors, SAM no-detection errors, correction after expansion, reference order and missing tags, optional LoRAs, lazy preview requests, RGBA compositing, no source/work tensor aliasing, and structural safetensors checks.
- Both UI workflow graphs were generated from the same definitions as their API forms. Connection IDs, types, inputs, and widget settings were checked.
- Python compilation, installer/startup shell syntax, and browser-script JavaScript syntax passed.
- Native Qwen, cache, SAM, and NodeOutput interfaces were read from the ComfyUI commit recorded in SOURCES.json.

## Not completed

Actual SAM/Qwen/LoRA inference calls are mocked in unit tests. No GPU is available in this execution environment. No A40 memory, latency, output identity, or segmentation accuracy benchmark was run. The real ComfyUI web interface and its image-upload/DOM widgets were not executed in a browser. Syntax validation is not a browser integration test.

The optional standalone Docker image was not built or published. Its build process is configured to run dependency checks, interface checks, and the included CPU tests. That configuration does not mean those build steps have already run on GitHub Actions.

No live GitHub files or RunPod configuration were changed. Existing model files are not included in either package.

## Reproduce

```bash
python -m unittest discover -s tests -v
python tests/large_canvas_check.py
python -m compileall -q node download_models.py preflight.py make_workflows.py
bash -n install.sh
node --check node/web/review.js
```

Large-canvas testing may require several GB of CPU RAM. Run outside an active model job. GPU tests should begin with mask preview, followed by one reference and the BF16 workflow. Reduce cache residency or choose the explicit INT8 variant only when required.

Detailed outputs: TEST_RESULTS.txt and LARGE_CANVAS_VALIDATION.json.
