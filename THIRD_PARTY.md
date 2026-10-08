# Provenance and third-party components

The uploaded `runpod-comfy-stateless-main.zip` is the source baseline for the migrated task graphs and local integration code. Its private LoRA TXT was excluded without being opened. The source audit and task decisions guided the migration. No license ownership of that supplied code is assumed or reassigned here.

External software is fetched at build time from pinned source commits recorded in `catalog/models.json`: ComfyUI, AusBoss, the SeedVR2 VideoUpscaler integration, MiniMaxH3Mod and MiniMaxRefPack. Their repositories and retained license notices govern those components. Ollama is fetched from the source-pinned release and verified by SHA256. Model weights are separate runtime downloads and remain subject to their own model terms. No model weights are distributed in this repository.

The Qwen Standard HQ graph uses the source's thin fixed-socket adapter around native `TextEncodeQwenImage21`; the sampler/encoder/VAE stack is native. The FLUX Standard HQ graph is reconstructed from native nodes and the source distilled model schedule. H3 Standard HQ is adapted from the source manual native graph with optional LoRA removed. SeedVR2 Standard HQ is the existing integration baseline with independent presentation. These are not claimed to be byte-identical official upstream template files.

Documentation checked for the design:

- RunPod Secrets: https://docs.runpod.io/pods/templates/secrets
- RunPod environment variables: https://docs.runpod.io/pods/templates/environment-variables
- RunPod storage: https://docs.runpod.io/pods/storage/types
- ComfyUI sidebar extensions: https://docs.comfy.org/custom-nodes/js/javascript_sidebar_tabs
- ComfyUI template support: https://docs.comfy.org/interface/features/template
- Docker build secrets: https://docs.docker.com/build/building/secrets/
- GitHub CLI repository creation: https://cli.github.com/manual/gh_repo_create
- GitHub Pages: https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages

The local helper F16 tags were checked on their publisher pages; quality superiority was not benchmarked:

- https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-thinking-fp16
- https://ollama.com/huihui_ai/qwen3-vl-abliterated:32b-instruct-fp16

Source pins preserve the uploaded implementation rather than silently substituting newer model families. Package dependency versions still require build validation; read `docs/architecture.md`.
