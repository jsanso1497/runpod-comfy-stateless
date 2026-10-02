# Stateless ComfyUI + SeedVR2

Normal ComfyUI UI with SeedVR2 7B video restoration preconfigured for the first
1920x1080 low-light/blur restoration test.

The Docker image intentionally does not bake the 15.3 GiB SeedVR2 model into the
image. The model and VAE download to disposable Pod storage at startup and are
verified by SHA-256. This preserves the stateless/no-persistent-volume design.

SeedVR2 source is pinned to commit:

```
4490bd1f482e026674543386bb2a4d176da245b9
```

See `START_HERE.md` for deployment steps.
