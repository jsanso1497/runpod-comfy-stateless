# Build fix

The prior build failed while importing SageAttention:

```
ImportError: ... sageattention/_fused...so: undefined symbol ...TensorImpl...
```

That is a compiled-extension/Torch ABI mismatch inherited from the earlier base image.
For the first SeedVR2 restoration test this update removes the inherited SageAttention package
and uses SeedVR2's supported PyTorch SDPA backend. This changes performance optimization only,
not the selected SeedVR2 restoration model or the comparison pipeline.

After the baseline test works, SageAttention can be compiled from source against the exact
Torch/CUDA environment and benchmarked separately.
