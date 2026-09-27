#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys


def detect():
    import torch

    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "none",
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": "none",
        "capability": None,
        "vram_gb": 0.0,
        "sage_importable": False,
        "sage_usable": False,
        "nvcc": shutil.which("nvcc") is not None,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["gpu"] = props.name
        result["capability"] = tuple(torch.cuda.get_device_capability(0))
        result["vram_gb"] = props.total_memory / 1024**3

    try:
        import sageattention  # noqa: F401
        result["sage_importable"] = True
    except Exception:
        pass

    supported = {(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1)}
    result["sage_usable"] = result["sage_importable"] and result["capability"] in supported
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sage-usable", action="store_true")
    args = parser.parse_args()

    try:
        info = detect()
    except Exception as exc:
        if args.sage_usable:
            return 1
        print(f"Hardware check failed: {exc}", file=sys.stderr)
        return 1

    if args.sage_usable:
        return 0 if info["sage_usable"] else 1

    print(f"Python:              {info['python']}")
    print(f"PyTorch:             {info['torch']}")
    print(f"Torch CUDA runtime:  {info['torch_cuda']}")
    print(f"CUDA available:      {info['cuda_available']}")
    print(f"GPU:                 {info['gpu']}")
    print(f"VRAM:                {info['vram_gb']:.1f} GB")
    print(f"Compute capability:  {info['capability']}")
    print(f"nvcc available:      {info['nvcc']}")
    print(f"SageAttention:       {'usable' if info['sage_usable'] else 'not selected'}")

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip().splitlines()[0]
            print(f"NVIDIA driver:       {out}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
