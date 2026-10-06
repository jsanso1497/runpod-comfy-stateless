#!/usr/bin/env python3
"""CPU tensor check of the installed Krea Rebalance implementation.

The full Rebalance-Pack __init__.py registers ComfyUI HTTP routes. It may only
be imported after a real PromptServer has been created. This earlier, standalone
check loads the real krea2.py and its relative dependencies in an isolated package
namespace without executing that server-dependent __init__.py.

The later check_workflows.py build-smoke step still starts real ComfyUI and
imports the complete node pack through its normal startup lifecycle. This script
neither replaces PromptServer nor suppresses node or dependency import failures.
"""
from __future__ import annotations

import argparse
import importlib
from importlib.machinery import ModuleSpec
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator


PROBE_PACKAGE = "_runpod_krea_rebalance_tensor_probe"
WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"


@contextmanager
def load_krea_module(pack: Path) -> Iterator[ModuleType]:
    """Load installed source files, not a mock scaler or a copied implementation."""
    pack = pack.resolve(strict=True)
    for filename in ("__init__.py", "krea2.py", "conditioning_rebalance.py"):
        if not (pack / filename).is_file():
            raise FileNotFoundError(f"Missing installed Rebalance file: {pack / filename}")

    prefix = PROBE_PACKAGE + "."
    if any(name == PROBE_PACKAGE or name.startswith(prefix) for name in sys.modules):
        raise RuntimeError("The isolated Rebalance probe namespace is already in use")

    # Provide package identity so upstream relative imports resolve normally.
    # Do NOT execute Rebalance-Pack/__init__.py in this pre-server process.
    package = ModuleType(PROBE_PACKAGE)
    package.__package__ = PROBE_PACKAGE
    package.__path__ = [str(pack)]
    package.__spec__ = ModuleSpec(PROBE_PACKAGE, loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules[PROBE_PACKAGE] = package
    try:
        importlib.invalidate_caches()
        module = importlib.import_module(PROBE_PACKAGE + ".krea2")
        expected_file = (pack / "krea2.py").resolve()
        if Path(module.__file__).resolve() != expected_file:
            raise RuntimeError("The tensor check did not load the installed krea2.py")
        yield module
    finally:
        for name in list(sys.modules):
            if name == PROBE_PACKAGE or name.startswith(prefix):
                del sys.modules[name]


def check_tensors(module: ModuleType) -> None:
    """Keep the layer-scaling, input-integrity and reference-metadata checks."""
    import torch

    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    cls = mappings.get("ConditioningKrea2Rebalance")
    if cls is None:
        raise RuntimeError("The installed Krea module is missing ConditioningKrea2Rebalance")

    required = cls.INPUT_TYPES().get("required", {})
    if not {"conditioning", "multiplier", "per_layer_weights"}.issubset(required):
        raise RuntimeError("The installed Krea Rebalance input schema has changed")

    x = torch.ones((1, 2, 30720), dtype=torch.float32, device="cpu")
    mask = torch.tensor([[1, 1]], dtype=torch.int64, device="cpu")
    metadata = {"attention_mask": mask, "preserve_fixture": "reference"}
    source = [[x, metadata]]
    output = cls().main(source, 1.0, WEIGHTS)[0]
    expected = torch.tensor(
        [float(value) for value in WEIGHTS.split(",")],
        dtype=torch.float32,
        device="cpu",
    ).repeat_interleave(2560)

    # Explicit failures also work when Python is started with optimization enabled.
    if output[0][0].shape != x.shape:
        raise AssertionError("Rebalance changed the conditioning tensor shape")
    torch.testing.assert_close(
        output[0][0], expected.view(1, 1, -1).expand_as(x), rtol=1e-5, atol=1e-6
    )
    if not torch.equal(x, torch.ones_like(x)):
        raise AssertionError("The input conditioning must not be mutated")
    if output[0][1].keys() != metadata.keys():
        raise AssertionError("Reference metadata keys must be retained")
    if output[0][1]["preserve_fixture"] != "reference":
        raise AssertionError("Reference metadata values must be retained")
    if output[0][1]["attention_mask"] is not mask:
        raise AssertionError("The original attention-mask object must be retained")
    if output[0][1] is metadata:
        raise AssertionError("The metadata dictionary must be copied")
    if not torch.equal(mask, torch.tensor([[1, 1]], dtype=mask.dtype)):
        raise AssertionError("The input attention mask must not be mutated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-home", type=Path, required=True)
    args = parser.parse_args()
    home = args.comfy_home.resolve(strict=True)
    original_argv = sys.argv[:]
    original_path = sys.path[:]
    try:
        sys.path.insert(0, str(home))
        sys.argv = ["check_rebalance_runtime", "--cpu"]
        import comfy.options
        comfy.options.enable_args_parsing()
        with load_krea_module(home / "custom_nodes" / "Rebalance-Pack") as module:
            check_tensors(module)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path
    print(
        "KREA REBALANCE CPU TENSOR CHECK PASS: twelve layer bands, metadata preserved; "
        "server-dependent pack initialization is reserved for the normal ComfyUI smoke test",
        flush=True,
    )


if __name__ == "__main__":
    main()
