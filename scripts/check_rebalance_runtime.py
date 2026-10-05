#!/usr/bin/env python3
"""CPU test of the actual pinned Rebalance node. No model weights or inference."""
import argparse
import importlib.util
import sys
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-home", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy_home))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = ["check_rebalance_runtime", "--cpu"]
    pack = args.comfy_home / "custom_nodes" / "Rebalance-Pack"
    name = "rebalance_runtime_probe"
    spec = importlib.util.spec_from_file_location(name, pack / "__init__.py", submodule_search_locations=[str(pack)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    cls = module.NODE_CLASS_MAPPINGS["ConditioningKrea2Rebalance"]
    weights = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"
    x = torch.ones((1, 2, 30720), dtype=torch.float32)
    mask = torch.tensor([[1, 1]], dtype=torch.int64)
    metadata = {"attention_mask": mask, "preserve_fixture": "reference"}
    source = [[x, metadata]]
    output = cls().main(source, 1.0, weights)[0]
    expected = torch.tensor([float(v) for v in weights.split(",")]).repeat_interleave(2560)
    assert output[0][0].shape == x.shape
    assert torch.allclose(output[0][0], expected.view(1, 1, -1).expand_as(x))
    assert torch.equal(x, torch.ones_like(x)), "The input conditioning must not be mutated"
    assert output[0][1] == metadata
    assert output[0][1]["attention_mask"] is mask, "Reference metadata must be retained"
    assert output[0][1] is not metadata, "Metadata must be copied"
    print("KREA REBALANCE CPU TENSOR CHECK PASS: twelve layer bands, metadata preserved")

if __name__ == "__main__":
    main()
