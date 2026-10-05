#!/usr/bin/env python3
"""Build-time CPU integration with the pinned RefMod creator and bundle serializer.

Uses a tiny synthetic VAE; checks wrapper/API compatibility, not model quality.
"""
import argparse
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile


def load_pack(name, path):
    spec = importlib.util.spec_from_file_location(name, path / '__init__.py', submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comfy-home', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy_home))
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = ['check_single_person_runtime', '--cpu']
    import torch
    import nodes
    refpack = load_pack('single_person_refmod_probe', args.comfy_home / 'custom_nodes/ComfyUI-MiniMaxH3Mod')
    person = load_pack('single_person_helpers_probe', args.comfy_home / 'custom_nodes/ComfyUI-SinglePersonH3')
    nodes.NODE_CLASS_MAPPINGS.update(refpack.NODE_CLASS_MAPPINGS)
    bundle = importlib.import_module('single_person_refmod_probe.bundle')
    class TinyVAE:
        def __init__(self): self.shapes = []
        def encode(self, image):
            self.shapes.append(tuple(image.shape))
            h, w = image.shape[1:3]
            return torch.ones((1, 24, 1, h // 16, w // 16), dtype=torch.float32)
    images = [torch.full((1, 96, 96, 3), 0.2), torch.full((1, 160, 96, 3), 0.5), torch.full((1, 96, 160, 3), 0.8)]
    refs = person.collect_references(images[0], 'runtime_probe', body_1=images[1], body_2=images[2])
    vae = TinyVAE()
    mods, name, total = person.encode_person_bundle(refs, vae, 256, 131072)
    assert len(mods) == 3 and all(mod.kind == 'image' and strength == 1.0 for mod, strength in mods)
    assert len({(mod.latent_h, mod.latent_w) for mod, _ in mods}) == 3, 'References were cropped to a common canvas'
    assert len(vae.shapes) == 3
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / name)
        bundle.save_bundle(path, name, mods)
        loaded = bundle.load_bundle(path, selection='Visual')
        assert person.saved_roles(loaded) == ['primary_headshot', 'body_1', 'body_2']
        for (original, _), (restored, _) in zip(mods, loaded):
            assert original.kind == restored.kind == 'image'
            assert torch.equal(original.latent, restored.latent)
            assert original.description == restored.description
    print('ONE-PERSON REFMOD CPU INTEGRATION PASS: real RefMod API and serializer; independent image shapes and saved roles. Synthetic VAE, no inference.')

if __name__ == '__main__':
    main()
