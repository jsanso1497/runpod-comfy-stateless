"""Source-only graph adaptations. No network, model loading or Comfy imports."""
from __future__ import annotations

VERSION = '1.1.1'


def update_k01_defaults(graph, api):
    """Fresh factory graphs default to Turbo; edited user graphs are not migrated."""
    byid = {n['id']: n for n in graph['nodes']}
    standard = api['10']['class_type'] == 'WBKreaStandardCondition'
    loader = byid[8]
    loader['widgets_values'][0] = 'krea2_turbo_bf16.safetensors'
    api['8']['inputs']['checkpoint'] = 'krea2_turbo_bf16.safetensors'
    # Sampling is connected to the loader outputs; keep visible fallback values honest.
    byid[11]['widgets_values'][2:4] = [8, 1.0]
    if standard:
        loader['type'] = 'WBKreaStandardModels'
        loader['title'] = '4. Native Krea generation checkpoint + encoder'
        loader['properties']['Node name for S&R'] = loader['type']
        api['8']['class_type'] = loader['type']
        api['8']['_meta'] = {'title': loader['title']}
        note = ('\n\nNative generation baseline: Turbo, 8 steps / CFG 1. '
                'No Identity Edit adapter and no image-reference conditioning. '
                'Native Raw is experimental: 52 steps / ComfyUI CFG 4.5; start around 1 MP. '
                'This is separate from the Identity Edit removal recipe (Raw 20 / CFG 3). '
                'Custom checkpoints still require an explicit sampling choice.')
    else:
        note = ('\n\nDefault: Turbo, 8 steps / CFG 1 for most edits. For removals, '
                'choose Raw and its 20-step / CFG-3 preset. reference_strength is '
                'the active source/identity fidelity dial in BOTH one- and two-input modes. '
                'Start at 4, reduce when an intentional change is resisted. scene_strength '
                'only affects image 1 with an additional reference. '
                'Grounding 768 is the baseline; 512 can help stubborn edits and 1024 '
                'is an experiment for likeness, not an automatic quality upgrade.')
    note += '\nAudit 1.1.1: use fresh Workbench factory copies for these defaults.'
    current = byid[15]['widgets_values'][0]
    if '\n\nNative generation baseline:' not in current and '\n\nDefault: Turbo,' not in current:
        byid[15]['widgets_values'][0] = current + note
    graph.setdefault('extra', {})['workbench_krea_practices'] = {
        'version': VERSION, 'factory_default': 'Turbo', 'gpu_tested': False,
        'purpose': 'native text-to-image' if standard else 'Identity Edit'}
    return graph, api


def _rebuild_ports(graph):
    byid = {n['id']: n for n in graph['nodes']}
    for n in byid.values():
        for port in n.get('inputs', []):
            port['link'] = None
        for port in n.get('outputs', []):
            port['links'] = []
    for lid, src, out, dst, inp, kind in graph['links']:
        byid[src]['outputs'][out]['links'].append(lid)
        byid[dst]['inputs'][inp]['link'] = lid
    remaining, done = set(byid), []
    while remaining:
        ready = sorted(i for i in remaining if all(l[1] in done for l in graph['links'] if l[3] == i))
        if not ready:
            raise ValueError('Developer graph adaptation introduced a cycle.')
        done.extend(ready)
        remaining.difference_update(ready)
    for order, ident in enumerate(done):
        byid[ident]['order'] = order


def adapt_developer_runtime(graph):
    """sRGB I/O and one target-grid VAE encode per source via the pinned pixel path.

    The upstream original stays archived unchanged. The source_latent socket is
    required by upstream, but its value is superseded by vae + source_image in
    the pixel path. Use the same empty latent as K01, rather than encoding a
    potentially huge full-resolution input that will be discarded and re-encoded.
    """
    byid = {n['id']: n for n in graph['nodes']}
    expected = {72: 'LoadImage', 90: 'LoadImage', 29: 'SaveImage',
                73: 'VAEEncode', 92: 'VAEEncode', 82: 'EmptySD3LatentImage',
                79: 'Krea2EditModelPatch', 83: 'ResolutionSelector'}
    if any(byid.get(i, {}).get('type') != kind for i, kind in expected.items()):
        raise ValueError('Unexpected developer graph layout for the audited runtime adaptation.')
    patch = byid[79]
    inputs = {p['name']: p for p in patch['inputs']}
    if any(inputs.get(k, {}).get('link') is None for k in ('vae','source_image','target_latent')):
        raise ValueError('Cannot omit redundant VAE encodes without the complete pixel path.')
    for ident in (72, 90):
        n = byid[ident]
        n['type'] = 'WBKreaLoadImage'
        n['title'] = 'Scene / source (sRGB)' if ident == 72 else 'Identity reference (sRGB)'
        n['properties'] = {'Node name for S&R': n['type']}
    byid[29]['type'] = 'WBKreaSaveImage'
    byid[29]['title'] = 'Save color-tagged sRGB PNG'
    byid[29]['properties'] = {'Node name for S&R': 'WBKreaSaveImage'}
    byid[29]['outputs'] = []
    removed = {73, 92}
    graph['nodes'] = [n for n in graph['nodes'] if n['id'] not in removed]
    graph['links'] = [l for l in graph['links'] if l[1] not in removed and l[3] not in removed]
    # Feed required placeholder source_latent from target. Pixel images are authoritative.
    slot = next(i for i, p in enumerate(patch['inputs']) if p['name'] == 'source_latent')
    new_id = max(l[0] for l in graph['links']) + 1
    graph['links'].append([new_id, 82, 0, 79, slot, 'LATENT'])
    # Krea patchification is 16px; preserve the selected AR and megapixel budget.
    if len(byid[83]['widgets_values']) != 3:
        raise ValueError('Resolution selector layout changed.')
    byid[83]['widgets_values'][2] = 16
    _rebuild_ports(graph)
    graph['last_link_id'] = new_id
    graph.setdefault('extra', {})['workbench_krea_runtime'] = {
        'version': VERSION, 'color_space': 'sRGB', 'pixel_path_preencode': True,
        'redundant_full_size_vae_encode': False, 'gpu_tested': False}
    return graph
