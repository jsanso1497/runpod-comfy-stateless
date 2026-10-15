"""Add two Qwen tasks without changing existing recipes or enhancer backends."""
PE_FILE = 'qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors'
STOCK_PATH = 'workflows/qwen/QNATIVE/Qwen21_Comfy_Template_BF16.json'
OVERLAY_PATH = 'workflows/qwen/QOVERLAY/Qwen21_Full_Resolution_Overlay_BF16.json'


def apply(name, value):
    if name == 'models.json':
        for model in value['models']:
            if model['id'] == 'qwen':
                local_nodes = model.setdefault('local_nodes', [])
                for node in ('qwen_photo', 'qwen_overlay'):
                    if node not in local_nodes:
                        local_nodes.append(node)
                toolbox = model.setdefault('toolbox', [])
                if 'qwen_stock_pe' not in toolbox:
                    toolbox.append('qwen_stock_pe')
    elif name == 'tasks.json':
        rows = [
            ('QNATIVE', 'ComfyUI Qwen Edit: stock template, BF16 models', STOCK_PATH,
             'Official edit subgraph and prompt enhancer unchanged. Main diffusion and image encoder use BF16. '
             'No Workbench enhancer, extra LoRA, alternate sampler or custom encode node.', []),
            ('QOVERLAY', 'Full-resolution edit overlay: manual/SAM mask + face protection', OVERLAY_PATH,
             'Edit a bounded crop with the same native subgraph. A second SAM selection on the ORIGINAL '
             'protects a face or described element. Review masks first; export a full-canvas transparent '
             'edit PNG, composite and alpha mask. Untouched pixels are copied.', ['sam']),
        ]
        known = {row['id'] for row in value['tasks']}
        for ident, title, path, description, extra in rows:
            if ident in known:
                raise ValueError('Duplicate stock-Qwen task: ' + ident)
            value['tasks'].append({'id': ident, 'family': 'qwen', 'workspaces': ['qwen'],
                'category': 'tasks', 'title': title, 'description': description,
                'requires': ['qwen', 'qwen_stock_pe'] + extra,
                'status': 'CPU geometry tested; live ComfyUI/GPU validation pending',
                'sources': ['https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1'],
                'variants': [{'label': 'BF16 / native defaults', 'file': path,
                              'source': 'Pinned official ComfyUI template; main model precision upgraded'}],
                'required_nodes': [] if ident == 'QNATIVE' else
                    ['Q21PhotoMask', 'Q21PhotoCrop', 'WBOverlayProtection', 'WBOverlayPrepare', 'WBOverlayExport']})
    elif name == 'assets.json':
        destination = 'text_encoders/' + PE_FILE
        if not any(row['destination'] == destination for row in value['assets']):
            value['assets'].append({'id': 'qwen-stock-pe-i2i-int8', 'family': 'qwen',
                'group': 'qwen_stock_pe', 'repo_id': 'Comfy-Org/Qwen-Image-2.1',
                'filename': destination, 'destination': destination, 'revision': 'main',
                'sha256': '32707d01b427e488af252b95c551989aad59f9fec611a694f5db6bde7f0f1f6c', 'estimated_bytes': 9500000000,
                'integrity_source': 'Official Comfy-Org file; provider integrity is checked by the existing downloader',
                'verification': 'Weights not downloaded or GPU-tested in this change'})
    return value
