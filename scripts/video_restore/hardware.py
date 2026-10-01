"""Small real CUDA/Sage kernel tests, run in an isolated process."""
import json
import os
import sys
import time


def inspect_hardware():
    import torch
    import psutil
    report = {'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
              'ram_gib': round(psutil.virtual_memory().total / 2**30, 1),
              'cuda_available': torch.cuda.is_available(), 'attention': 'sdpa',
              'sage_test': 'not run', 'warnings': []}
    if not report['cuda_available']:
        raise RuntimeError('No usable CUDA GPU. Deploy this image on a NVIDIA GPU Pod.')
    device = torch.cuda.get_device_properties(0)
    report.update(gpu=device.name, vram_gib=round(device.total_memory / 2**30, 1),
                  compute_capability=f'{device.major}.{device.minor}')
    if device.major < 8:
        raise RuntimeError('Use an Ampere or newer NVIDIA GPU for this preset.')
    if report['ram_gib'] < 55:
        report['warnings'].append('At least 64 GB system RAM is recommended; 128 GB is preferable for 4K.')
    if report['vram_gib'] < 40:
        report['warnings'].append('Low-VRAM fallback will be slower. A 48-80 GB GPU is recommended for the 7B 4K test.')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q = torch.randn(1, 4, 64, 64, device='cuda', dtype=dtype)
    ref = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    if not torch.isfinite(ref).all():
        raise RuntimeError('PyTorch SDPA produced non-finite results in the CUDA smoke test.')
    requested = os.environ.get('RESTORE_ATTENTION', 'sdpa').lower()
    if requested not in {'auto', 'sdpa', 'sageattn_2'}:
        raise RuntimeError('RESTORE_ATTENTION must be auto, sdpa, or sageattn_2.')
    if requested != 'sdpa':
        try:
            from sageattention import sageattn_varlen
            errors = []
            for head_dim in (64, 128):
                q, k, v = [torch.randn(96, 4, head_dim, device='cuda', dtype=dtype) for _ in range(3)]
                cu = torch.tensor([0, 64, 96], device='cuda', dtype=torch.int32)
                result = sageattn_varlen(q, k, v, cu, cu, 64, 64, is_causal=False)
                references = []
                for a, b in [(0, 64), (64, 96)]:
                    part = torch.nn.functional.scaled_dot_product_attention(
                        q[a:b].transpose(0, 1).unsqueeze(0),
                        k[a:b].transpose(0, 1).unsqueeze(0),
                        v[a:b].transpose(0, 1).unsqueeze(0))
                    references.append(part.squeeze(0).transpose(0, 1))
                ref = torch.cat(references)
                torch.cuda.synchronize()
                if result.shape != ref.shape or not torch.isfinite(result).all():
                    raise RuntimeError('SageAttention returned invalid output.')
                error = ((result.float() - ref.float()).square().mean().sqrt() /
                         ref.float().square().mean().sqrt().clamp_min(1e-6)).item()
                errors.append(round(error, 6))
                if error > 0.05:
                    raise RuntimeError(f'SageAttention relative RMS difference {error:.3f} exceeded 0.05.')
            report.update(attention='sageattn_2', sage_test='passed actual varlen CUDA kernels',
                          sage_relative_rms=errors)
        except Exception as exc:
            report['sage_test'] = f'failed; using SDPA: {type(exc).__name__}: {exc}'
            report['warnings'].append(report['sage_test'])
            if requested == 'sageattn_2':
                raise RuntimeError(report['sage_test']) from exc
    report['note'] = 'Kernel tests are not a full model benchmark or a visual-fidelity guarantee.'
    return report

if __name__ == '__main__':
    try:
        print('HARDWARE_JSON=' + json.dumps(inspect_hardware()))
    except Exception as exc:
        print('HARDWARE_JSON=' + json.dumps({'error': str(exc)}))
        sys.exit(1)
