"""Element-wise lgamma for z in [0, 1] using Triton inline PTX.

Extracted from CUDA's __internal_lgamma_pos (reference/gamma.ptx).
Two polynomial approximation paths:
  PATH A  z in [0, ~0.897): Horner polynomial, then -log(result)
  PATH B  z in [~0.897, 1]:  polynomial in (1-z)
"""

import torch
import triton
import triton.language as tl



if __name__ == "__main__":
    device = "cuda:0"
    torch.manual_seed(42)

    z = torch.rand(1024, dtype=torch.float64, device=device).clamp(1e-6, 1.0 - 1e-6)
    ref = torch.lgamma(z)
    ours = triton_lgamma(z)

    diff = (ours - ref).abs()
    rel = diff / ref.abs().clamp(min=1e-300)
    print(f"N = {z.numel()}")
    print(f"  max abs err : {diff.max().item():.2e}")
    print(f"  max rel err : {rel.max().item():.2e}")
    print(f"  mean rel err: {rel.mean().item():.2e}")

    # spot checks
    for v in [0.01, 0.1, 0.5, 0.9, 0.99]:
        zt = torch.tensor([v], dtype=torch.float64, device=device)
        print(f"  lgamma({v}) : ours={triton_lgamma(zt).item():.10e}  "
              f"torch={torch.lgamma(zt).item():.10e}")
