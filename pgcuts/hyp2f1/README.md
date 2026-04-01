# hyp2f1 — GPU-accelerated Gauss hypergeometric function

CUDA/Triton kernels computing `2F1(-m, b; c; z)` for **non-positive integers `a = -m`** with `z ∈ [0, 1]`.

**Features:** Full PyTorch broadcasting (`b`, `c`, `z`), internal `float64` precision (output matches `z`), N-D stride-based access.

## Installation & API

Requires PyTorch and Triton. CUDA kernels JIT-compile via `torch.utils.cpp_extension` (cached locally).

  **Signature:** `fn(a, b, c, z, scale=1.0)`
- **`a`**: int ≤ 0 (i.e., `a = -m`)
- **`b`, `c`**: float or Tensor
- **`z`**: CUDA Tensor, `z ∈ [0, 1]`
- **`scale`**: float (default `1.0`). Applied in `fp64` before casting to output dtype. Use to prevent `fp32` overflow.

| Kernel | Backend | Strategy | Best for / Notes |
|---|---|---|---|
| `par_hyp2f1_precomp` | CUDA | 8-thread Kummer + prefactor | Fastest, but only works on NVIDIA|
| `par_hyp2f1` | CUDA | 8-thread Kummer | Large `m` |
| `triton_hyp2f1` | Triton | 32-lane associative scan | Portable, no JIT cache |
| `hyp2f1` | CUDA | Pfaff/Kummer auto-select | General use, small `m` |
| `fast_hyp2f1` | CUDA | Kummer-only, branchless | No warp divergence |
| `mp_hyp2f1` | CUDA | Direct series | Small `m` only (**unstable** `m > ~50`) |

## Usage & Autograd

`Hyp2F1` supports `torch.autograd` w.r.t `z` via `d/dz 2F1(a,b;c;z) = (ab/c) 2F1(a+1,b+1;c+1;z)`. Internally uses `par_hyp2f1_precomp`.

```python
import torch
from hyp2f1 import par_hyp2f1_precomp, Hyp2F1

# 1. 1D & Broadcasting (c: 1x128, z: 1000x1 -> out: 1000x128)
z = torch.rand(1000, 1, dtype=torch.float64, device="cuda")
c = torch.rand(1, 128, dtype=torch.float64, device="cuda") + 1.0
res = par_hyp2f1_precomp(-2048, 1.5, c, z) 

# 2. Autograd & fp32 Scale Conditioning (prevents fp32 overflow)
z_fp32 = torch.rand(1000, dtype=torch.float32, device="cuda", requires_grad=True)
out = Hyp2F1.apply(-8192, 1.0, 1.5, z_fp32, 1e-30) 
out.sum().backward() # z_fp32.grad is populated
```

## Benchmarks & Numerics (RTX 3090, fp64)

- Kummer transformation (`2F1 = prefactor · 2F1(-m, b; C'; 1-z)`) ensures stability near `z=1`. Parallel kernels use associative tree reduction `(carry, sum)`.
- Kummer-based GPU kernels match `scipy` to ~11-digits (`m=2048`) and produce **0 NaNs** (unlike `scipy` which produces all NaNs for `m≥16384` near `z=1`).
-  `mp_hyp2f1` (no Kummer) is unstable >50.

**Execution Time (ms)**
| Setup | Kernel | m=10 | m=100 | m=512 | m=2048 | m=8192 |
|---|---|---|---|---|---|---|
| **1D** `z(10k,)` | `par_hyp2f1_precomp` | 0.15 | 0.22 | 0.48 | 1.51 | 5.95 |
| `scalar b,c` | `par_hyp2f1` | 0.11 | 0.18 | 0.44 | 1.48 | 5.62 |
| | `triton_hyp2f1` | 0.22 | 0.28 | 0.57 | 1.72 | 6.18 |
| | `fast_hyp2f1` | 0.07 | 0.19 | 0.75 | 2.80 | 11.06 |
| | `hyp2f1` | 0.10 | 0.38 | 1.67 | 6.33 | 22.21 |
| | `mp_hyp2f1` | 0.06 | 0.18 | 0.73 | 2.78 | 11.00 |
| | *scipy (CPU)* | *0.5* | *2.2* | *10.0* | *~40* | *~160* |
| **Broadcast** | `par_hyp2f1_precomp` | 0.49 | 1.35 | 4.54 | 17.30 | 68.39 |
| `128K out` | `par_hyp2f1` | 1.11 | 1.99 | 5.15 | 17.93 | 68.99 |
| | `triton_hyp2f1` | 1.01 | 1.80 | 5.45 | 17.69 | 68.22 |
| | `fast_hyp2f1` | 0.28 | 1.13 | 5.03 | 19.53 | 77.55 |
| | `hyp2f1` | 0.28 | 1.19 | 5.27 | 19.48 | 77.53 |
| | `mp_hyp2f1` | 0.13 | 0.98 | 4.88 | 19.38 | 77.36 |
| | *scipy (CPU)* | *-* | *-* | *~1300* | *-* | *-* |

*`Hyp2F1.apply` (Autograd Fwd+Bwd, m=[10, 100, 512, 2048]): 0.48ms, 0.61ms, 1.31ms, 3.73ms*
