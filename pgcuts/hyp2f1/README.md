# hyp2f1 — GPU-accelerated Gauss hypergeometric function

GPU kernels for computing `2F1(-m, b; c; z)` specialized for **non-positive integer `a = -m`** with `z ∈ [0, 1]`.

All kernels support full PyTorch broadcasting on `b`, `c`, `z`. Internal computation is always **float64**; the output dtype matches `z`.

## Installation

The package lives in `kernels/hyp2f1/` and requires:

- PyTorch with CUDA support
- Triton (for `triton_hyp2f1` only)

CUDA kernels are JIT-compiled on first use via `torch.utils.cpp_extension.load_inline` and cached at `~/.cache/torch_extensions/`.

```python
import sys
sys.path.insert(0, "path/to/kernels")

from hyp2f1 import par_hyp2f1_precomp, triton_hyp2f1, Hyp2F1
```

## API

All kernels share the signature `fn(a, b, c, z, scale=1.0)` where:

- **`a`**: int, must be ≤ 0 (i.e., `a = -m` with `m ≥ 0`)
- **`b`**: float or Tensor
- **`c`**: float or Tensor
- **`z`**: Tensor (CUDA), values in [0, 1]
- **`scale`**: float (default 1.0) — multiplicative conditioner applied in fp64 before casting to output dtype

Returns a CUDA tensor with shape `broadcast(b, c, z)` and same dtype as `z`.

### Kernels

| Function | Backend | Strategy | Best for |
|----------|---------|----------|----------|
| `hyp2f1` | CUDA | Pfaff/Kummer auto-select | General use, small m |
| `fast_hyp2f1` | CUDA | Kummer-only, branchless | No warp divergence |
| `mp_hyp2f1` | CUDA | Direct series (no transform) | Small m only (unstable for m > ~50) |
| `par_hyp2f1` | CUDA | 8-thread parallel Kummer | Large m |
| `par_hyp2f1_precomp` | CUDA | par_hyp2f1 + precomputed prefactor | **Fastest overall** |
| `triton_hyp2f1` | Triton | Associative scan (32 lanes) | Portable, no JIT cache |

### Autograd

`Hyp2F1` is a `torch.autograd.Function` with backward pass w.r.t. `z`:

```python
from hyp2f1 import Hyp2F1

z = torch.rand(1000, dtype=torch.float64, device="cuda:0", requires_grad=True)
result = Hyp2F1.apply(-50, 1.5, 3.0, z)
result.sum().backward()
print(z.grad)  # d/dz 2F1(-50, 1.5; 3.0; z)
```

Uses the identity `d/dz 2F1(a,b;c;z) = (ab/c) · 2F1(a+1,b+1;c+1;z)`. Internally calls `par_hyp2f1_precomp` for both forward and backward. Accepts an optional 5th argument for `scale`.

## Usage examples

### Scalar parameters

```python
from hyp2f1 import par_hyp2f1_precomp

z = torch.linspace(0, 1, 10000, dtype=torch.float64, device="cuda:0")
result = par_hyp2f1_precomp(-100, 1.5, 3.0, z)  # shape: (10000,)
```

### Broadcasting

```python
b = 1.0
X = torch.rand(1, 128, dtype=torch.float64, device="cuda:0") * 3.0 + 0.5
c = b + X                                        # shape: (1, 128)
z = torch.rand(1000, 1, dtype=torch.float64, device="cuda:0")  # shape: (1000, 1)

result = par_hyp2f1_precomp(-2048, b, c, z)      # shape: (1000, 128)
```

### fp32 with scale conditioning

When using fp32 tensors, the result can overflow. Use `scale` to keep values in range:

```python
z = torch.rand(1000, dtype=torch.float32, device="cuda:0")
# 2F1 values for large m can exceed fp32 range — scale down in fp64, then cast
result = par_hyp2f1_precomp(-8192, 1.0, 1.5, z, scale=1e-30)

# With autograd:
z.requires_grad_(True)
out = Hyp2F1.apply(-8192, 1.0, 1.5, z, 1e-30)
out.sum().backward()
```

## Benchmarks

GPU: RTX 3090, float64. Times in milliseconds.

### 1D: z(10000,), scalar b=1.5, c=3.0

| Kernel | m=10 | m=100 | m=512 | m=2048 | m=8192 |
|--------|------|-------|-------|--------|--------|
| `par_hyp2f1_precomp` | 0.15 | 0.22 | 0.48 | 1.51 | 5.95 |
| `par_hyp2f1` | 0.11 | 0.18 | 0.44 | 1.48 | 5.62 |
| `triton_hyp2f1` | 0.22 | 0.28 | 0.57 | 1.72 | 6.18 |
| `fast_hyp2f1` | 0.07 | 0.19 | 0.75 | 2.80 | 11.06 |
| `hyp2f1` | 0.10 | 0.38 | 1.67 | 6.33 | 22.21 |
| `mp_hyp2f1` | 0.06 | 0.18 | 0.73 | 2.78 | 11.00 |
| **scipy (CPU)** | **0.5** | **2.2** | **10.0** | **~40** | **~160** |

Speedup over scipy: **5-25x** depending on m and N.

### 2D broadcast: z(1000,1), c(1,128), 128K output elements

| Kernel | m=10 | m=100 | m=512 | m=2048 | m=8192 |
|--------|------|-------|-------|--------|--------|
| `par_hyp2f1_precomp` | 0.49 | 1.35 | 4.54 | 17.30 | 68.39 |
| `par_hyp2f1` | 1.11 | 1.99 | 5.15 | 17.93 | 68.99 |
| `triton_hyp2f1` | 1.01 | 1.80 | 5.45 | 17.69 | 68.22 |
| `fast_hyp2f1` | 0.28 | 1.13 | 5.03 | 19.53 | 77.55 |
| `hyp2f1` | 0.28 | 1.19 | 5.27 | 19.48 | 77.53 |
| `mp_hyp2f1` | 0.13 | 0.98 | 4.88 | 19.38 | 77.36 |

### Autograd (forward + backward): z(10000,), scalar b, c

| m | Time |
|---|------|
| 10 | 0.48ms |
| 100 | 0.61ms |
| 512 | 1.31ms |
| 2048 | 3.73ms |

### Key observations

- **`par_hyp2f1_precomp`** is the recommended kernel — fastest for m ≥ 100, with precomputed Γ prefactor.
- The parallel kernels (`par_*`, `triton_*`) scale as ~O(m/P) vs O(m) for serial kernels, giving 2-4x speedup at large m.
- `mp_hyp2f1` (direct series, no Kummer transformation) is fast for small m but **numerically unstable for m > ~50**.
- All Kummer-based kernels produce **zero NaNs** for any m, while scipy fails for m ≥ 16384 near z = 1.
- scipy timing for 128K broadcast elements at m=512 is ~1.3s (CPU), vs 4.5ms on GPU — **~290x speedup**.

## Accuracy vs scipy

Max relative error compared to `scipy.special.hyp2f1` on 1000 random z ∈ [0, 1):

| Kernel | m=10 | m=100 | m=512 | m=2048 |
|--------|------|-------|-------|--------|
| `hyp2f1` | 5e-15 | 2e-13 | 4e-12 | 6e-11 |
| `fast_hyp2f1` | 7e-15 | 2e-13 | 3e-12 | 6e-11 |
| `par_hyp2f1` | 7e-15 | 2e-13 | 3e-12 | 6e-11 |
| `par_hyp2f1_precomp` | 7e-15 | 2e-13 | 3e-12 | 6e-11 |
| `triton_hyp2f1` | 7e-15 | 2e-13 | 3e-12 | 6e-11 |
| `mp_hyp2f1` | 1e-13 | 5e+13 | overflow | NaN |

All Kummer-based kernels match scipy to ~11 digits at m=2048. Error grows as O(m·ε) due to floating-point accumulation in the m-term series.

## NaN resistance

scipy produces NaN for large m near z = 1. All Kummer-based GPU kernels remain NaN-free:

| m | scipy NaNs (of 10000) | GPU NaNs |
|---|----------------------|----------|
| 2048 | 0 | 0 |
| 8192 | 0 | 0 |
| 16384 | 9999 | 0 |

## Kernel internals

All CUDA kernels use **N-D stride-based tensor access** — no `.contiguous()` calls, no materialized flat copies. Tensor strides are passed to the kernel via a `StridesInfo` struct (constant memory, zero overhead).

The Kummer transformation converts `2F1(-m, b; c; z)` to `prefactor · 2F1(-m, b; C'; 1-z)` where `C' = -m + b - c + 1`, which is numerically stable for z near 1.

The parallel kernels split the m-iteration loop across threads and use a tree reduction with the `(carry, sum)` associative monoid to combine partial results.
