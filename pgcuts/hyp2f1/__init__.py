"""
GPU-accelerated Gauss hypergeometric function 2F1(-m, b; c; z).

Specialized for non-positive integer ``a = -m`` with z in [0, 1].
Supports full PyTorch broadcasting on b, c, z.
Output dtype matches z dtype; internal computation always in float64.

All kernels accept ``scale`` (default 1.0) — a multiplicative conditioner
applied in fp64 before casting to the output dtype, useful for preventing
overflow/underflow when the output is fp32.

Kernels
-------
hyp2f1            : Pfaff/Kummer CUDA kernel (auto-selects transformation).
fast_hyp2f1       : Kummer-only CUDA kernel (branchless, no warp divergence).
mp_hyp2f1         : Direct series CUDA kernel (mpmath-style, no transformations).
par_hyp2f1        : Parallel Kummer CUDA kernel (8 threads per element).
par_hyp2f1_precomp: Same as par_hyp2f1 with precomputed lgamma prefactor.
triton_hyp2f1     : Triton kernel with associative scan (32 lanes per element).

Autograd
--------
Hyp2F1            : torch.autograd.Function with forward/backward w.r.t. z.
                    Accepts optional scale as 5th argument.
"""

from .cuda_kernels import hyp2f1, fast_hyp2f1, mp_hyp2f1, par_hyp2f1, par_hyp2f1_precomp
from .triton_kernels import triton_hyp2f1
from .autograd import Hyp2F1

__all__ = [
    "hyp2f1", "fast_hyp2f1", "mp_hyp2f1", "par_hyp2f1", "par_hyp2f1_precomp",
    "triton_hyp2f1", "Hyp2F1",
]
