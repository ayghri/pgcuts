"""GPU-accelerated Gauss hypergeometric function 2F1.

Kernels
-------
hyp2f1            : Pfaff/Kummer CUDA kernel.
fast_hyp2f1       : Kummer-only CUDA kernel.
mp_hyp2f1         : Direct series CUDA kernel.
par_hyp2f1        : Parallel Kummer CUDA kernel.
par_hyp2f1_precomp: par_hyp2f1 with precomputed lgamma.
triton_hyp2f1     : Triton kernel with assoc scan.

Autograd
--------
Hyp2F1            : torch.autograd.Function.
"""

from .cuda_kernels import (
    hyp2f1,
    fast_hyp2f1,
    mp_hyp2f1,
    par_hyp2f1,
    par_hyp2f1_precomp,
)
from .triton_kernels import triton_hyp2f1
from .autograd import Hyp2F1

__all__ = [
    "hyp2f1",
    "fast_hyp2f1",
    "mp_hyp2f1",
    "par_hyp2f1",
    "par_hyp2f1_precomp",
    "triton_hyp2f1",
    "Hyp2F1",
]
