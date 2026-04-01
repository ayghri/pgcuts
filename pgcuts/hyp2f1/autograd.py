"""PyTorch autograd wrapper for 2F1(-m, b; c; z).

Derivative identity:
    d/dz 2F1(a, b; c; z) = (a*b/c) * 2F1(a+1, b+1; c+1; z)

Backend selection:
    - Default: Triton (portable across GPU vendors)
    - Fallback: CUDA (NVIDIA only, JIT-compiled)

Set HYP2F1_BACKEND='cuda' to force the CUDA backend.
"""

import os
import torch

_BACKEND = os.environ.get(
    "HYP2F1_BACKEND", "triton"
).lower()

if _BACKEND == "cuda":
    from .cuda_kernels import (
        par_hyp2f1_precomp as _hyp2f1_kernel,
    )
else:
    from .triton_kernels import (
        triton_hyp2f1 as _hyp2f1_kernel,
    )


def _reduce_broadcast(grad, target_shape):
    """Sum grad over broadcast dimensions."""
    while grad.dim() > len(target_shape):
        grad = grad.sum(0)
    for i, (gs, ts) in enumerate(
        zip(grad.shape, target_shape)
    ):
        if ts == 1 and gs > 1:
            grad = grad.sum(i, keepdim=True)
    return grad


class Hyp2F1(torch.autograd.Function):
    """Differentiable 2F1(-m, b; c; z) with backward.

    Uses Triton backend by default (portable).

    Usage::

        result = Hyp2F1.apply(a, b, c, z)
        result.sum().backward()   # computes z.grad
    """

    @staticmethod
    def forward(ctx, a, b, c, z):
        """Compute 2F1(a, b; c; z) forward pass."""
        a_val = int(a)
        ctx.save_for_backward(z)
        ctx.a_val = a_val
        ctx.b = b
        ctx.c = c
        ctx.z_shape = z.shape
        return _hyp2f1_kernel(a_val, b, c, z)

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradient w.r.t. z."""
        (z,) = ctx.saved_tensors
        a_val = ctx.a_val
        b = ctx.b
        c = ctx.c
        m = -a_val

        if m == 0:
            return None, None, None, torch.zeros_like(z)

        # d/dz 2F1(a,b;c;z) = (a*b/c) * 2F1(a+1,b+1;c+1;z)
        df = _hyp2f1_kernel(a_val + 1, b + 1, c + 1, z)

        device, dtype = z.device, z.dtype
        b_t = (
            b
            if isinstance(b, torch.Tensor)
            else torch.tensor(
                b, dtype=dtype, device=device
            )
        )
        c_t = (
            c
            if isinstance(c, torch.Tensor)
            else torch.tensor(
                c, dtype=dtype, device=device
            )
        )
        deriv_scale = a_val * b_t / c_t

        grad_full = grad_output * deriv_scale * df
        grad_z = _reduce_broadcast(
            grad_full, ctx.z_shape
        )
        return None, None, None, grad_z
