"""
PyTorch autograd wrapper for 2F1(-m, b; c; z) with gradient w.r.t. z.

Derivative identity:
    d/dz 2F1(a, b; c; z) = (a * b / c) * 2F1(a+1, b+1; c+1; z)

For a = -m (non-positive integer), a+1 = -(m-1), so the derivative is
again a terminating 2F1 with one fewer term. When m = 0 the function is
constant (= 1) and the gradient is zero.
"""

import torch
from .cuda_kernels import par_hyp2f1_precomp


def _reduce_broadcast(grad, target_shape):
    """Sum grad over dimensions that were broadcast to match target_shape."""
    # Remove leading dims added by broadcast
    while grad.dim() > len(target_shape):
        grad = grad.sum(0)
    # Sum over dims where target had size 1
    for i, (gs, ts) in enumerate(zip(grad.shape, target_shape)):
        if ts == 1 and gs > 1:
            grad = grad.sum(i, keepdim=True)
    return grad


class Hyp2F1(torch.autograd.Function):
    """Differentiable 2F1(-m, b; c; z) with backward pass w.r.t. z.

    Uses par_hyp2f1_precomp (CUDA, P=8) for both forward and backward.

    Usage::

        result = Hyp2F1.apply(a, b, c, z)
        result.sum().backward()   # computes z.grad

        # With scale to avoid fp32 overflow:
        result = Hyp2F1.apply(a, b, c, z, 1e-10)
    """

    @staticmethod
    def forward(ctx, a, b, c, z):
        a_val = int(a)
        ctx.save_for_backward(z)
        ctx.a_val = a_val
        ctx.b = b
        ctx.c = c
        ctx.z_shape = z.shape
        return par_hyp2f1_precomp(a_val, b, c, z)

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        a_val = ctx.a_val
        b = ctx.b
        c = ctx.c
        m = -a_val

        if m == 0:
            return None, None, None, torch.zeros_like(z)

        # d/dz 2F1(a,b;c;z) = (a*b/c) * 2F1(a+1,b+1;c+1;z)
        # a+1 = -(m-1), still a terminating series
        df = par_hyp2f1_precomp(a_val + 1, b + 1, c + 1, z)

        # Scale factor: a * b / c  (broadcast-safe)
        device, dtype = z.device, z.dtype
        b_t = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=dtype, device=device)
        c_t = c if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=dtype, device=device)
        deriv_scale = float(a_val) * b_t / c_t

        grad_full = grad_output * deriv_scale * df

        # Sum over broadcast dims to match original z shape
        grad_z = _reduce_broadcast(grad_full, ctx.z_shape)
        return None, None, None, grad_z
