import numpy as np
import torch
from torch.autograd import Function
from typing import Tuple
from scipy.special import hyp2f1 as sp_hyp2f1

_EPS_Z = 1e-12  # keep z away from {0,1} if you know you stay in that domain


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _from_numpy(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(
        device=device, dtype=dtype
    )


class Hyp2F1(Function):
    """
    Torch autograd wrapper for SciPy's 2F1(a,b;c;z).
    - Gradients are implemented w.r.t. z only:
        d/dz 2F1(a,b;c;z) = (a*b/c) * 2F1(a+1, b+1; c+1; z)
    - If any of a,b,c require grad, we raise (to avoid silent wrong grads).
    - Works with broadcasting across a,b,c,z.
    - Runs SciPy on CPU; safely moves tensors between devices.
    """

    @staticmethod
    def forward(
        ctx, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        # Guard: do not allow grads for a,b,c (unsupported)
        if a.requires_grad or b.requires_grad or c.requires_grad:
            raise RuntimeError(
                "Hyp2F1 backward supports gradients w.r.t. z only (a,b,c must not require grad)."
            )

        # Save for backward
        ctx.save_for_backward(a.detach(), b.detach(), c.detach(), z.detach())

        # Move to numpy CPU and broadcast
        a_np, b_np, c_np, z_np = map(_to_numpy, (a, b, c, z))
        a_np, b_np, c_np, z_np = np.broadcast_arrays(a_np, b_np, c_np, z_np)

        # Optional: clamp z away from known branch cut endpoints (helps in practice)
        z_np = np.clip(z_np, -1.0 + _EPS_Z, 1.0 - _EPS_Z)

        # Evaluate SciPy hypergeometric 2F1
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            out_np = sp_hyp2f1(a_np, b_np, c_np, z_np)

        out = _from_numpy(out_np, z.device, z.dtype)
        return out

    @staticmethod
    def backward(
        ctx, *grad_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor | None, ...]:
        (a, b, c, z) = ctx.saved_tensors
        device = z.device

        # Convert to numpy
        a_np, b_np, c_np, z_np = map(_to_numpy, (a, b, c, z))
        a1, b1, c1 = a_np + 1.0, b_np + 1.0, c_np + 1.0
        z_np = np.clip(z_np, -1.0 + _EPS_Z, 1.0 - _EPS_Z)

        # dF/dz = (a*b/c) * 2F1(a+1,b+1;c+1;z)
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            Fp_np = sp_hyp2f1(a1, b1, c1, z_np)
        pref = (a_np * b_np) / c_np
        dF_dz_np = pref * Fp_np

        dF_dz = _from_numpy(dF_dz_np, device, z.dtype)
        grad_z = grad_outputs[0].to(z.dtype) * dF_dz

        # No grads for a,b,c (return zeros). Must match number of inputs.
        return None, None, None, grad_z


def hyp2f1(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """
    Public wrapper. Works with broadcasting. Only z may require grad.
    """
    return Hyp2F1.apply(a, b, c, z)  # type: ignore
