"""SciPy-based 2F1 with PyTorch autograd wrapper."""
import numpy as np
import torch
from torch.autograd import Function
from typing import Tuple
from scipy.special import hyp2f1 as sp_hyp2f1

_EPS_Z = 1e-12


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    return x.detach().cpu().numpy()


def _from_numpy(
    x: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert numpy array to tensor."""
    return torch.from_numpy(
        np.ascontiguousarray(x)
    ).to(device=device, dtype=dtype)


class Hyp2F1(Function):
    """Torch autograd wrapper for SciPy 2F1(a,b;c;z).

    Gradients implemented w.r.t. z only.
    """

    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 2F1 forward via SciPy."""
        if (
            a.requires_grad
            or b.requires_grad
            or c.requires_grad
        ):
            raise RuntimeError(
                "Hyp2F1 backward supports gradients "
                "w.r.t. z only."
            )

        ctx.save_for_backward(
            a.detach(), b.detach(), c.detach(),
            z.detach(),
        )

        a_np, b_np, c_np, z_np = map(
            _to_numpy, (a, b, c, z)
        )
        a_np, b_np, c_np, z_np = np.broadcast_arrays(
            a_np, b_np, c_np, z_np
        )
        z_np = np.clip(
            z_np, -1.0 + _EPS_Z, 1.0 - _EPS_Z
        )

        with np.errstate(
            over="raise", under="ignore",
            invalid="raise",
        ):
            out_np = sp_hyp2f1(a_np, b_np, c_np, z_np)

        out = _from_numpy(out_np, z.device, z.dtype)
        return out

    @staticmethod
    def backward(
        ctx, *grad_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor | None, ...]:
        """Compute gradient w.r.t. z via SciPy."""
        (a, b, c, z) = ctx.saved_tensors

        a_np, b_np, c_np, z_np = map(
            _to_numpy, (a, b, c, z)
        )
        a1, b1, c1 = (
            a_np + 1.0,
            b_np + 1.0,
            c_np + 1.0,
        )
        z_np = np.clip(
            z_np, -1.0 + _EPS_Z, 1.0 - _EPS_Z
        )

        with np.errstate(
            over="raise", under="ignore",
            invalid="raise",
        ):
            fp_np = sp_hyp2f1(a1, b1, c1, z_np)
        pref = (a_np * b_np) / c_np
        df_dz_np = pref * fp_np

        df_dz = _from_numpy(
            df_dz_np, z.device, z.dtype
        )
        grad_z = grad_outputs[0].to(z.dtype) * df_dz

        return None, None, None, grad_z


def hyp2f1(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Public wrapper for 2F1. Only z may require grad."""
    return Hyp2F1.apply(a, b, c, z)
