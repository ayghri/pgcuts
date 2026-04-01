"""FlashCut: Custom autograd for Hyp-RCut.

Computes the HyCut loss with two gradient terms:
  - Term A: gradient through the cross-entropy cut
  - Term B: gradient through the 2F1 envelope
"""

import torch
from torch import Tensor

from ..hyp2f1.autograd import Hyp2F1

_h = Hyp2F1.apply


class FlashCutRCut(torch.autograd.Function):
    """Hyp-RCut loss with manual two-term backward.

    Forward:
        cut[l] = c_l * h_val(alpha_l)
        c_l = mean_e(w_e * (-p_left[e,l] * log_p_r[e,l]))
        h_val(alpha_l) = 2F1(-m, 1; 2; alpha_l)

    Backward (two terms):
        Term A: dC_l/dinputs * h_val(alpha_l)
        Term B: c_l * h_prime(alpha_l) * (1/E)
    """

    @staticmethod
    def forward(
        ctx, p_left, log_p_right, w, alpha_ema, m
    ):
        """Compute per-cluster cut values.

        Args:
            p_left: Softmax probs, shape (E, K).
            log_p_right: Log-softmax, shape (E, K).
            w: Edge weights, shape (E,).
            alpha_ema: EMA proportions, shape (K,).
            m: Polynomial degree.

        Returns:
            Per-cluster cut values, shape (K,).
        """
        num_edges, num_clusters = p_left.shape
        dtype = p_left.dtype
        z = alpha_ema.clamp(1e-7, 1 - 1e-7)

        with torch.no_grad():
            h_val = _h(-m, 1.0, 2.0, z).to(dtype)

            if m > 0:
                h_prime = (
                    (-m / 2.0)
                    * _h(-m + 1, 2.0, 3.0, z)
                ).to(dtype)
            else:
                h_prime = torch.zeros(
                    num_clusters,
                    dtype=dtype,
                    device=p_left.device,
                )

        # Per-cluster cross-entropy cut
        cut_vals = (
            w.unsqueeze(-1)
            * (-p_left * log_p_right)
        ).mean(0)  # (K,)

        ctx.save_for_backward(p_left, log_p_right, w)
        ctx.h_val = h_val
        ctx.h_prime = h_prime
        ctx.cut_vals = cut_vals.detach()
        ctx.num_edges = num_edges

        return cut_vals * h_val  # (K,)

    @staticmethod
    def backward(ctx, grad_output):
        """Compute two-term gradient."""
        p_left, log_p_right, w = ctx.saved_tensors
        h_val = ctx.h_val
        h_prime = ctx.h_prime
        cut_vals = ctx.cut_vals
        num_edges = ctx.num_edges
        go = grad_output

        # Term A: gradient through cut
        scale_a = (go * h_val) / num_edges
        w_col = w.unsqueeze(-1)

        grad_p_left = (
            w_col * (-log_p_right)
            * scale_a.unsqueeze(0)
        )
        grad_log_p_right = (
            w_col * (-p_left) * scale_a.unsqueeze(0)
        )

        # Term B: gradient through envelope
        envelope_grad = (
            go * cut_vals * h_prime
        ) / num_edges
        grad_p_left = (
            grad_p_left + envelope_grad.unsqueeze(0)
        )

        return (
            grad_p_left,
            grad_log_p_right,
            None,
            None,
            None,
        )


def flashcut_rcut(
    p_left: Tensor,
    log_p_right: Tensor,
    w: Tensor,
    alpha_ema: Tensor,
    m: int,
) -> Tensor:
    """Compute Hyp-RCut with manual backward.

    Returns:
        (K,) per-cluster cut x envelope values.
    """
    return FlashCutRCut.apply(
        p_left, log_p_right, w, alpha_ema, m
    )
