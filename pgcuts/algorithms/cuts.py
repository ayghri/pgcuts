"""Probabilistic graph cut algorithms -- single-step functions.

Each function computes the cut loss for one batch of edges, given
soft cluster assignments P = softmax(logits / tau).

The three objectives share the same structure:
    cut_loss = sum_edges  w_ij * cut_term(p_i, p_j) * envelope(alpha)

where:
    - PRCut:    cut_term = p_i(1-p_j),  envelope = 1/p_bar_l
    - Hyp-RCut: cut_term = p_i(1-p_j),  envelope = 2F1(-m,1;2;a_l)
    - Hyp-NCut: cut_term = p_i(1-p_j),  envelope = Phi(bin_i,a)/deg_i

All functions return (cut_loss, balance_loss, updated_ema_state).
"""

from typing import Tuple

import torch
from torch import Tensor

from ..hyp2f1.autograd import Hyp2F1
from ..losses.pncut import compute_ncut_bin_phi

_h = Hyp2F1.apply


def prcut_original_step(
    probs: Tensor,
    left_idx: Tensor,
    right_idx: Tensor,
    w: Tensor,
    p_ema: Tensor,
    n: int,
    ema: float = 0.9,
) -> Tuple[Tensor, Tensor]:
    """Original PRCut step with analytical gradient.

    No GradientMixer needed. The analytical gradient
    includes both the cut term and the 1/p_bar envelope
    derivative, computed edge-wise.

    Args:
        probs: Soft assignments, shape (U, K).
        left_idx: Left endpoint indices, shape (E,).
        right_idx: Right endpoint indices, shape (E,).
        w: Edge weights, shape (E,).
        p_ema: EMA cluster proportions, shape (K,).
        n: Total dataset size (for gradient scaling).
        ema: EMA decay factor.

    Returns:
        surrogate_loss: Scalar (backward gives PRCut grad).
        p_ema: Updated EMA (detached).
    """
    p_left = probs[left_idx]
    p_right = probs[right_idx]
    ov_p = ema * p_ema + (1 - ema) * probs.mean(0).detach()

    w_col = w.unsqueeze(-1)
    cut_per_k = (w_col * p_left * (1.0 - p_right)).sum(0)

    with torch.no_grad():
        grad_l = w_col * (1 - 2 * p_right) / ov_p - cut_per_k / (ov_p**2 * n)
        grad_r = w_col * (1 - 2 * p_left) / ov_p - cut_per_k / (ov_p**2 * n)

    surrogate = (grad_l * p_left).sum() + (grad_r * p_right).sum()

    with torch.no_grad():
        p_ema = (ema * p_ema + (1 - ema) * probs.mean(0).detach()).detach()

    return surrogate, p_ema


def prcut_step(
    probs: Tensor,
    left_idx: Tensor,
    right_idx: Tensor,
    w: Tensor,
    p_ema: Tensor,
    ema: float = 0.9,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Probabilistic Ratio Cut -- one batch step.

    Args:
        probs: Soft assignments, shape (U, K).
        left_idx: Indices into probs for left endpoints,
            shape (E,).
        right_idx: Indices into probs for right endpoints,
            shape (E,).
        w: Edge weights, shape (E,).
        p_ema: EMA cluster proportions, shape (K,).
        ema: EMA decay factor.

    Returns:
        cut_loss: Scalar PRCut loss.
        balance: Scalar entropy balance loss (-H(p_bar)).
        p_ema: Updated EMA cluster proportions (detached).
    """
    p_left = probs[left_idx]
    p_right = probs[right_idx]

    cut_per_k = (w.unsqueeze(-1) * p_left * (1.0 - p_right)).mean(0)
    p_bar = ema * p_ema + (1 - ema) * probs.mean(0).detach()
    cut_loss = (cut_per_k / (p_bar + 1e-12)).sum() / w.sum()

    balance = -torch.special.entr(probs.mean(0)).sum()

    with torch.no_grad():
        p_ema = (ema * p_ema + (1 - ema) * probs.mean(0).detach()).detach()

    return cut_loss, balance, p_ema


def hyp_rcut_step(
    logits: Tensor,
    left_idx: Tensor,
    right_idx: Tensor,
    w: Tensor,
    p_ema: Tensor,
    m: int = 512,
    ema: float = 0.9,
    distance: str = "ce",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Hypergeometric Ratio Cut -- one batch step.

    Args:
        logits: Raw model output, shape (U, K).
        left_idx: Indices for left endpoints, shape (E,).
        right_idx: Indices for right endpoints, shape (E,).
        w: Edge weights, shape (E,).
        p_ema: EMA cluster proportions, shape (K,).
        m: Polynomial degree for 2F1.
        ema: EMA decay factor.
        distance: 'xor' for P_i(1-P_j) or 'ce' for
            -P_i log P_j.

    Returns:
        cut_loss: Scalar Hyp-RCut loss.
        balance: Scalar entropy balance loss.
        p_ema: Updated EMA (detached).
    """
    probs = torch.softmax(logits, dim=-1)
    p_left = probs[left_idx]

    if distance == "ce":
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

        log_p_right = F.log_softmax(logits[right_idx], dim=-1)
        cut_per_k = (w.unsqueeze(-1) * (-p_left * log_p_right)).mean(0)
    else:
        p_right = probs[right_idx]
        cut_per_k = (w.unsqueeze(-1) * p_left * (1.0 - p_right)).mean(0)

    alpha = ema * p_ema + (1 - ema) * probs.mean(0)
    z = alpha.clamp(1e-7, 1 - 1e-7)
    device = logits.device
    h_val = _h(
        -m,
        torch.tensor(1.0, device=device),
        torch.tensor(2.0, device=device),
        z,
    )

    cut_loss = (cut_per_k * h_val).sum() / w.sum()
    balance = -torch.special.entr(probs.mean(0)).sum()

    with torch.no_grad():
        p_ema = (ema * p_ema + (1 - ema) * probs.mean(0).detach()).detach()

    return cut_loss, balance, p_ema


def hyp_ncut_step(
    logits: Tensor,
    left_idx: Tensor,
    right_idx: Tensor,
    w: Tensor,
    left_node_ids: Tensor,
    alpha_ema: Tensor,
    q_stars: Tensor,
    beta_stars: Tensor,
    bin_weights: Tensor,
    node_to_bin: Tensor,
    degrees: Tensor,
    m: int = 512,
    ema: float = 0.9,
    distance: str = "ce",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Hypergeometric Normalized Cut -- one batch step.

    Args:
        logits: Raw model output, shape (U, K).
        left_idx: Indices for left endpoints, shape (E,).
        right_idx: Indices for right endpoints,
            shape (E,).
        w: Edge weights, shape (E,).
        left_node_ids: Global node IDs of left endpoints,
            shape (E,).
        alpha_ema: Per-bin EMA proportions, shape (d, K).
        q_stars: Per-bin representative degrees, shape (d,).
        beta_stars: Per-bin minimum degrees, shape (d,).
        bin_weights: Holder weights, shape (d,).
        node_to_bin: Node ID to bin index, shape (N,).
        degrees: Per-vertex degrees, shape (N,).
        m: Polynomial degree for 2F1.
        ema: EMA decay factor.
        distance: 'xor' for P_i(1-P_j) or 'ce' for
            -P_i log P_j.

    Returns:
        cut_loss: Scalar Hyp-NCut loss.
        balance: Scalar entropy balance loss.
        alpha_ema: Updated per-bin EMA (detached).
    """
    probs = torch.softmax(logits, dim=-1)
    p_left = probs[left_idx]

    if distance == "ce":
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

        log_p_right = F.log_softmax(logits[right_idx], dim=-1)
        cut_per_edge = w.unsqueeze(-1) * (-p_left * log_p_right)
    else:
        assert distance == "xor", "only ce or xor distance supported"
        p_right = probs[right_idx]
        cut_per_edge = w.unsqueeze(-1) * p_left * (1.0 - p_right)

    alpha_live = ema * alpha_ema + (1 - ema) * probs.mean(0).unsqueeze(0)
    phi_bins = compute_ncut_bin_phi(q_stars, alpha_live, beta_stars, bin_weights, m)

    left_bins = node_to_bin[left_node_ids]
    phi_edges = phi_bins[left_bins]
    deg_left = degrees[left_node_ids].clamp(min=1e-6).unsqueeze(-1)

    cut_loss = (cut_per_edge * phi_edges / deg_left).sum() / w.sum()
    balance = -torch.special.entr(probs.mean(0)).sum()

    with torch.no_grad():
        alpha_ema = (
            ema * alpha_ema + (1 - ema) * probs.detach().mean(0).unsqueeze(0)
        ).detach()

    return cut_loss, balance, alpha_ema
