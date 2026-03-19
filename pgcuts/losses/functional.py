from typing import Tuple, Union

import torch
from scipy.special import roots_legendre


def entropy(p: torch.Tensor) -> torch.Tensor:
    """Element-wise entropy: -p * log(p)."""
    return torch.special.entr(p)


def sum_excluding_self(t_array: torch.Tensor) -> torch.Tensor:
    """For each row i, compute the sum of all other rows (excluding row i)."""
    to_keep = ~torch.eye(t_array.shape[0], device=t_array.device, dtype=torch.bool)
    indices = torch.where(to_keep)[1].view(t_array.shape[0], -1)
    return t_array[indices].sum(1)


# ---------------------------------------------------------------------------
# Legendre quadrature primitives
# ---------------------------------------------------------------------------


def legendre_quadrature(
    a: float, b: float, degree: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns roots and weights for Legendre quadrature on interval [a, b].

    Args:
        a: Left limit of integration.
        b: Right limit of integration.
        degree: Polynomial degree (number of quadrature points).

    Returns:
        (roots, weights) tensors.
    """
    assert a < b
    roots, weights = roots_legendre(degree)
    weights = weights * (b - a) / 2
    roots = (b - a) / 2 * roots + (a + b) / 2
    return torch.tensor(roots).float(), torch.tensor(weights).float()


def batch_quadrature(
    integral_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Legendre quadrature for integrating a polynomial of given degree on [0, 1].

    Args:
        integral_batch_size: Batch size determining quadrature degree.

    Returns:
        (roots, weights) tensors.
    """
    return legendre_quadrature(0, 1, integral_batch_size // 2 + 1)


# ---------------------------------------------------------------------------
# Quadrature estimation functions
# ---------------------------------------------------------------------------


def estimate_quadrature(
    p: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    integral_subset: Union[torch.Tensor, None] = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute the integral estimate using Gauss-Legendre quadrature.

    Args:
        p: Probability tensor, shape (B, k).
        roots: Quadrature roots, shape (T,).
        weights: Quadrature weights, shape (T,).
        integral_subset: Optional subset indices for pairwise estimation.
        gamma: Exponent applied to the integral estimate.

    Returns:
        Integral estimate tensor.
    """
    if integral_subset is None:
        # (B, k, 1)
        p_integral = p.unsqueeze(-1)
        # (B, k, T)
        integral_estimate = torch.log(1 - p_integral * roots.view(1, 1, -1))
        # (k, T)
        log_integrals = integral_estimate.sum(0) + torch.log(weights.view(1, -1))
        # (1, 1, k, T)
        log_integrals = log_integrals.unsqueeze(0).unsqueeze(0)
    else:
        # (a, a, s, k)
        p_integral = p[integral_subset]
        # (a, a, s, k, T)
        integral_estimate = torch.log(
            1 - p_integral.unsqueeze(-1) * roots.view(1, 1, 1, 1, -1)
        )
        # (a, a, k, T)
        log_integrals = integral_estimate.sum(2) + torch.log(
            weights.view(1, 1, 1, -1)
        )
    # (a, a, k)
    return torch.exp(log_integrals).sum(-1) ** gamma


def integral_quadrature(probs: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Compute quadrature integral inline (allocates its own roots/weights).

    Args:
        probs: Probability tensor, shape (B, K).
        gamma: Exponent applied to the integral estimate.

    Returns:
        Integral values, shape (K,).
    """
    b_size = probs.shape[0]
    roots, weights = batch_quadrature(b_size)
    roots = roots.to(probs)
    weights = weights.to(probs)
    # (B, K, 1)
    p_integral = probs.unsqueeze(-1)
    # (B, K, T)
    integral_estimate = torch.log(1 - p_integral * roots.view(1, 1, -1))
    # (K, T)
    log_integrals = integral_estimate.sum(0) + torch.log(weights.view(1, -1))
    # (K,)
    return torch.exp(log_integrals).sum(-1) ** gamma


def simple_quadrature_estimation(
    p: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Simplified quadrature estimation (no subset, pre-provided roots/weights).

    Args:
        p: Probability tensor, shape (A, K).
        roots: Quadrature roots, shape (T,).
        weights: Quadrature weights, shape (T,).
        gamma: Exponent.

    Returns:
        Integral estimate, shape (1, 1, K).
    """
    # (A, K, 1)
    p_integral = p.unsqueeze(-1)
    # (A, K, T)
    integral_estimate = torch.log(1 - p_integral * roots.view(1, 1, -1))
    # (K, T)
    log_integrals = integral_estimate.sum(0) + torch.log(weights.view(1, -1))
    # (1, 1, K, T)
    log_integrals = log_integrals.unsqueeze(0).unsqueeze(0)
    # (1, 1, K)
    return torch.exp(log_integrals).sum(-1) ** gamma


def pairwise_quadrature(
    p_l: torch.Tensor, p_r: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Pairwise quadrature across left and right probability batches.

    Args:
        p_l: Left probabilities, shape (Bl, K).
        p_r: Right probabilities, shape (Br, K).
        gamma: Exponent.

    Returns:
        Pairwise integral, shape (Bl, Br, K).
    """
    bl_size = p_l.shape[0]
    br_size = p_r.shape[0]
    roots, weights = batch_quadrature(bl_size + br_size - 2)
    # (1, 1, T)
    roots = roots.to(p_l).view(1, 1, -1)
    # (1, 1, 1, T)
    weights = weights.to(p_l).view(1, 1, 1, -1)
    # (Bl, K, T)
    log_p_l_1 = sum_excluding_self(torch.log(1 - p_l.unsqueeze(-1) * roots))
    # (Br, K, T)
    log_p_r_1 = sum_excluding_self(torch.log(1 - p_r.unsqueeze(-1) * roots))
    # (Bl, Br, K, T)
    log_p_pairwise = log_p_l_1.unsqueeze(1) + log_p_r_1.unsqueeze(0)
    return torch.exp(log_p_pairwise).mul(weights).sum(-1) ** gamma


def coupled_quadrature_estimation(
    p_left: torch.Tensor,
    p_right: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Coupled quadrature estimation for left/right probability pairs.

    Args:
        p_left: Left probabilities, shape (A, K).
        p_right: Right probabilities, shape (B, K).
        roots: Quadrature roots, shape (T,).
        weights: Quadrature weights, shape (T,).
        gamma: Exponent.

    Returns:
        Coupled integral estimate, shape (A, B, K).
    """
    # (A, 1, K)
    p_l = p_left.unsqueeze(1)
    # (1, B, K)
    p_r = p_right.unsqueeze(0)
    # (A, 1, K, T)
    log_p1_l = torch.log(1 - p_l.unsqueeze(-1) * roots.view(1, 1, -1))
    # (1, B, K, T)
    log_p1_r = torch.log(1 - p_r.unsqueeze(-1) * roots.view(1, 1, -1))
    # (A, B, K, T)
    integral_estimate = (
        log_p1_l.sum(0, keepdim=True)
        + log_p1_r.sum(1, keepdim=True)
        - log_p1_l
        - log_p1_r
    )
    # (A, B, K, T)
    log_integrals = integral_estimate + torch.log(weights.view(1, 1, 1, -1))
    # (A, B, K)
    return torch.exp(log_integrals).sum(-1) ** gamma


# ---------------------------------------------------------------------------
# Graph-based quadrature
# ---------------------------------------------------------------------------


def graph_quadrature(
    p: torch.Tensor,
    edge_index: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Core graph quadrature computation.

    Args:
        p: Probabilities, shape (B, K).
        edge_index: Edge indices, shape (2, E).
        roots: Quadrature roots, shape (T,).
        weights: Quadrature weights, shape (T,).
        gamma: Exponent.

    Returns:
        Per-edge integral, shape (E, K).
    """
    # (B, K, T)
    log_1_pt = torch.log(1 - p.unsqueeze(-1) * roots.view(1, 1, -1))
    # (E, K, T)
    batch_quad = log_1_pt.sum(0, keepdim=True) - log_1_pt[edge_index].sum(dim=0)
    # (E, K)
    return (torch.exp(batch_quad) * weights.view(1, 1, -1)).sum(-1) ** gamma


def graph_integral_quadrature(
    probs: torch.Tensor, edge_index: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Graph-based quadrature integral (allocates its own roots/weights).

    Args:
        probs: Probabilities, shape (B, K).
        edge_index: Edge indices, shape (2, E).
        gamma: Exponent.

    Returns:
        Per-edge integral, shape (E, K).
    """
    b_size = probs.shape[0]
    roots, weights = batch_quadrature(b_size)
    roots = roots.to(probs)
    weights = weights.to(probs)
    return graph_quadrature(probs, edge_index, roots, weights, gamma)


# ---------------------------------------------------------------------------
# High-level PRCut loss functions
# ---------------------------------------------------------------------------


def graph_prcut(
    probs: torch.Tensor, edge_index: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Graph-based PRCut loss.

    Args:
        probs: Cluster probabilities, shape (B, K).
        edge_index: Edge indices, shape (2, E).
        gamma: Quadrature exponent.

    Returns:
        Scalar loss.
    """
    integral = graph_integral_quadrature(probs, edge_index, gamma)
    p = probs[edge_index]
    multipliers = (p[0] + p[1] - 2 * p[0] * p[1]) * integral
    return multipliers.mean()


def pairwise_prcut(
    p_l: torch.Tensor,
    p_r: torch.Tensor,
    similarities: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Pairwise PRCut loss with similarity weighting.

    Args:
        p_l: Left probabilities, shape (Bl, K).
        p_r: Right probabilities, shape (Br, K).
        similarities: Similarity matrix, shape (Bl, Br).
        gamma: Quadrature exponent.

    Returns:
        Scalar loss.
    """
    # (Bl, Br, K)
    quadrature = pairwise_quadrature(p_l, p_r, gamma=gamma)
    p_l = p_l.unsqueeze(1)
    p_r = p_r.unsqueeze(0)
    # (Bl, Br, K)
    multipliers = p_l + p_r - 2 * p_l * p_r
    return (similarities * (multipliers * quadrature).sum(-1)).mean()


def compute_prcut_loss(
    p: torch.Tensor,
    indices_left: torch.Tensor,
    indices_right: torch.Tensor,
    similarities: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    integral_subset: Union[torch.Tensor, None] = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Full PRCut loss with quadrature integration.

    Args:
        p: Probabilities, shape (B, k).
        indices_left: Left node indices.
        indices_right: Right node indices.
        similarities: Similarity weights.
        roots: Quadrature roots.
        weights: Quadrature weights.
        integral_subset: Optional subset indices.
        gamma: Exponent.

    Returns:
        Scalar loss.
    """
    p_l = p[indices_left].unsqueeze(1)
    # (1, A, k)
    p_r = p[indices_right].unsqueeze(0)
    # (A, A, k)
    multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r)
    # (A, A, K)
    right_quantity = estimate_quadrature(
        p=p, roots=roots, weights=weights, integral_subset=integral_subset, gamma=gamma
    )
    return (multipliers * right_quantity).sum()


def compute_decoupled_prcut_loss(
    p_left: torch.Tensor,
    p_right: torch.Tensor,
    similarities: torch.Tensor,
    p_integral: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Decoupled PRCut loss using separate integral probabilities.

    Args:
        p_left: Left probabilities, shape (B1, K).
        p_right: Right probabilities, shape (B2, K).
        similarities: Similarity matrix, shape (B1, B2).
        p_integral: Probabilities for integral estimation.
        roots: Quadrature roots.
        weights: Quadrature weights.
        gamma: Exponent.

    Returns:
        Scalar loss.
    """
    # (B1, 1, K)
    p_l = p_left.unsqueeze(1)
    # (1, B2, K)
    p_r = p_right.unsqueeze(0)
    # (B1, B2, K)
    multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r)
    right_quantity = simple_quadrature_estimation(
        p=p_integral, roots=roots, weights=weights, gamma=gamma
    )
    return (multipliers * right_quantity).sum()


def compute_coupled_prcut_loss(
    p_left: torch.Tensor,
    p_right: torch.Tensor,
    similarities: torch.Tensor,
    roots: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Coupled PRCut loss with joint quadrature estimation.

    Args:
        p_left: Left probabilities, shape (B1, K).
        p_right: Right probabilities, shape (B2, K).
        similarities: Similarity matrix, shape (B1, B2).
        roots: Quadrature roots.
        weights: Quadrature weights.
        gamma: Exponent.

    Returns:
        Scalar loss.
    """
    # (B1, 1, K)
    p_l = p_left.unsqueeze(1)
    # (1, B2, K)
    p_r = p_right.unsqueeze(0)
    # (B1, B2, K)
    multipliers = similarities.unsqueeze(-1) * (p_l + p_r - 2 * p_l * p_r)
    right_quantity = coupled_quadrature_estimation(
        p_left=p_left, p_right=p_right, roots=roots, weights=weights, gamma=gamma
    )
    return (multipliers * right_quantity).sum()


# ---------------------------------------------------------------------------
# Softmax variants
# ---------------------------------------------------------------------------


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Softmax with masked positions set to -inf."""
    logits = logits.masked_fill(mask, float("-inf"))
    return torch.nn.functional.softmax(logits, dim=dim)


def noisy_softmax(
    logits: torch.Tensor, noise_scale: float = 1.0, tau: float = 1.0, dim: int = -1
) -> torch.Tensor:
    """Softmax with additive Gaussian noise and temperature scaling."""
    return torch.softmax(
        (logits + noise_scale * torch.randn_like(logits)) / tau, dim=dim
    )


def topk_softmax(
    logits: torch.Tensor, k: int = 2, dim: int = -1
) -> torch.Tensor:
    """Softmax restricted to the top-k logits."""
    _, indices = torch.topk(logits, k, dim=dim)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(index=indices, value=False, dim=-1)
    return masked_softmax(logits, mask, dim=dim)
