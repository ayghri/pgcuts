"""PRCut loss modules."""
from typing import Tuple

import torch
from torch import nn


@torch.no_grad()
def offline_gradient(
    w_mat: torch.Tensor, probs: torch.Tensor
) -> torch.Tensor:
    """Compute full PRCut gradient offline.

    Args:
        w_mat: Weight matrix, shape (n, n).
        probs: Probability matrix, shape (n, k).

    Returns:
        Gradient w.r.t. probs, shape (n, k).
    """
    ov_p = probs.mean(0)
    left = (
        w_mat.sum(0).unsqueeze(1)
        - 2 * torch.mm(w_mat, probs.detach())
    ) / ov_p
    right = (
        -(w_mat.mm(probs) - w_mat.mm(probs) * probs)
        .sum(0)
        / ov_p ** 2
        / probs.size(0)
    )
    return left + right


@torch.no_grad()
def batch_cluster_prcut_loss(
    w_mat: torch.Tensor,
    p_l: torch.Tensor,
    p_r: torch.Tensor,
    ov_p: torch.Tensor,
) -> torch.Tensor:
    """Per-cluster PRCut loss for a batch.

    Args:
        w_mat: Weight matrix, shape (a, b).
        p_l: Left probabilities, shape (a, k).
        p_r: Right probabilities, shape (b, k).
        ov_p: Cluster likelihood, shape (k,).

    Returns:
        Per-cluster loss, shape (k,).
    """
    p_l = p_l.unsqueeze(1)
    p_r = p_r.unsqueeze(0)
    return (
        w_mat.unsqueeze(-1)
        * (p_l + p_r - 2 * p_l * p_r)
    ).sum(dim=(0, 1)) / ov_p


@torch.no_grad()
def batch_gradient(
    w_mat: torch.Tensor,
    p_l: torch.Tensor,
    p_r: torch.Tensor,
    ov_p: torch.Tensor,
    n: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute PRCut batch gradient.

    Args:
        w_mat: Weight matrix, shape (a, b).
        p_l: Left probabilities, shape (a, k).
        p_r: Right probabilities, shape (b, k).
        ov_p: Cluster likelihood, shape (k,).
        n: Number of samples.

    Returns:
        Tuple of gradients w.r.t. left and right.
    """
    left_l = w_mat.mm(1 - 2 * p_r) / ov_p
    left_r = w_mat.t().mm(1 - 2 * p_l) / ov_p
    right = (
        -batch_cluster_prcut_loss(
            w_mat, p_l, p_r, ov_p
        )
        / ov_p
        / n
    )
    return left_l + right, left_r + right


class PRCutGradLoss(nn.Module):
    """PRCut loss with analytical gradient.

    Uses the analytical gradient to construct a
    surrogate loss for backpropagation.
    """

    def forward(
        self, w_mat, p_l, p_r, ov_p, n
    ) -> torch.Tensor:
        """Compute surrogate PRCut loss.

        Args:
            w_mat: Weight tensor, shape (a, b).
            p_l: Left probabilities, shape (a, k).
            p_r: Right probabilities, shape (b, k).
            ov_p: Cluster likelihood, shape (k,).
            n: Number of samples.

        Returns:
            Scalar loss.
        """
        p_l_grad, p_r_grad = batch_gradient(
            w_mat, p_l, p_r, ov_p, n
        )
        return (p_l_grad * p_l).sum() + (
            p_r_grad * p_r
        ).sum()


class PRCutBatchLoss(nn.Module):
    """PRCut batch estimate with EMA tracking.

    Maintains running estimate of cluster probabilities.
    """

    def __init__(
        self, num_clusters: int, gamma: float
    ) -> None:
        """Initialize PRCutBatchLoss.

        Args:
            num_clusters: Number of clusters.
            gamma: EMA decay rate.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.gamma = gamma
        self.clusters_p = nn.Parameter(
            torch.ones(num_clusters) / num_clusters
        )

    @torch.no_grad()
    def update_cluster_p(
        self, probs: torch.Tensor
    ) -> None:
        """Update cluster probability estimates.

        Args:
            probs: Assignment probs, shape (n, k).
        """
        self.clusters_p.data.mul_(1 - self.gamma)
        self.clusters_p.data.add_(
            probs.detach().mean(0) * self.gamma
        )

    @property
    def cluster_likelihood(self) -> torch.Tensor:
        """Return current cluster likelihood."""
        return self.clusters_p

    @torch.no_grad()
    def forward(
        self, w_mat, p_l, p_r
    ) -> torch.Tensor:
        """Compute PRCut batch loss.

        Args:
            w_mat: Weight tensor, shape (a, b).
            p_l: Left probabilities, shape (a, k).
            p_r: Right probabilities, shape (b, k).

        Returns:
            Scalar loss.
        """
        p_i = p_l.unsqueeze(1)
        p_j = p_r.unsqueeze(0)
        return (
            (
                w_mat.unsqueeze(-1)
                * (p_i + p_j - 2 * p_i * p_j)
            ).sum(dim=(0, 1))
            / self.clusters_p
        ).sum()


class SimplexL2Loss(nn.Module):
    """Simplex L2 regularization loss."""

    def forward(
        self,
        probs: torch.Tensor,
        ov_p: torch.Tensor,
    ) -> torch.Tensor:
        """Compute simplex L2 loss.

        Args:
            probs: Assignment probs, shape (n, k).
            ov_p: Cluster likelihood, shape (k,).

        Returns:
            Scalar loss.
        """
        return (
            (ov_p - 1 / probs.size(1))
            * probs.mean(0)
        ).sum()
