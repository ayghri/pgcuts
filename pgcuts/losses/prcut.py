from typing import Tuple

import torch
from torch import nn


@torch.no_grad()
def offline_gradient(W: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Computes the full PRCut gradient offline (no autograd needed).

    Parameters:
        W: Weight matrix, shape (n, n).
        P: Probability matrix, shape (n, k).

    Returns:
        Gradient with respect to P, shape (n, k).
    """
    ov_P = P.mean(0)
    left = (W.sum(0).unsqueeze(1) - 2 * torch.mm(W, P.detach())) / ov_P
    right = -(W.mm(P) - W.mm(P) * P).sum(0) / ov_P**2 / P.size(0)
    return left + right


@torch.no_grad()
def batch_cluster_prcut_loss(
    W: torch.Tensor,
    P_l: torch.Tensor,
    P_r: torch.Tensor,
    ov_P: torch.Tensor,
) -> torch.Tensor:
    """Per-cluster PRCut loss for a batch of left/right probabilities.

    Parameters:
        W: Weight matrix, shape (a, b).
        P_l: Left probabilities, shape (a, k).
        P_r: Right probabilities, shape (b, k).
        ov_P: Cluster likelihood, shape (k,).

    Returns:
        Per-cluster loss, shape (k,).
    """
    P_l = P_l.unsqueeze(1)
    P_r = P_r.unsqueeze(0)
    return (W.unsqueeze(-1) * (P_l + P_r - 2 * P_l * P_r)).sum(dim=(0, 1)) / ov_P


@torch.no_grad()
def batch_gradient(
    W: torch.Tensor,
    P_l: torch.Tensor,
    P_r: torch.Tensor,
    ov_P: torch.Tensor,
    n: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the PRCut batch gradient for a given weight matrix and probabilities.

    Parameters:
        W: Weight matrix, shape (a, b).
        P_l: Left probability distribution, shape (a, k).
        P_r: Right probability distribution, shape (b, k).
        ov_P: Cluster likelihood, shape (k,).
        n: Number of samples.

    Returns:
        Tuple of gradients w.r.t. left and right probabilities.
    """
    left_l = W.mm(1 - 2 * P_r) / ov_P
    left_r = W.t().mm(1 - 2 * P_l) / ov_P
    right = -batch_cluster_prcut_loss(W, P_l, P_r, ov_P) / ov_P / n
    return left_l + right, left_r + right


class PRCutGradLoss(nn.Module):
    """Computes the PRCut loss that can be backpropagated.

    Uses the analytical gradient to construct a surrogate loss whose
    backward pass yields the correct PRCut gradient.

    Parameters:
        W: Weight tensor, shape (a, b).
        P_l: Left probabilities, shape (a, k).
        P_r: Right probabilities, shape (b, k).
        ov_P: Cluster likelihood estimator, shape (k,).
        n: Number of samples.

    Returns:
        Scalar loss for backpropagation.
    """

    def forward(self, W, P_l, P_r, ov_P, n) -> torch.Tensor:
        P_l_grad, P_r_grad = batch_gradient(W, P_l, P_r, ov_P, n)
        return (P_l_grad * P_l).sum() + (P_r_grad * P_r).sum()


class PRCutBatchLoss(nn.Module):
    """Computes the PRCut batch estimate with EMA cluster likelihood tracking.

    Maintains a running estimate of cluster probabilities and uses it to
    compute an unbiased batch estimate of the PRCut objective.

    Parameters:
        num_clusters: Number of clusters.
        gamma: EMA decay rate for cluster likelihood update.
    """

    def __init__(self, num_clusters: int, gamma: float) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.gamma = gamma
        self.clusters_p = nn.Parameter(torch.ones(num_clusters) / num_clusters)

    @torch.no_grad()
    def update_cluster_p(self, P: torch.Tensor) -> None:
        self.clusters_p.data.mul_(1 - self.gamma)
        self.clusters_p.data.add_(P.detach().mean(0) * self.gamma)

    @property
    def cluster_likelihood(self) -> torch.Tensor:
        return self.clusters_p

    @torch.no_grad()
    def forward(self, W, P_l, P_r) -> torch.Tensor:
        P_i = P_l.unsqueeze(1)
        P_j = P_r.unsqueeze(0)
        return (
            (W.unsqueeze(-1) * (P_i + P_j - 2 * P_i * P_j)).sum(dim=(0, 1))
            / self.clusters_p
        ).sum()


class SimplexL2Loss(nn.Module):
    """Simplex L2 regularization loss with unbiased gradient.

    The original loss is sum_l (p_l - 1/k)^2 where p_l = mean_i(P_{i,l}).
    The gradient is 2(p_l - 1/k)/n, estimated via the cluster likelihood ov_P.
    """

    def forward(self, P: torch.Tensor, ov_P: torch.Tensor) -> torch.Tensor:
        return ((ov_P - 1 / P.size(1)) * P.mean(0)).sum()
