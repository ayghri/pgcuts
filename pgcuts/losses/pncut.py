"""PGCut losses -- RatioCut and NCut with hyp envelope.

RatioCut (s(v)=1 for all v):
    H(alpha_bar) = 2F1(-m, 1; 2; alpha_bar)
    Same for every vertex.

NCut (s(v)=d_v, the degree):
    Holder bound with binning (Theorem 2).
    Phi varies per vertex, unlike RatioCut.
"""

from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn, Tensor

from ..hyp2f1.autograd import Hyp2F1

_hyp2f1 = Hyp2F1.apply


# -----------------------------------------------------------
# Binning
# -----------------------------------------------------------


def equal_size_bins(degrees: np.ndarray, num_bins: int) -> List[dict]:
    """Partition vertices into equal-size bins."""
    sorted_idx = np.argsort(degrees)
    splits = np.array_split(sorted_idx, num_bins)
    bins = []
    for indices in splits:
        if len(indices) == 0:
            continue
        bins.append(
            {
                "beta_star": float(degrees[indices].min()),
                "indices": indices,
                "count": len(indices),
            }
        )
    return bins


def log_kmeans_bins(degrees: np.ndarray, num_bins: int) -> List[dict]:
    """Partition vertices via K-Means on log-degrees."""
    # pylint: disable=import-outside-toplevel
    from sklearn.cluster import KMeans

    log_deg = np.log(degrees + 1e-6).reshape(-1, 1)
    labels = KMeans(n_clusters=num_bins, n_init=10, random_state=42).fit_predict(
        log_deg
    )
    bins = []
    for label in range(num_bins):
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue
        bins.append(
            {
                "beta_star": float(degrees[indices].min()),
                "indices": indices,
                "count": len(indices),
            }
        )
    return sorted(bins, key=lambda b: b["beta_star"])


# -----------------------------------------------------------
# Shared
# -----------------------------------------------------------


def edge_source_weights(w_mat: Tensor, probs: Tensor) -> Tensor:
    """M_il(P) = sum_j W_ij P_il (1 - P_jl)."""
    return probs * torch.mm(w_mat, 1.0 - probs)


# -----------------------------------------------------------
# RatioCut -- Theorem 1 (homogeneous beta = 1)
# -----------------------------------------------------------


class RatioCutLoss(nn.Module):
    """Probabilistic RatioCut with hyp envelope.

    H_l = 2F1(-m, 1; 2; alpha_bar_l).
    """

    def __init__(self, n: int, ema_decay: float = 0.0) -> None:
        """Initialize RatioCutLoss.

        Args:
            n: Dataset size (used as polynomial degree).
            ema_decay: EMA decay for alpha tracking.
        """
        super().__init__()
        self.n = n
        self.ema_decay = ema_decay

    def forward(
        self,
        w_mat: Tensor,
        probs: Tensor,
        alpha_ema: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute RatioCut envelope loss.

        Args:
            w_mat: Adjacency matrix, shape (n, n).
            probs: Assignment probs, shape (n, K).
            alpha_ema: Running means, shape (K,).

        Returns:
            (loss, updated_alpha_bar).
        """
        alpha_bar = probs.detach().mean(0)
        if alpha_ema is not None and self.ema_decay > 0:
            alpha_bar = self.ema_decay * alpha_ema + (1 - self.ema_decay) * alpha_bar

        h_val = _hyp2f1(
            -self.n,
            1.0,
            2.0,
            alpha_bar.clamp(1e-7, 1 - 1e-7),
        )

        m_weights = edge_source_weights(w_mat, probs)
        loss = (m_weights.sum(0) * h_val).sum() / w_mat.sum().clamp(min=1e-9)

        return loss, alpha_bar


# -----------------------------------------------------------
# NCut -- Theorem 2 (Holder bound, beta = d_i)
# -----------------------------------------------------------


class NCutLoss(nn.Module):
    """Probabilistic NCut with Holder-binned envelope.

    Phi varies per vertex, unlike RatioCut.
    """

    def __init__(
        self,
        degrees: np.ndarray,
        num_bins: int = 16,
        binning: str = "equal",
        ema_decay: float = 0.0,
    ) -> None:
        """Initialize NCutLoss.

        Args:
            degrees: Vertex degrees, shape (n,).
            num_bins: Number of Holder bins.
            binning: 'equal' or 'log_kmeans'.
            ema_decay: EMA decay for per-bin tracking.
        """
        super().__init__()
        self.ema_decay = ema_decay
        n = len(degrees)

        if binning == "equal":
            self.bins = equal_size_bins(degrees, num_bins)
        elif binning == "log_kmeans":
            self.bins = log_kmeans_bins(degrees, num_bins)
        else:
            raise ValueError(f"Unknown binning: {binning}")

        self.register_buffer(
            "degrees_t",
            torch.tensor(degrees, dtype=torch.float32),
        )

        beta_stars = torch.tensor(
            [b["beta_star"] for b in self.bins],
            dtype=torch.float32,
        )
        self.register_buffer("beta_stars", beta_stars)

        counts = torch.tensor(
            [b["count"] for b in self.bins],
            dtype=torch.float32,
        )
        self.register_buffer("bin_weights", counts / counts.sum())

        # Node-to-bin mapping
        node_to_bin = np.zeros(n, dtype=np.int64)
        for j, b in enumerate(self.bins):
            node_to_bin[b["indices"]] = j
        self.register_buffer(
            "node_to_bin",
            torch.tensor(node_to_bin, dtype=torch.long),
        )

        self._bin_indices = [
            torch.tensor(b["indices"], dtype=torch.long) for b in self.bins
        ]

    def _bin_means(self, probs: Tensor) -> Tensor:
        """Per-bin mean assignments, shape (d, K)."""
        num_bins = len(self.bins)
        num_clusters = probs.shape[1]
        alpha_bars = torch.zeros(
            num_bins,
            num_clusters,
            device=probs.device,
            dtype=probs.dtype,
        )
        for j, idx in enumerate(self._bin_indices):
            idx = idx.to(probs.device)
            alpha_bars[j] = probs[idx].mean(0)
        return alpha_bars

    def compute_phi(
        self,
        q: Tensor,
        alpha_bars: Tensor,
        m: int,
    ) -> Tensor:
        """Compute per-vertex Holder envelope.

        Args:
            q: Per-vertex degrees, shape (num_v,).
            alpha_bars: Per-bin means, shape (d, K).
            m: Polynomial degree for 2F1.

        Returns:
            Phi values, shape (num_v, K).
        """
        num_v = q.shape[0]
        num_bins, num_clusters = alpha_bars.shape
        device = q.device

        beta = self.beta_stars.to(device)
        w = self.bin_weights.to(device)

        # c = q_i / beta*_j + 1
        c = q.unsqueeze(1) / beta.unsqueeze(0) + 1.0

        z = alpha_bars.clamp(1e-7, 1 - 1e-7)

        # Broadcast to (num_v, d, K)
        c_3d = c.unsqueeze(2).expand(num_v, num_bins, num_clusters)
        z_3d = z.unsqueeze(0).expand(num_v, num_bins, num_clusters)

        # 2F1(-m, 1; c; z) -- (num_v, d, K)
        f_val = _hyp2f1(-m, 1.0, c_3d, z_3d)

        # h = (1/q) * f -- (num_v, d, K)
        h_val = f_val / q.view(num_v, 1, 1)

        # Holder composition in log space
        log_h = torch.log(h_val.clamp(min=1e-30))
        w_3d = w.view(1, num_bins, 1)
        log_phi = (w_3d * log_h).sum(dim=1)

        return torch.exp(log_phi)

    def forward(
        self,
        w_mat: Tensor,
        probs: Tensor,
        alpha_bars_ema: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute NCut envelope loss.

        Args:
            w_mat: Adjacency matrix, shape (n, n).
            probs: Assignment probs, shape (n, K).
            alpha_bars_ema: Running per-bin means,
                shape (d, K).

        Returns:
            (loss, updated_alpha_bars).
        """
        n = probs.shape[0]
        device = probs.device

        # Per-bin means
        alpha_bars = self._bin_means(probs.detach())
        if alpha_bars_ema is not None and self.ema_decay > 0:
            alpha_bars = (
                self.ema_decay * alpha_bars_ema + (1 - self.ema_decay) * alpha_bars
            )

        q = self.degrees_t.to(device).clamp(min=1e-6)

        m = n

        # Phi for each vertex -- (n, K)
        phi = self.compute_phi(q, alpha_bars, m)

        # M_il * Phi_il
        m_weights = edge_source_weights(w_mat, probs)
        loss = (m_weights * phi).sum() / w_mat.sum().clamp(min=1e-9)

        return loss, alpha_bars


# -----------------------------------------------------------
# Edge-pair interface for minibatch training
# -----------------------------------------------------------


def compute_ncut_bin_phi(
    q_stars: Tensor,
    alpha_bars: Tensor,
    beta_stars: Tensor,
    bin_weights: Tensor,
    m: int,
) -> Tensor:
    """Compute per-bin envelope Phi (without 1/q).

    Args:
        q_stars: Per-bin degrees, shape (d,).
        alpha_bars: Per-bin means, shape (d, K).
        beta_stars: Per-bin exponents, shape (d,).
        bin_weights: Holder weights, shape (d,).
        m: Polynomial degree.

    Returns:
        Phi values, shape (d, K).
    """
    num_bins, num_clusters = alpha_bars.shape

    q = q_stars.clamp(min=1e-6)

    # c[i,j] = q*_i / beta*_j + 1
    c = q.unsqueeze(1) / beta_stars.unsqueeze(0) + 1.0

    z = alpha_bars.clamp(1e-7, 1 - 1e-7)

    # Evaluate 2F1 at (d*d, K) points
    c_2d = (
        c.unsqueeze(2)
        .expand(num_bins, num_bins, num_clusters)
        .reshape(num_bins * num_bins, num_clusters)
    )
    z_2d = (
        z.unsqueeze(0)
        .expand(num_bins, num_bins, num_clusters)
        .reshape(num_bins * num_bins, num_clusters)
    )

    f_val = _hyp2f1(-m, 1.0, c_2d, z_2d).view(num_bins, num_bins, num_clusters)

    log_f = torch.log(f_val.clamp(min=1e-30))
    w = bin_weights.view(1, num_bins, 1)
    log_phi = (w * log_f).sum(dim=1)

    return torch.exp(log_phi)
