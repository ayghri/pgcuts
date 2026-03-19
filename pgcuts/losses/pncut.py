"""PGCut losses — RatioCut and NCut with hypergeometric envelope.

RatioCut (s(v)=1 for all v):
    β_i = 1 (homogeneous) → Theorem 1 applies directly.
    H(ᾱ) = ₂F₁(-m, 1; 2; ᾱ)  is the same for every vertex.

NCut (s(v)=d_v, the degree):
    β_i = d_i (heterogeneous) → Theorem 2 (Hölder bound with binning).
    Bin vertices by degree, use β*_j = min degree in bin j.
    For each source vertex i with degree d_i:
        Φ_ℓ(d_i) = ∏_j [ H_{β*_j}(d_i; ᾱ_{ℓj}, m) ]^{w_j}
    where H_{β*}(q; ᾱ, m) = (1/q) · ₂F₁(-m, 1; q/β*+1; ᾱ).

The NCut envelope Φ_ℓ(d_i) varies per vertex — this is the key
difference from RatioCut where the envelope is per-cluster only.

All ₂F₁ evaluations run on GPU via pgcuts.hyp2f1.Hyp2F1.
"""

from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn, Tensor

from ..hyp2f1.autograd import Hyp2F1

_hyp2f1 = Hyp2F1.apply


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def equal_size_bins(degrees: np.ndarray, num_bins: int) -> List[dict]:
    """Partition vertices into equal-size bins sorted by degree."""
    sorted_idx = np.argsort(degrees)
    splits = np.array_split(sorted_idx, num_bins)
    bins = []
    for indices in splits:
        if len(indices) == 0:
            continue
        bins.append({
            "beta_star": float(degrees[indices].min()),
            "indices": indices,
            "count": len(indices),
        })
    return bins


def log_kmeans_bins(degrees: np.ndarray, num_bins: int) -> List[dict]:
    """Partition vertices into bins via K-Means on log-degrees."""
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
        bins.append({
            "beta_star": float(degrees[indices].min()),
            "indices": indices,
            "count": len(indices),
        })
    return sorted(bins, key=lambda b: b["beta_star"])


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

def edge_source_weights(W: Tensor, P: Tensor) -> Tensor:
    """M_{iℓ}(P) = Σ_j W_{ij} P_{iℓ}(1 − P_{jℓ})."""
    return P * torch.mm(W, 1.0 - P)


# ---------------------------------------------------------------------------
# RatioCut — Theorem 1 (homogeneous β = 1)
# ---------------------------------------------------------------------------

class RatioCutLoss(nn.Module):
    """Probabilistic RatioCut with hypergeometric envelope.

    H_ℓ = ₂F₁(-m, 1; 2; ᾱ_ℓ) — same for every vertex (β=1).
    """

    def __init__(self, n: int, ema_decay: float = 0.0) -> None:
        super().__init__()
        self.n = n
        self.ema_decay = ema_decay

    def forward(
        self, W: Tensor, P: Tensor, alpha_ema: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        K = P.shape[1]
        device = P.device

        alpha_bar = P.detach().mean(0)
        if alpha_ema is not None and self.ema_decay > 0:
            alpha_bar = self.ema_decay * alpha_ema + (1 - self.ema_decay) * alpha_bar

        H = _hyp2f1(-self.n, 1.0, 2.0, alpha_bar.clamp(1e-7, 1 - 1e-7))

        M = edge_source_weights(W, P)
        loss = (M.sum(0) * H).sum() / W.sum().clamp(min=1e-9)

        return loss, alpha_bar


# ---------------------------------------------------------------------------
# NCut — Theorem 2 (Hölder bound for heterogeneous β = d_i)
# ---------------------------------------------------------------------------

class NCutLoss(nn.Module):
    """Probabilistic NCut with Hölder-binned hypergeometric envelope.

    For each source vertex i with degree d_i, the envelope is:
        Φ_ℓ(d_i) = ∏_j [ (1/d_i) · ₂F₁(-m, 1; d_i/β*_j+1; ᾱ_{ℓj}) ]^{w_j}

    This varies per vertex, unlike RatioCut.

    Args:
        degrees: Vertex degrees, shape (n,).
        num_bins: Number of Hölder bins.
        binning: 'equal' or 'log_kmeans'.
        ema_decay: EMA decay for per-bin ᾱ tracking.
    """

    def __init__(
        self,
        degrees: np.ndarray,
        num_bins: int = 16,
        binning: str = "equal",
        ema_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.ema_decay = ema_decay
        n = len(degrees)

        if binning == "equal":
            self.bins = equal_size_bins(degrees, num_bins)
        elif binning == "log_kmeans":
            self.bins = log_kmeans_bins(degrees, num_bins)
        else:
            raise ValueError(f"Unknown binning: {binning}")

        d = len(self.bins)

        self.register_buffer("degrees_t", torch.tensor(degrees, dtype=torch.float32))

        beta_stars = torch.tensor(
            [b["beta_star"] for b in self.bins], dtype=torch.float32
        )
        self.register_buffer("beta_stars", beta_stars)

        counts = torch.tensor([b["count"] for b in self.bins], dtype=torch.float32)
        self.register_buffer("bin_weights", counts / counts.sum())

        # Node-to-bin mapping
        node_to_bin = np.zeros(n, dtype=np.int64)
        for j, b in enumerate(self.bins):
            node_to_bin[b["indices"]] = j
        self.register_buffer("node_to_bin", torch.tensor(node_to_bin, dtype=torch.long))

        self._bin_indices = [
            torch.tensor(b["indices"], dtype=torch.long) for b in self.bins
        ]

    def _bin_means(self, P: Tensor) -> Tensor:
        """Per-bin mean assignments ᾱ_{ℓj}, shape (d, K)."""
        d = len(self.bins)
        K = P.shape[1]
        alpha_bars = torch.zeros(d, K, device=P.device, dtype=P.dtype)
        for j, idx in enumerate(self._bin_indices):
            idx = idx.to(P.device)
            alpha_bars[j] = P[idx].mean(0)
        return alpha_bars

    def compute_phi(
        self,
        q: Tensor,
        alpha_bars: Tensor,
        m: int,
    ) -> Tensor:
        """Compute per-vertex Hölder envelope Φ_ℓ(q_i).

        Args:
            q: Per-vertex degrees, shape (V,). Can be a subset of vertices.
            alpha_bars: Per-bin means, shape (d, K).
            m: Polynomial degree for ₂F₁.

        Returns:
            Phi: shape (V, K).
        """
        V = q.shape[0]
        d, K = alpha_bars.shape
        device = q.device

        beta = self.beta_stars.to(device)  # (d,)
        w = self.bin_weights.to(device)  # (d,)

        # c = q_i / β*_j + 1  — shape (V, d)
        c = q.unsqueeze(1) / beta.unsqueeze(0) + 1.0  # (V, d)

        # z = ᾱ_{ℓj} — shape (d, K)
        z = alpha_bars.clamp(1e-7, 1 - 1e-7)

        # Broadcast to (V, d, K):
        # c: (V, d, 1), z: (1, d, K)
        c_3d = c.unsqueeze(2).expand(V, d, K)
        z_3d = z.unsqueeze(0).expand(V, d, K)

        # ₂F₁(-m, 1; c; z) — shape (V, d, K)
        F = _hyp2f1(-m, 1.0, c_3d, z_3d)

        # H = (1/q) · F — shape (V, d, K)
        H = F / q.view(V, 1, 1)

        # Hölder composition in log space: log Φ = Σ_j w_j · log H_j
        log_H = torch.log(H.clamp(min=1e-30))
        w_3d = w.view(1, d, 1)  # (1, d, 1)
        log_Phi = (w_3d * log_H).sum(dim=1)  # (V, K)

        return torch.exp(log_Phi)

    def forward(
        self,
        W: Tensor,
        P: Tensor,
        alpha_bars_ema: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute NCut envelope loss with per-vertex Φ.

        Args:
            W: Adjacency matrix, shape (n, n).
            P: Assignment probs, shape (n, K).
            alpha_bars_ema: Running per-bin means, shape (d, K).

        Returns:
            (loss, updated_alpha_bars).
        """
        n, K = P.shape
        device = P.device

        # Per-bin means
        alpha_bars = self._bin_means(P.detach())
        if alpha_bars_ema is not None and self.ema_decay > 0:
            alpha_bars = (
                self.ema_decay * alpha_bars_ema + (1 - self.ema_decay) * alpha_bars
            )

        # Per-vertex degrees as query values
        q = self.degrees_t.to(device).clamp(min=1e-6)  # (n,)

        m = n  # use dataset size as m for the full-graph formulation

        # Φ_ℓ(d_i) for each vertex — shape (n, K)
        Phi = self.compute_phi(q, alpha_bars, m)

        # M_{iℓ} · Φ_ℓ(d_i)
        M = edge_source_weights(W, P)  # (n, K)
        loss = (M * Phi).sum() / W.sum().clamp(min=1e-9)

        return loss, alpha_bars


# ---------------------------------------------------------------------------
# Edge-pair interface for minibatch training
# ---------------------------------------------------------------------------

def compute_ncut_edge_phi(
    degrees_left: Tensor,
    alpha_bars: Tensor,
    beta_stars: Tensor,
    bin_weights: Tensor,
    m: int,
) -> Tensor:
    """Compute Φ_ℓ(d_i) for source vertices in an edge batch.

    This is the core function used in the edge-pair training loop.

    Args:
        degrees_left: Degrees of left (source) vertices, shape (E,).
        alpha_bars: Per-bin mean assignments, shape (d, K).
        beta_stars: Per-bin representative exponents, shape (d,).
        bin_weights: Hölder weights, shape (d,).
        m: Polynomial degree.

    Returns:
        Phi: shape (E, K) — one envelope value per edge per cluster.
    """
    E = degrees_left.shape[0]
    d, K = alpha_bars.shape

    q = degrees_left.clamp(min=1e-6)  # (E,)

    # c = q / β* + 1 — shape (E, d)
    c = q.unsqueeze(1) / beta_stars.unsqueeze(0) + 1.0

    z = alpha_bars.clamp(1e-7, 1 - 1e-7)  # (d, K)

    # Broadcast to (E, d, K)
    c_3d = c.unsqueeze(2).expand(E, d, K)
    z_3d = z.unsqueeze(0).expand(E, d, K)

    F = _hyp2f1(-m, 1.0, c_3d, z_3d)  # (E, d, K)
    H = F / q.view(E, 1, 1)

    log_H = torch.log(H.clamp(min=1e-30))
    w = bin_weights.view(1, d, 1)
    log_Phi = (w * log_H).sum(dim=1)  # (E, K)

    return torch.exp(log_Phi)
