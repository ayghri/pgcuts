"""sklearn-like clustering interface for PGCuts.

Usage:
    from pgcuts import HyCut

    model = HyCut(n_clusters=10)
    labels = model.fit_predict(X)
"""
from typing import Literal
import numpy as np
import torch
from torch import nn

from .graph import build_rbf_knn_graph
from .losses.pncut import log_kmeans_bins
from .optim import GradientMixer
from .algorithms.cuts import (
    prcut_step,
    hyp_rcut_step,
    hyp_ncut_step,
)


class HyCut:
    """Unsupervised clustering via probabilistic graph cuts.

    Builds a KNN similarity graph on the input features,
    then optimizes a differentiable upper bound on the
    Normalized Cut using a linear model with edge-pair
    sampling.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    objective : {'hyp_ncut', 'hyp_rcut', 'prcut'}
        Which graph cut objective to optimize.
    n_neighbors : int
        Number of nearest neighbors for the KNN graph.
    steps : int
        Number of optimization steps.
    batch_size : int
        Number of edges sampled per step.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for AdamW.
    m : int
        Polynomial degree for the hypergeometric bound.
    num_bins : int
        Number of degree bins (hyp_ncut only).
    tau_start : float
        Initial temperature for softmax.
    tau_end : float
        Final temperature for softmax.
    ema : float
        EMA decay for cluster proportion tracking.
    device : str
        Device to run on ('cuda' or 'cpu').
    seed : int
        Random seed.

    Examples
    --------
    >>> from pgcuts import HyCut
    >>> model = HyCut(n_clusters=10)
    >>> labels = model.fit_predict(X)

    >>> model = HyCut(
    ...     n_clusters=100,
    ...     objective='hyp_ncut',
    ...     steps=3000,
    ... )
    >>> labels = model.fit_predict(X)
    >>> print(model.ncut_)
    """

    def __init__(
        self,
        n_clusters: int,
        objective: Literal[
            "hyp_ncut", "hyp_rcut", "prcut"
        ] = "hyp_ncut",
        n_neighbors: int = 50,
        steps: int = 1500,
        batch_size: int = 8192,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        m: int = 512,
        num_bins: int = 16,
        tau_start: float = 10.0,
        tau_end: float = 1.0,
        distance: Literal["xor", "ce"] = "ce",
        ema: float = 0.9,
        device: str = "cuda",
        seed: int = 42,
    ):
        """Initialize HyCut with hyperparameters."""
        self.n_clusters = n_clusters
        self.objective = objective
        self.n_neighbors = n_neighbors
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.m = m
        self.distance = distance
        self.num_bins = num_bins
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.ema = ema
        self.device = device
        self.seed = seed

        # Fitted attributes
        self.labels_ = None
        self.ncut_ = None
        self.rcut_ = None
        self.model_ = None
        self.graph_ = None

    def fit(self, features: np.ndarray) -> "HyCut":
        """Fit the model to data.

        Parameters
        ----------
        features : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        self
        """
        features = np.asarray(features, dtype=np.float32)
        num_samples, num_dims = features.shape
        num_clusters = self.n_clusters
        device = torch.device(self.device)

        # Build graph
        graph = build_rbf_knn_graph(
            features,
            n_neighbors=min(
                self.n_neighbors, num_samples - 1
            ),
        )
        self.graph_ = graph
        degrees = np.array(
            graph.sum(1)
        ).flatten().astype(np.float32)

        # Preload on GPU
        features_t = torch.tensor(
            features, dtype=torch.float32, device=device
        )
        pairs = np.array(graph.nonzero()).T
        n_edges = pairs.shape[0]
        edge_w = np.array(
            graph[pairs[:, 0], pairs[:, 1]]
        ).flatten().astype(np.float32)
        pairs_gpu = torch.tensor(
            pairs, dtype=torch.long, device=device
        )
        edge_w_gpu = torch.tensor(
            edge_w, device=device
        )
        degrees_t = torch.tensor(
            degrees, dtype=torch.float32, device=device
        )

        # Binning (for hyp_ncut)
        bins = log_kmeans_bins(degrees, self.num_bins)
        n_bins = len(bins)
        bin_weights = torch.tensor(
            [b["count"] for b in bins],
            dtype=torch.float32,
            device=device,
        )
        bin_weights = bin_weights / bin_weights.sum()
        beta_stars = torch.tensor(
            [b["beta_star"] for b in bins],
            dtype=torch.float32,
            device=device,
        )
        q_stars = torch.tensor(
            [
                degrees[b["indices"]].mean()
                for b in bins
            ],
            dtype=torch.float32,
            device=device,
        )
        node_to_bin = torch.zeros(
            num_samples, dtype=torch.long, device=device
        )
        for j, b in enumerate(bins):
            node_to_bin[
                torch.tensor(
                    b["indices"],
                    dtype=torch.long,
                    device=device,
                )
            ] = j

        # Model
        torch.manual_seed(self.seed)
        model = nn.Linear(num_dims, num_clusters).to(
            device
        )
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.steps, eta_min=1e-5
        )
        gm = GradientMixer(
            list(model.named_parameters()),
            loss_scale={"cut": 1.0, "balance": 1.0},
        )
        p_ema = (
            torch.ones(num_clusters, device=device)
            / num_clusters
        )
        alpha_ema = (
            torch.ones(
                n_bins, num_clusters, device=device
            )
            / num_clusters
        )

        # Train
        for step in range(1, self.steps + 1):
            tau = self.tau_start + (
                self.tau_end - self.tau_start
            ) * (step / self.steps)

            idx = torch.randint(
                n_edges,
                (self.batch_size,),
                device=device,
            )
            bp = pairs_gpu[idx]
            w = edge_w_gpu[idx]
            all_nodes = torch.cat(
                [bp[:, 0], bp[:, 1]]
            )
            unique_nodes, inv = torch.unique(
                all_nodes, return_inverse=True
            )
            left_idx = inv[: self.batch_size]
            right_idx = inv[self.batch_size :]

            logits = (
                model(features_t[unique_nodes]) / tau
            )

            if self.objective == "prcut":
                probs = torch.softmax(logits, dim=-1)
                cut_loss, balance, p_ema = prcut_step(
                    probs,
                    left_idx,
                    right_idx,
                    w,
                    p_ema,
                    self.ema,
                )
            elif self.objective == "hyp_rcut":
                cut_loss, balance, p_ema = hyp_rcut_step(
                    logits,
                    left_idx,
                    right_idx,
                    w,
                    p_ema,
                    self.m,
                    self.ema,
                )
            elif self.objective == "hyp_ncut":
                cut_loss, balance, alpha_ema = (
                    hyp_ncut_step(
                        logits,
                        left_idx,
                        right_idx,
                        w,
                        bp[:, 0],
                        alpha_ema,
                        q_stars,
                        beta_stars,
                        bin_weights,
                        node_to_bin,
                        degrees_t,
                        self.m,
                        self.ema,
                    )
                )
            else:
                raise ValueError(
                    f"Unknown objective "
                    f"'{self.objective}'. "
                    "Expected one of: 'hyp_ncut', "
                    "'hyp_rcut', 'prcut'."
                )

            opt.zero_grad()
            with gm("cut"):
                cut_loss.backward(retain_graph=True)
            with gm("balance"):
                balance.backward()
            opt.step()
            sched.step()

        # Predict
        with torch.no_grad():
            self.labels_ = (
                model(features_t)
                .argmax(dim=-1)
                .cpu()
                .numpy()
            )

        self.model_ = model

        # Compute cut metrics
        from .metrics import (  # pylint: disable=import-outside-toplevel
            compute_rcut_ncut,
        )

        self.rcut_, self.ncut_ = compute_rcut_ncut(
            graph, self.labels_
        )

        return self

    def fit_predict(
        self, features: np.ndarray
    ) -> np.ndarray:
        """Fit and return cluster labels.

        Parameters
        ----------
        features : array-like of shape
            (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        return self.fit(features).labels_

    def predict(
        self, features: np.ndarray
    ) -> np.ndarray:
        """Predict cluster labels for new data.

        Parameters
        ----------
        features : array-like of shape
            (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() first")
        features_t = torch.tensor(
            np.asarray(features, dtype=np.float32),
            device=next(
                self.model_.parameters()
            ).device,
        )
        with torch.no_grad():
            return (
                self.model_(features_t)
                .argmax(dim=-1)
                .cpu()
                .numpy()
            )
