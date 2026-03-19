"""PGCuts -- Probabilistic Graph Cuts."""

from . import hyp2f1, losses, metrics, graph, utils, models, algorithms, data

from .hyp2f1 import Hyp2F1, hyp2f1
from .losses import PRCutGradLoss, PRCutBatchLoss, HyCutLoss, SimplexL2Loss
from .losses.functional import graph_prcut, pairwise_prcut
from .metrics import (
    evaluate_clustering,
    nmi_score,
    ari_score,
    ratio_cut_score,
    compute_rcut_ncut,
)
from .graph import knn_graph, gaussian_rbf_kernel, build_rbf_knn_graph
from .utils import GradientMixer, get_pairs_unique_map
from .data import ShuffledRangeDataset

__all__ = [
    # hyp2f1
    "Hyp2F1",
    "hyp2f1",
    # losses
    "PRCutGradLoss",
    "PRCutBatchLoss",
    "HyCutLoss",
    "SimplexL2Loss",
    "graph_prcut",
    "pairwise_prcut",
    # metrics
    "evaluate_clustering",
    "nmi_score",
    "ari_score",
    "ratio_cut_score",
    "compute_rcut_ncut",
    # graph
    "knn_graph",
    "gaussian_rbf_kernel",
    "build_rbf_knn_graph",
    # utils
    "GradientMixer",
    "get_pairs_unique_map",
    # data
    "ShuffledRangeDataset",
]
