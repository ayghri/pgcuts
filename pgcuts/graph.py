"""Graph construction utilities for PGCuts."""
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import (
    kneighbors_graph,
    NearestNeighbors,
)
import torch


def symmetrize(w_mat: sp.spmatrix) -> sp.csr_matrix:
    """Symmetrize a sparse matrix: (W + W^T) / 2."""
    return sp.csr_matrix((w_mat + w_mat.T) / 2)


def knn_graph(
    features: np.ndarray,
    n_neighbors: int = 10,
    mode: str = "distance",
    metric: str = "minkowski",
) -> sp.csr_matrix:
    """Build a symmetric sparse KNN graph.

    Args:
        features: Data matrix, shape (n, d).
        n_neighbors: Number of nearest neighbors.
        mode: "distance" or "connectivity".
        metric: Distance metric.

    Returns:
        Symmetric sparse distance/connectivity matrix.
    """
    w_mat = kneighbors_graph(
        features,
        n_neighbors=n_neighbors,
        mode=mode,
        metric=metric,
    )
    return symmetrize(w_mat)


def gaussian_rbf_kernel(
    distances: sp.spmatrix, sigma: float = None
) -> sp.csr_matrix:
    """Convert sparse distances to Gaussian RBF.

    Args:
        distances: Sparse distance matrix.
        sigma: Bandwidth. If None, uses median of max
            distances per row.

    Returns:
        Sparse similarity matrix.
    """
    w_mat = sp.csr_matrix(distances.copy())
    if sigma is None:
        maxes = (
            distances.max(axis=1).toarray().flatten()
        )
        sigma = (
            float(np.median(maxes[maxes > 0]))
            if np.any(maxes > 0)
            else 1.0
        )
    w_mat.data = np.exp(
        -(w_mat.data ** 2) / (sigma ** 2)
    )
    return symmetrize(w_mat)


def build_rbf_knn_graph(
    features: np.ndarray,
    n_neighbors: int = 100,
) -> sp.csr_matrix:
    """Build an RBF-weighted KNN graph.

    Pipeline: KNN distances -> row-normalize ->
    scale by degree -> Gaussian RBF -> symmetrize.

    Args:
        features: Data matrix, shape (n, d).
        n_neighbors: Number of nearest neighbors.

    Returns:
        Sparse symmetric similarity matrix.
    """
    distances = kneighbors_graph(
        features, n_neighbors=n_neighbors, mode="distance"
    )
    if not sp.issparse(distances):
        distances = sp.csr_matrix(distances)

    # Symmetrize distances
    w_mat = (distances + distances.T) / 2

    # Row-normalize
    w_norm = w_mat / w_mat.sum(axis=1)

    # Count neighbors per node and scale
    connect = w_mat.copy()
    connect.data = np.ones_like(w_mat.data)
    total_neighbors = sp.diags(
        np.array(connect.sum(axis=1).flatten())[0], 0
    )
    w_exp = w_norm @ total_neighbors

    # Apply Gaussian RBF
    w_exp.data = np.exp(-(w_exp.data ** 2) / 2.0)

    # Final symmetrization
    w_exp = (w_exp + w_exp) / 2

    return sp.csr_matrix(w_exp)


def build_knn_graph_gpu(
    features: torch.Tensor,
    n_neighbors: int = 50,
    sigma_mode: str = "median",
    sigma_scale: float = 1.0,
) -> torch.Tensor:
    """Build a sparse RBF KNN graph on GPU.

    Args:
        features: Feature tensor on GPU, shape (N, D).
        n_neighbors: Number of nearest neighbors.
        sigma_mode: How to compute bandwidth.
        sigma_scale: Multiply sigma by this factor.

    Returns:
        Sparse COO tensor (N, N) on same device.
    """
    from torch_cluster import (  # pylint: disable=import-outside-toplevel
        knn_graph as _knn_graph,
    )

    num_nodes = features.shape[0]

    # KNN on GPU -- returns (2, N*k) edge_index
    edge_index = _knn_graph(
        features, k=n_neighbors, loop=False
    )
    src, dst = edge_index[0], edge_index[1]

    # Compute distances
    dists = (features[src] - features[dst]).norm(dim=-1)

    # Symmetrize: add reverse edges
    all_src = torch.cat([src, dst])
    all_dst = torch.cat([dst, src])
    all_dists = torch.cat([dists, dists])

    # RBF kernel
    if sigma_mode == "none":
        weights = torch.ones_like(all_dists)
    else:
        if sigma_mode == "median":
            sigma = all_dists.median()
        elif sigma_mode == "mean":
            sigma = all_dists.mean()
        else:
            raise ValueError(
                f"Unknown sigma_mode: {sigma_mode}"
            )
        sigma = sigma * sigma_scale
        weights = torch.exp(
            -(all_dists ** 2) / (2 * sigma ** 2)
        )

    # Build sparse tensor
    indices = torch.stack([all_src, all_dst])
    w_mat = torch.sparse_coo_tensor(
        indices, weights, (num_nodes, num_nodes)
    ).coalesce()

    return w_mat


def get_knn_distances(
    x_left: np.ndarray,
    x_right: np.ndarray,
    num_neighbors: int,
    mode: str = "distance",
    metric: str = "minkowski",
) -> sp.csr_matrix:
    """Compute mutual KNN distances between two sets.

    Args:
        x_left: Left point set, shape (n1, d).
        x_right: Right point set, shape (n2, d).
        num_neighbors: Number of neighbors.
        mode: "distance" or "connectivity".
        metric: Distance metric.

    Returns:
        Symmetric sparse distance matrix.
    """
    affinity_1 = (
        NearestNeighbors(
            n_neighbors=num_neighbors, metric=metric
        )
        .fit(x_right)
        .kneighbors_graph(x_left, mode=mode)
    )
    affinity_2 = (
        NearestNeighbors(n_neighbors=num_neighbors)
        .fit(x_left)
        .kneighbors_graph(x_right, mode=mode)
        .T
    )
    return sp.csr_matrix(
        0.5 * (affinity_1 + affinity_2)
    )


def compute_sp_similarities(
    affinity_mat: sp.spmatrix,
) -> sp.csr_matrix:
    """Convert sparse distances to Gaussian similarities.

    Args:
        affinity_mat: Sparse distance matrix.

    Returns:
        Sparse similarity matrix.
    """
    maxes = (
        affinity_mat.max(axis=1).toarray().flatten()
    )
    sigma = (
        float(np.median(maxes[maxes > 0]))
        if np.any(maxes > 0)
        else 1.0
    )
    similarity_mat = (
        affinity_mat + affinity_mat.T
    ) / 2
    similarity_mat = sp.csr_matrix(similarity_mat)
    similarity_mat.data = np.exp(
        -(similarity_mat.data ** 2) / sigma ** 2
    )
    return similarity_mat


def sp_knn_similarity(
    x_left: np.ndarray,
    x_right: np.ndarray,
    num_neighbors: int,
    mode: str = "distance",
    metric: str = "minkowski",
) -> sp.csr_matrix:
    """Build sparse KNN similarity between two sets.

    Args:
        x_left: Left point set, shape (n1, d).
        x_right: Right point set, shape (n2, d).
        num_neighbors: Number of neighbors.
        mode: "distance" or "connectivity".
        metric: Distance metric.

    Returns:
        Sparse similarity matrix.
    """
    w_mat = get_knn_distances(
        x_left, x_right, num_neighbors, mode, metric
    )
    return compute_sp_similarities(w_mat)


def torch_knn_similarity(
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    num_neighbors: int,
    mode: str = "distance",
    metric: str = "minkowski",
) -> torch.Tensor:
    """Compute KNN similarity as a dense torch tensor.

    Args:
        x_left: Left point set tensor.
        x_right: Right point set tensor.
        num_neighbors: Number of neighbors.
        mode: "distance" or "connectivity".
        metric: Distance metric.

    Returns:
        Dense similarity tensor.
    """
    return torch.tensor(
        sp_knn_similarity(
            x_left.cpu().numpy(),
            x_right.cpu().numpy(),
            num_neighbors,
            mode=mode,
            metric=metric,
        ).toarray()
    ).to(x_left)


def torch_pairwise_similarities(
    x1: torch.Tensor,
    x2: torch.Tensor,
    factor: float = 1.0,
) -> torch.Tensor:
    """Dense pairwise Gaussian similarity.

    Args:
        x1: First point set, shape (n1, d).
        x2: Second point set, shape (n2, d).
        factor: Bandwidth scaling factor.

    Returns:
        Similarity matrix, shape (n1, n2).
    """
    distances = torch.cdist(
        x1.unsqueeze(0), x2.unsqueeze(0)
    ).squeeze(0)
    sigma = torch.median(distances.max(1)[0])
    return torch.exp(
        -((distances / sigma / factor) ** 2)
    )


def sparse_laplacian(
    w_mat: sp.spmatrix,
) -> sp.spmatrix:
    """Compute graph Laplacian L = D - W.

    Args:
        w_mat: Sparse similarity/adjacency matrix.

    Returns:
        Sparse Laplacian matrix.
    """
    similarity = (
        compute_sp_similarities(w_mat)
        if not hasattr(w_mat, "data")
        else sp.csr_matrix(w_mat)
    )
    degree_mat = sp.diags(
        np.array(similarity.sum(axis=1)).flatten(), 0
    )
    return degree_mat - similarity
