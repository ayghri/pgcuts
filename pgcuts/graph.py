import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import torch


def symmetrize(W: sp.spmatrix) -> sp.csr_matrix:
    """Symmetrize a sparse matrix: (W + W^T) / 2."""
    return sp.csr_matrix((W + W.T) / 2)


def knn_graph(
    X: np.ndarray,
    n_neighbors: int = 10,
    mode: str = "distance",
    metric: str = "minkowski",
) -> sp.csr_matrix:
    """Build a symmetric sparse KNN graph.

    Args:
        X: Data matrix, shape (n, d).
        n_neighbors: Number of nearest neighbors.
        mode: "distance" or "connectivity".
        metric: Distance metric.

    Returns:
        Symmetric sparse distance/connectivity matrix.
    """
    W = kneighbors_graph(X, n_neighbors=n_neighbors, mode=mode, metric=metric)
    return symmetrize(W)


def gaussian_rbf_kernel(distances: sp.spmatrix, sigma: float = None) -> sp.csr_matrix:
    """Convert a sparse distance matrix to Gaussian RBF similarities.

    Args:
        distances: Sparse distance matrix.
        sigma: Bandwidth. If None, uses the median of max distances per row.

    Returns:
        Sparse similarity matrix.
    """
    W = sp.csr_matrix(distances.copy())
    if sigma is None:
        maxes = distances.max(axis=1).toarray().flatten()
        sigma = float(np.median(maxes[maxes > 0])) if np.any(maxes > 0) else 1.0
    W.data = np.exp(-(W.data**2) / (sigma**2))
    return symmetrize(W)


def build_rbf_knn_graph(
    X: np.ndarray,
    n_neighbors: int = 100,
) -> sp.csr_matrix:
    """Build an RBF-weighted KNN graph (pipeline used in training scripts).

    Pipeline: KNN distances -> row-normalize -> scale by degree -> Gaussian RBF -> symmetrize.

    Args:
        X: Data matrix, shape (n, d).
        n_neighbors: Number of nearest neighbors.

    Returns:
        Sparse symmetric similarity matrix.
    """
    distances = kneighbors_graph(X, n_neighbors=n_neighbors, mode="distance")
    if not sp.issparse(distances):
        distances = sp.csr_matrix(distances)

    # Symmetrize distances
    W = (distances + distances.T) / 2

    # Row-normalize
    W_norm = W / W.sum(axis=1)

    # Count number of neighbors per node and scale
    connect = W.copy()
    connect.data = np.ones_like(W.data)
    total_neighbors = sp.diags(np.array(connect.sum(axis=1).flatten())[0], 0)
    W_exp = W_norm @ total_neighbors

    # Apply Gaussian RBF
    W_exp.data = np.exp(-(W_exp.data**2) / 2.0)

    # Final symmetrization
    W_exp = (W_exp + W_exp) / 2

    return sp.csr_matrix(W_exp)


def get_knn_distances(
    x_left: np.ndarray,
    x_right: np.ndarray,
    num_neighbors: int,
    mode: str = "distance",
    metric: str = "minkowski",
) -> sp.csr_matrix:
    """Compute mutual KNN distances between two sets of points.

    Args:
        x_left: Left point set, shape (n1, d).
        x_right: Right point set, shape (n2, d).
        num_neighbors: Number of neighbors.
        mode: "distance" or "connectivity".
        metric: Distance metric.

    Returns:
        Symmetric sparse distance matrix, shape (n1, n2).
    """
    affinity_1 = (
        NearestNeighbors(n_neighbors=num_neighbors, metric=metric)
        .fit(x_right)
        .kneighbors_graph(x_left, mode=mode)
    )
    affinity_2 = (
        NearestNeighbors(n_neighbors=num_neighbors)
        .fit(x_left)
        .kneighbors_graph(x_right, mode=mode)
        .T
    )
    return sp.csr_matrix(0.5 * (affinity_1 + affinity_2))


def compute_sp_similarities(affinity_mat: sp.spmatrix) -> sp.csr_matrix:
    """Convert a sparse distance matrix to Gaussian similarities with median bandwidth.

    Args:
        affinity_mat: Sparse distance matrix.

    Returns:
        Sparse similarity matrix.
    """
    maxes = affinity_mat.max(axis=1).toarray().flatten()
    sigma = float(np.median(maxes[maxes > 0])) if np.any(maxes > 0) else 1.0
    similarity_mat = (affinity_mat + affinity_mat.T) / 2
    similarity_mat = sp.csr_matrix(similarity_mat)
    similarity_mat.data = np.exp(-(similarity_mat.data**2) / sigma**2)
    return similarity_mat


def sp_knn_similarity(
    x_left: np.ndarray,
    x_right: np.ndarray,
    num_neighbors: int,
    mode: str = "distance",
    metric: str = "minkowski",
) -> sp.csr_matrix:
    """Build a sparse KNN similarity matrix between two point sets.

    Args:
        x_left: Left point set, shape (n1, d).
        x_right: Right point set, shape (n2, d).
        num_neighbors: Number of neighbors.
        mode: "distance" or "connectivity".
        metric: Distance metric.

    Returns:
        Sparse similarity matrix.
    """
    W = get_knn_distances(x_left, x_right, num_neighbors, mode, metric)
    return compute_sp_similarities(W)


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
    x1: torch.Tensor, x2: torch.Tensor, factor: float = 1.0
) -> torch.Tensor:
    """Dense pairwise Gaussian similarity with median bandwidth.

    Args:
        x1: First point set, shape (n1, d).
        x2: Second point set, shape (n2, d).
        factor: Bandwidth scaling factor.

    Returns:
        Similarity matrix, shape (n1, n2).
    """
    distances = torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0)).squeeze(0)
    sigma = torch.median(distances.max(1)[0])
    return torch.exp(-((distances / sigma / factor) ** 2))


def sparse_laplacian(W: sp.spmatrix) -> sp.spmatrix:
    """Compute the graph Laplacian L = D - W from a similarity matrix.

    Args:
        W: Sparse similarity/adjacency matrix.

    Returns:
        Sparse Laplacian matrix.
    """
    S = compute_sp_similarities(W) if not hasattr(W, "data") else sp.csr_matrix(W)
    degree_mat = sp.diags(np.array(S.sum(axis=1)).flatten(), 0)
    return degree_mat - S
