"""Clustering evaluation metrics."""
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score
import sklearn.metrics as metrics
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple

from munkres import Munkres


def unsupervised_contingency(
    true_labels, cluster_labels
):
    """Compute contingency matrix with optimal matching.

    Args:
        true_labels: True labels.
        cluster_labels: Cluster labels.

    Returns:
        Tuple of (matrix, row_ind, col_ind).
    """
    true_labels = np.array(true_labels)
    cluster_labels = np.array(cluster_labels)

    unique_true = np.unique(true_labels)
    unique_cluster = np.unique(cluster_labels)

    contingency_matrix = np.zeros(
        (len(unique_true), len(unique_cluster))
    )

    for i, true_label in enumerate(unique_true):
        for j, cluster_label in enumerate(
            unique_cluster
        ):
            contingency_matrix[i, j] = np.sum(
                (true_labels == true_label)
                & (cluster_labels == cluster_label)
            )

    row_ind, col_ind = linear_sum_assignment(
        -contingency_matrix
    )
    contingency_matrix = contingency_matrix[
        row_ind, :
    ][:, col_ind]

    return contingency_matrix, row_ind, col_ind


def unsupervised_accuracy(
    true_labels, cluster_labels
):
    """Compute unsupervised accuracy.

    Args:
        true_labels: True labels.
        cluster_labels: Cluster labels.

    Returns:
        Unsupervised accuracy score.
    """
    contingency_matrix, row_ind, col_ind = (
        unsupervised_contingency(
            true_labels, cluster_labels
        )
    )

    optimal_matching_sum = contingency_matrix[
        row_ind, col_ind
    ].sum()
    return optimal_matching_sum / len(true_labels)


def nmi_score(true_labels, cluster_labels):
    """Compute normalized mutual information.

    Args:
        true_labels: True labels.
        cluster_labels: Cluster labels.

    Returns:
        NMI score.
    """
    return normalized_mutual_info_score(
        true_labels,
        cluster_labels,
        average_method="geometric",
    )


def get_cluster_labels_from_indices(
    indices: np.ndarray,
) -> np.ndarray:
    """Get cluster labels from indices.

    Args:
        indices: Indices of the clusters.

    Returns:
        Cluster labels array.
    """
    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def calculate_cost_matrix(
    confusion: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Calculate cost matrix for Munkres algorithm.

    Args:
        confusion: Confusion matrix.
        n_clusters: Number of clusters.

    Returns:
        Cost matrix.
    """
    cost_matrix = np.zeros(
        (n_clusters, n_clusters)
    )
    for j in range(n_clusters):
        s = np.sum(confusion[:, j])
        for i in range(n_clusters):
            t = confusion[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def assign_clusters(
    true_labels: np.ndarray,
    cluster_assignments: np.ndarray,
    num_classes: int,
    num_clusters: int = -1,
) -> np.ndarray:
    """Assign clusters based on optimal matching.

    Args:
        true_labels: True labels.
        cluster_assignments: Predicted clusters.
        num_classes: Number of true classes.
        num_clusters: Number of predicted clusters.

    Returns:
        Transformed cluster assignments.
    """
    unique_clusters_pred = np.unique(
        cluster_assignments
    )
    if num_clusters < 0:
        num_clusters = unique_clusters_pred.shape[0]

    cluster_to_index = dict(
        zip(
            unique_clusters_pred,
            np.arange(unique_clusters_pred.shape[0]),
        )
    )
    offset_assignments = np.array(
        [
            cluster_to_index[c]
            for c in cluster_assignments
        ]
    )
    confusion_matrix = (
        np.eye(num_clusters)[offset_assignments]
        .T.dot(np.eye(num_classes)[true_labels])
    )

    cluster_to_label = np.argmax(
        confusion_matrix, axis=1
    )
    return np.array(
        [
            cluster_to_label[c]
            for c in offset_assignments
        ]
    )


def assign_unique_clusters(
    true_labels: np.ndarray,
    cluster_assignments: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    """Assign clusters using Munkres algorithm.

    Args:
        true_labels: True labels.
        cluster_assignments: Predicted clusters.
        num_clusters: Number of clusters.

    Returns:
        Transformed predictions.
    """
    c_matrix = metrics.confusion_matrix(
        true_labels, cluster_assignments, labels=None
    )
    cost_matrix = calculate_cost_matrix(
        c_matrix, n_clusters=num_clusters
    )
    indices = np.array(
        Munkres().compute(cost_matrix)
    )
    kmeans_to_true = get_cluster_labels_from_indices(
        indices
    )
    y_pred = kmeans_to_true[cluster_assignments]
    return y_pred


def evaluate_clustering(
    true_labels: np.ndarray,
    cluster_assignments: np.ndarray,
    num_classes: int,
    num_clusters: int = -1,
) -> Dict:
    """Evaluate clustering performance.

    Args:
        true_labels: True labels.
        cluster_assignments: Predicted clusters.
        num_classes: Number of true classes.
        num_clusters: Number of predicted clusters.

    Returns:
        Dict with accuracy, NMI, confusion matrix.
    """
    assert (
        true_labels.shape
        == cluster_assignments.shape
    )
    if num_clusters == num_classes:
        predicted_labels = assign_clusters(
            true_labels,
            cluster_assignments,
            num_clusters,
        )
    else:
        predicted_labels = assign_clusters(
            true_labels,
            cluster_assignments,
            num_classes,
            num_clusters,
        )
    accuracy = metrics.accuracy_score(
        true_labels, predicted_labels
    )
    nmi = metrics.normalized_mutual_info_score(
        true_labels, predicted_labels
    )
    confusion_matrix = metrics.confusion_matrix(
        true_labels, predicted_labels
    )
    return {
        "accuracy": accuracy,
        "nmi": nmi,
        "confusion_matrix": confusion_matrix,
    }


def cluster_acc_score(y_true, y_pred):
    """Compute cluster accuracy score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy score.
    """
    y_true = y_true.astype(np.int64)
    dim = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((dim, dim), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(
        w.max() - w
    )
    return w[row_ind, col_ind].sum() / y_pred.size


def ari_score(y_true, y_pred):
    """Compute adjusted Rand index.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        ARI score.
    """
    return metrics.adjusted_rand_score(
        y_true, y_pred
    )


def ratio_cut_score(w_mat, y, num_clusters):
    """Compute ratio cut value for a graph.

    Args:
        w_mat: Adjacency matrix.
        y: Cluster assignments.
        num_clusters: Number of clusters.

    Returns:
        Ratio cut value.
    """
    indicator = np.eye(num_clusters)[y]
    a_mat = w_mat.dot(indicator)
    return np.sum(
        np.diag(
            (1 - indicator).T.dot(a_mat)
        )
        / indicator.sum(0)
    )


def compute_rcut_ncut(
    w_mat, labels
) -> Tuple[float, float]:
    """Compute ratio cut and normalized cut values.

    Args:
        w_mat: Sparse or dense weight matrix.
        labels: Integer cluster labels.

    Returns:
        (rcut, ncut) tuple.
    """
    n = w_mat.shape[0]
    if labels.min() < 0:
        return np.nan, np.nan

    num_clusters = int(labels.max()) + 1

    # One-hot encoding matrix (sparse)
    row = np.arange(n)
    col = labels
    data = np.ones(n)
    indicator = sp.csr_matrix(
        (data, (row, col)), shape=(n, num_clusters)
    )

    # Degree vector
    d = np.array(w_mat.sum(axis=1)).flatten()

    # Volumes: sum of degrees in each cluster
    vol = indicator.T @ d

    # Sizes: number of nodes in each cluster
    sizes = np.array(
        indicator.sum(axis=0)
    ).flatten()

    # Association: diag(A^T W A)
    wa = w_mat @ indicator
    assoc = np.array(
        indicator.multiply(wa).sum(axis=0)
    ).flatten()

    cut = vol - assoc

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):
        rcut_k = cut / sizes
        ncut_k = cut / vol

    rcut_k = np.where(sizes > 0, rcut_k, 1.0)
    ncut_k = np.where(vol > 0, ncut_k, 1.0)

    rcut = float(np.nansum(rcut_k))
    ncut = float(np.nansum(ncut_k))

    return rcut, ncut


def soft_ncut(w_sparse, probs, degrees=None):
    """Soft (expected) normalized cut.

    Args:
        w_sparse: Sparse adjacency matrix (n, n).
        probs: Soft assignments, shape (n, K).
        degrees: Optional precomputed degrees.

    Returns:
        Soft NCut value (float).
    """
    if degrees is None:
        degrees = np.array(
            w_sparse.sum(axis=1)
        ).flatten()

    w_one_minus_p = w_sparse @ (1.0 - probs)
    cut = np.array(
        (probs * w_one_minus_p).sum(axis=0)
    ).flatten()

    vol = (
        degrees[:, None] * probs
    ).sum(axis=0)

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):
        ncut_k = cut / vol

    ncut_k = np.where(vol > 0, ncut_k, 1.0)
    return float(np.nansum(ncut_k))


def soft_rcut(w_sparse, probs):
    """Soft (expected) ratio cut.

    Args:
        w_sparse: Sparse adjacency matrix (n, n).
        probs: Soft assignments, shape (n, K).

    Returns:
        Soft RCut value (float).
    """
    w_one_minus_p = w_sparse @ (1.0 - probs)
    cut = np.array(
        (probs * w_one_minus_p).sum(axis=0)
    ).flatten()
    size = probs.sum(axis=0)

    with np.errstate(
        divide="ignore", invalid="ignore"
    ):
        rcut_k = cut / size

    rcut_k = np.where(size > 0, rcut_k, 1.0)
    return float(np.nansum(rcut_k))
