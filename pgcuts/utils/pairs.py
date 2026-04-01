"""Pair generation utilities for PGCuts."""
from typing import Tuple
import numpy as np


def generate_unique_lower_pairs_sparse(
    w_mat, b: int
):
    """Generate pair sampler for sparse matrix.

    Args:
        w_mat: Sparse adjacency matrix.
        b: Number of pairs to sample.

    Returns:
        Callable that returns sampled pairs.
    """
    nnz_coords = np.array(w_mat.nonzero()).T
    nnz_coords = nnz_coords[
        nnz_coords[:, 0] < nnz_coords[:, 1]
    ]
    num_nnz = nnz_coords.shape[0]

    def get_unique_lower_pairs_sparse():
        """Sample unique lower-triangular pairs."""
        samples = np.random.choice(
            num_nnz, size=b, replace=False
        )
        return nnz_coords[samples]

    return get_unique_lower_pairs_sparse


def _strict_upper_tri_to_coords(k):
    """Convert flat index to upper-tri coordinates.

    Args:
        k: Flat index into upper triangle.

    Returns:
        (row, col) coordinate arrays.
    """
    j = np.floor(
        (1.0 + np.sqrt(1.0 + 8.0 * k)) / 2.0
    ).astype(int)
    num_elements = j * (j - 1) // 2
    return k - num_elements, j


def get_unique_lower_pairs(
    n: int, b: int
) -> np.ndarray:
    """Generate b random unique pairs (i, j), i < j.

    Args:
        n: Range upper bound.
        b: Number of pairs.

    Returns:
        Array of shape (b, 2).
    """
    max_possible_pairs = n * (n - 1) // 2

    if (
        b <= 0
        or b > max_possible_pairs
        or n < 2
    ):
        raise ValueError(
            "Number of pairs 'b' cannot be <=0 "
            "or > n(n-1)/2, and must be n>=2."
        )

    k_sampled = np.random.choice(
        max_possible_pairs, size=b, replace=False
    )
    i_vals, j_vals = _strict_upper_tri_to_coords(
        k_sampled
    )

    pairs_array = np.stack(
        (i_vals, j_vals), axis=1
    )

    return pairs_array


def get_pairs_unique_map(
    pairs: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Return unique indices and inverse maps.

    Args:
        pairs: Array of (i,j) pairs, shape (k, 2).

    Returns:
        Tuple of (unique, left_inv, right_inv).
    """
    all_indices = np.concatenate(
        (pairs[:, 0], pairs[:, 1])
    )

    unique_indices, inverse_indices = np.unique(
        all_indices, return_inverse=True
    )
    k = pairs.shape[0]

    return (
        unique_indices,
        inverse_indices[:k],
        inverse_indices[k:],
    )


if __name__ == "__main__":
    import math

    n_elements = 10
    num_pairs_to_generate = 5

    try:
        random_pairs = get_unique_lower_pairs(
            n_elements, num_pairs_to_generate
        )
        print(
            f"Generated {len(random_pairs)} "
            f"unique ordered pairs from "
            f"range [0, {n_elements - 1}]:"
        )
        print(random_pairs)

        uniq, inverse_left, inverse_right = (
            get_pairs_unique_map(random_pairs)
        )
        print(f"Unique values involved: {uniq}")
        print(
            "inverse maps: "
            f"{inverse_left, inverse_right}"
        )
        assert np.all(
            random_pairs[:, 0]
            == uniq[inverse_left]
        )
        assert np.all(
            random_pairs[:, 1]
            == uniq[inverse_right]
        )

        n_small = 4
        max_b = math.comb(n_small, 2)
        all_pairs = get_unique_lower_pairs(
            n_small, max_b
        )
        print(
            f"\nAll possible pairs for n={n_small}"
            f" (should be {max_b}):"
        )
        print(
            np.array(
                sorted(
                    [tuple(a) for a in all_pairs]
                )
            )
        )

        zero_pairs = get_unique_lower_pairs(10, 0)
        print(
            f"\nGenerating 0 pairs: {zero_pairs}"
        )

    except ValueError as e:
        print(f"\nError: {e}")

    _w_test = np.random.randint(
        0, 2, size=(100, 100)
    )
