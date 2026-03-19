from typing import Tuple
import numpy as np


def generate_unique_lower_pairs_sparse(W, b: int):
    nnz_coords = np.array(W.nonzero()).T
    nnz_coords = nnz_coords[nnz_coords[:, 0] < nnz_coords[:, 1]]
    num_nnz = nnz_coords.shape[0]

    def get_unique_lower_pairs_sparse():
        samples = np.random.choice(num_nnz, size=b, replace=False)
        return nnz_coords[samples]

    return get_unique_lower_pairs_sparse


def _strict_upper_tri_to_coords(k):
    """
    |-|0|1|3|6|...
    |-|-|2|4|7|...
    |-|-|-|5|8|...
    |-|-|-|-|9|...
    ..............
    """
    # Calculate j (the second element, column index in upper triangle)
    # Use the formula j = floor((1 + sqrt(1 + 8k)) / 2)
    j = np.floor((1.0 + np.sqrt(1.0 + 8.0 * k)) / 2.0).astype(int)
    # Calculate i (the first element, row index within the column)
    # Use the formula i = k - j*(j-1)//2
    num_elements = j * (j - 1) // 2
    return k - num_elements, j


def get_unique_lower_pairs(n: int, b: int) -> np.ndarray:
    """
    Generates b random unique pairs (i, j) such that 0 <= i < j < n.
    """
    max_possible_pairs = n * (n - 1) // 2

    if b <= 0 or b > max_possible_pairs or n < 2:
        raise ValueError(
            "Number of pairs 'b' cannot be <=0 or > n(n-1)/2, and must be n>=2."
        )

    # We're sampling unique elements of the strict upper triangular matrix
    # then getting the indices of the samples in (i,j) coordinates
    k_sampled = np.random.choice(max_possible_pairs, size=b, replace=False)
    i_vals, j_vals = _strict_upper_tri_to_coords(k_sampled)

    pairs_array = np.stack((i_vals, j_vals), axis=1)

    return pairs_array


def get_pairs_unique_map(pairs: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Return unique indices in the pairs, map from unique_indices
    to left and right sides of the pairs.
    Args:
        pairs (np.ndarray): P of (i,j) pairs, shape (k,2)

    Returns:
        Tuple[np.ndarray, ...]:
            - U uniques integers used in pairs, shape (m,)
            - Left inverse L, shape (k,), U[L[a]] = P[a,0]
            - Right inverse R, shape (k,), U[R[a]] = P[a,1]
    """
    # Consolidate all indices from pairs
    all_indices = np.concatenate((pairs[:, 0], pairs[:, 1]))

    # unique indices and their inverse indexing
    unique_indices, inverse_indices = np.unique(
        all_indices, return_inverse=True
    )
    k = pairs.shape[0]

    return unique_indices, inverse_indices[:k], inverse_indices[k:]


if __name__ == "__main__":
    import math

    n_elements = 10
    num_pairs_to_generate = 5

    try:
        random_pairs = get_unique_lower_pairs(n_elements, num_pairs_to_generate)
        print(
            f"Generated {len(random_pairs)} unique ordered pairs from range [0, {n_elements - 1}]:"
        )
        print(random_pairs)

        uniq, inverse_left, inverse_right = get_pairs_unique_map(random_pairs)
        print(f"Unique values involved: {uniq}")
        print(f"inverse maps: {inverse_left, inverse_right}")
        assert np.all(random_pairs[:, 0] == uniq[inverse_left])
        assert np.all(random_pairs[:, 1] == uniq[inverse_right])

        # Example: generate all possible pairs
        n_small = 4
        max_b = math.comb(n_small, 2)
        all_pairs = get_unique_lower_pairs(n_small, max_b)
        print(f"\nAll possible pairs for n={n_small} (should be {max_b}):")
        # Sort the final list for consistent output view and verification
        print(np.array(sorted([tuple(a) for a in all_pairs])))

        zero_pairs, _ = get_unique_lower_pairs(10, 0)
        print(f"\nGenerating 0 pairs: {zero_pairs}")

    except ValueError as e:
        print(f"\nError: {e}")

    W = np.random.randint(0, 2, size=(100, 100))
