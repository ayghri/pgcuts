"""Data utilities for PGCuts."""
import torch
from torch.utils.data import Dataset


class ShuffledRangeDataset(Dataset):
    """Dataset yielding batches of shuffled indices.

    Automatically reshuffles when all batches consumed.
    """

    def __init__(self, n: int, k: int) -> None:
        """Initialize dataset.

        Args:
            n: Total number of elements.
            k: Batch size (indices per item).
        """
        self.n = n
        self.k = k
        self.perm = torch.randperm(n)
        self.num_batches = n // k
        self.taken = 0

    def __len__(self) -> int:
        """Return number of batches."""
        return self.n // self.k

    def __getitem__(
        self, idx: int
    ) -> torch.Tensor:
        """Get batch of indices.

        Args:
            idx: Batch index.

        Returns:
            Tensor of indices.
        """
        self.taken += 1
        if self.taken >= self.num_batches:
            self.perm = torch.randperm(self.n)
            self.taken = 0
        start = idx * self.k
        return self.perm[start : start + self.k]

    def shuffle(self) -> None:
        """Reshuffle the permutation."""
        self.perm = torch.randperm(self.n)
