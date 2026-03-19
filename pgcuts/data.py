import torch
from torch.utils.data import Dataset


class ShuffledRangeDataset(Dataset):
    """Dataset that yields batches of k indices from a shuffled permutation of [0, n).

    Automatically reshuffles when all batches have been consumed.

    Args:
        n: Total number of elements.
        k: Batch size (number of indices per item).
    """

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k
        self.perm = torch.randperm(n)
        self.num_batches = n // k
        self.taken = 0

    def __len__(self) -> int:
        return self.n // self.k

    def __getitem__(self, idx: int) -> torch.Tensor:
        self.taken += 1
        if self.taken >= self.num_batches:
            self.perm = torch.randperm(self.n)
            self.taken = 0
        start = idx * self.k
        return self.perm[start : start + self.k]

    def shuffle(self) -> None:
        """Reshuffle the permutation (call at epoch end)."""
        self.perm = torch.randperm(self.n)
