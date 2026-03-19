from typing import Tuple

import torch
from torch import nn

from ..hyp2f1.funct import hyp2f1


class HyCutLoss(nn.Module):
    """Hypergeometric envelope loss for graph cut minimization.

    Computes the HyCut objective using the Gauss hypergeometric function
    2F1(-m, b; c; z) as an upper bound on the expected graph cut.

    Parameters:
        m: Polynomial degree for the hypergeometric function.
        b: Second parameter of 2F1 (default: 1.0).
        c: Third parameter of 2F1 (default: 2.0).
        ema_decay: Exponential moving average decay for cluster proportions.
    """

    def __init__(
        self, m: int, b: float = 1.0, c: float = 2.0, ema_decay: float = 0.9
    ) -> None:
        super().__init__()
        self.m = m
        self.b = b
        self.c = c
        self.ema_decay = ema_decay

    def forward(
        self,
        p_left: torch.Tensor,
        log_p_right: torch.Tensor,
        weights: torch.Tensor,
        alphas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the HyCut loss.

        Args:
            p_left: Softmax probabilities for left nodes, shape (E, K).
            log_p_right: Log-softmax probabilities for right nodes, shape (E, K).
            weights: Edge weights, shape (E,).
            alphas: Current EMA cluster proportions, shape (K,).

        Returns:
            Tuple of (loss, updated_alphas).
        """
        m = torch.tensor(self.m, device=p_left.device, dtype=torch.int32)
        b = torch.tensor(self.b, device=p_left.device, dtype=p_left.dtype)
        c = torch.tensor(self.c, device=p_left.device, dtype=p_left.dtype)

        # Cross-entropy weighted by edge weights: (E, K) -> (K,)
        hycut = (weights.unsqueeze(-1) * (-p_left * log_p_right)).mean(0)

        # Compute mean cluster proportions for the batch
        p_mean = p_left.detach().mean(0)

        # EMA update of cluster proportions
        alpha_input = p_mean * (1 - self.ema_decay) + self.ema_decay * alphas

        # Apply hypergeometric envelope
        hycut = (hycut * hyp2f1(-m, b, c, alpha_input)).sum()
        hycut = hycut / weights.sum()

        # Update alphas
        updated_alphas = alphas * self.ema_decay + (1 - self.ema_decay) * p_mean

        return hycut, updated_alphas
