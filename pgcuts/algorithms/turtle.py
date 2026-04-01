"""Turtle algorithm for multi-space clustering."""

from typing import Iterable
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.special import entr as entropy
from ..layers import PerFeatLinear


def init_classifier(ins_features, num_clusters, lr, weight_norm=False):
    """Initialize a classifier with optimizer."""
    classifier = PerFeatLinear(
        ins_features, num_clusters, weight_norm=weight_norm
    )
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999))
    return classifier, optimizer


class Turtle(nn.Module):
    """Multi-space clustering with inner/outer loops."""

    def __init__(
        self,
        num_clusters,
        ins_features: Iterable[int],
        lr_outer: float,
        lr_inner: float,
        entropy_weight: float,
    ):
        """Initialize Turtle module.

        Args:
            num_clusters: Number of clusters.
            ins_features: Input feature dimensions.
            lr_outer: Learning rate for outer loop.
            lr_inner: Learning rate for inner loop.
            entropy_weight: Weight for entropy loss.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.ins_features = ins_features
        self.entropy_weight = entropy_weight
        self.classifier_outer, self.optimizer_outer = init_classifier(
            ins_features,
            num_clusters,
            lr=lr_outer,
            weight_norm=True,
        )
        self.lr_inner = lr_inner
        self.classifier_inner = None
        self.optimizer_inner = None
        self.init_inner()

    def init_inner(self):
        """Initialize the inner classifier."""
        self.classifier_inner, self.optimizer_inner = init_classifier(
            self.ins_features,
            self.num_clusters,
            lr=self.lr_inner,
        )

    def forward(self, z_input):
        """Forward pass through outer classifier.

        Args:
            z_input: Input features.

        Returns:
            Softmax probabilities.
        """
        z_per_space = self.classifier_outer(z_input)
        return torch.softmax(z_per_space, dim=-2)

    def loss_inner(
        self,
        logits_inner: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute inner loop loss.

        Args:
            logits_inner: Inner classifier logits.
            labels: Target labels.

        Returns:
            Cross-entropy loss.
        """
        return F.cross_entropy(logits_inner, labels.detach())

    def loss_outer(
        self,
        logits_inner: torch.Tensor,
        labels: torch.Tensor,
        labels_per_spaces: torch.Tensor,
    ):
        """Compute outer loop loss.

        Args:
            logits_inner: Inner classifier logits.
            labels: Target labels.
            labels_per_spaces: Per-space label probs.

        Returns:
            Combined prediction and entropy loss.
        """
        loss_pred = F.cross_entropy(logits_inner.detach(), labels)
        entr_val = entropy(  # pylint: disable=not-callable
            labels_per_spaces.mean(0)
        )
        loss_reg = -self.entropy_weight * entr_val
        return loss_pred + loss_reg

    @torch.no_grad()
    def predict(self, z_input):
        """Predict cluster assignments.

        Args:
            z_input: Input features.

        Returns:
            Predicted cluster indices.
        """
        labels_per_spaces = self.classifier_outer(z_input)
        return torch.sum(labels_per_spaces, dim=-1).argmax(dim=1)
