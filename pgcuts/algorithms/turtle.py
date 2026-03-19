from typing import Iterable
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.special import entr as entropy
from ..models.layers import PerFeatLinear
from .. import metrics


def init_classifier(ins_features, num_clusters, lr, weight_norm=False):
    classifier = PerFeatLinear(ins_features, num_clusters, weight_norm=weight_norm)
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999))
    return classifier, optimizer


class Turtle(nn.Module):
    def __init__(
        self,
        num_clusters,
        ins_features: Iterable[int],
        lr_outer: float,
        lr_inner: float,
        entropy_weight: float,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.ins_features = ins_features
        self.entropy_weight = entropy_weight
        self.classifier_outer, self.optimizer_outer = init_classifier(
            ins_features, num_clusters, lr=lr_outer, weight_norm=True
        )
        self.lr_inner = lr_inner
        self.classifier_inner, self.optimizer_inner = None, None
        self.init_inner()

    def init_inner(self):
        self.classifier_inner, self.optimizer_inner = init_classifier(
            self.ins_features,
            self.num_clusters,
            lr=self.lr_inner,
        )

    def forward(self, Z_input):
        z_per_space = self.classifier_outer(Z_input)  # shape (B, K, num_feats)
        return torch.softmax(z_per_space, dim=-2)

    def loss_inner(self, logits_inner: torch.Tensor, labels: torch.Tensor):
        # [(B,K, num_spaces),], [B, K]
        return F.cross_entropy(logits_inner, labels.detach())

    def loss_outer(
        self,
        logits_inner: torch.Tensor,
        labels: torch.Tensor,
        labels_per_spaces: torch.Tensor,
    ):
        loss_pred = F.cross_entropy(logits_inner.detach(), labels)
        loss_reg = -self.entropy_weight * entropy(labels_per_spaces.mean(0))
        return loss_pred + loss_reg

    @torch.no_grad()
    def predict(self, Z_input):
        labels_per_spaces = self.classifier_outer(Z_input)
        return torch.sum(labels_per_spaces, dim=-1).argmax(dim=1)
