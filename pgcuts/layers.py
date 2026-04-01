"""Neural network layers for PGCuts."""
from typing import Iterable
import torch
from torch import nn
from torch.nn.utils import (
    parametrizations,
    remove_weight_norm as _remove_wn,
)
from torch import Tensor


class PerFeatLinear(nn.Module):
    """Per-feature-space linear projection.

    Each representation space has its own linear layer.
    Output shape: (b, F, out_features).
    """

    def __init__(
        self,
        ins_features: Iterable[int],
        out_features: int,
        weight_norm: bool = False,
    ):
        """Initialize per-feature linear layers.

        Args:
            ins_features: List of input dimensions.
            out_features: Output feature dimension.
            weight_norm: Whether to apply weight norm.
        """
        super().__init__()
        self.ins_features = ins_features
        self.num_clusters = out_features
        self._weight_norm = weight_norm

        self.linears = []
        for i, d in enumerate(ins_features):
            layer = nn.Linear(d, out_features)
            name = f"linear_{i}"
            if weight_norm:
                layer = (
                    parametrizations.weight_norm(layer)
                )
                name = "normed_" + name
            self.linears.append(layer)
            self.add_module(name, layer)

    def forward(
        self,
        feats_per_space: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through per-space linears.

        Args:
            feats_per_space: List of F tensors,
                each of shape (b, ins_features[i]).

        Returns:
            Output of shape (b, F, out_features).
        """
        y_per_space = []
        for feats, linear in zip(
            feats_per_space, self.linears
        ):
            y_per_space.append(linear(feats))
        y_per_space = torch.stack(
            y_per_space, dim=1
        )
        return y_per_space

    def remove_weight_norm(self):
        """Remove weight normalization."""
        if self._weight_norm:
            for layer in self.linears:
                _remove_wn(layer)

    def reset_parameters(self) -> None:
        """Reset all layer parameters."""
        for layer in self.linears:
            layer.reset_parameters()


class DecoupledLinear(nn.Module):
    """Decoupled linear layer for multiple spaces.

    Each space has its own linear projection;
    all spaces share the same dimension.
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_spaces,
        bias=True,
        device=None,
        dtype=None,
    ):
        """Initialize decoupled linear layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            num_spaces: Number of spaces.
            bias: Whether to include bias.
            device: Device for weights.
            dtype: Data type for weights.
        """
        super().__init__()
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.in_features = in_features
        self.out_features = out_features
        self.num_spaces = num_spaces
        self.weight = nn.Parameter(
            torch.empty(
                (num_spaces, in_features, out_features),
                **factory_kwargs,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    (1, num_spaces, out_features),
                    **factory_kwargs,
                )
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, features: Tensor) -> Tensor:
        """Forward pass through decoupled linear.

        Args:
            features: Input tensor of shape
                (b, num_spaces, in_features).

        Returns:
            Output of shape (b, num_spaces, out_features).
        """
        output = torch.einsum(
            "bcf,cfd->bcd", features, self.weight
        )
        if self.bias is not None:
            output = output + self.bias
        return output

    def reset_parameters(self) -> None:
        """Reset parameters with kaiming uniform."""
        nn.init.kaiming_uniform_(
            self.weight, a=5 ** 0.5
        )
        if self.bias is not None:
            calc_fan = nn.init._calculate_fan_in_and_fan_out  # pylint: disable=protected-access
            fan_in, _ = calc_fan(self.weight)
            bound = (
                1 / fan_in ** 0.5 if fan_in > 0 else 0
            )
            nn.init.uniform_(
                self.bias, -bound, bound
            )

    def extra_repr(self) -> str:
        """Return string representation."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_numspaces={self.num_spaces}, "
            f"bias={self.bias is not None}"
        )
