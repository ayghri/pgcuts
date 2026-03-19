from typing import Iterable
import torch
from torch import nn
from torch.nn.utils import parametrizations, remove_weight_norm
from torch import Tensor


class PerFeatLinear(nn.Module):
    def __init__(
        self,
        ins_features: Iterable[int],
        out_features: int,
        weight_norm: bool = False,
    ):
        """
        Simple class where ins_features is a list of F dimensions.
        The output is of shape (b, F, out_features)
        Each representation has its own linear projection layer
        Args:
            ins_features (Iterable[int]): List of input feature dimensions
            out_features (int): Output feature dimension
            weight_norm (bool, optional): Whether to apply weight normalization. Defaults to False.
        """
        super().__init__()
        self.ins_features = ins_features
        self.num_clusters = out_features
        self.weight_norm = weight_norm

        self.linears = []
        for i, d in enumerate(ins_features):
            layer = nn.Linear(d, out_features)
            name = f"linear_{i}"
            if weight_norm:
                layer = parametrizations.weight_norm(layer)
                name = "normed_" + name
            self.linears.append(layer)
            self.add_module(name, layer)

    def forward(self, feats_per_space: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats_per_space (Iterable[torch.Tensor]): List of F tensors,
            the i-th tensor has shape (b, ins_features[i])
        Returns:
            torch.Tensor: Output tensor of shape (b, F, out_features)
        """
        y_per_space = []
        for feats, linear in zip(feats_per_space, self.linears):
            y_per_space.append(linear(feats))
        y_per_space = torch.stack(y_per_space, dim=1)
        return y_per_space

    def remove_weight_norm(self):
        if self.weight_norm:
            for layer in self.linears:
                remove_weight_norm(layer)

    def reset_parameters(self) -> None:
        for layer in self.linears:
            layer.reset_parameters()


class DecoupledLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_spaces,
        bias=True,
        device=None,
        dtype=None,
    ):
        """
        Linear layer where each space has its own linear projection layer, all spaces
        should have the same dimension.
        Args:
            in_features (int): Input feature dimension
            out_features (int): Output feature dimension
            num_spaces (int): Number of spaces
            bias (bool, optional): Whether to include bias. Defaults to True.
            device ([type], optional): Device to store the weights. Defaults to None.
            dtype ([type], optional): Data type of the weights. Defaults to None.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

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
                torch.empty((1, num_spaces, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Input tensor of shape (b, num_spaces, in_features)
        Returns:
            Tensor: Output tensor of shape (b,  num_spaces, out_features)
        """
        output = torch.einsum("bcf,cfd->bcd", input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            + f"num_numspaces={self.num_spaces}, bias={self.bias is not None}"
        )
