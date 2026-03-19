import torch
from contextlib import contextmanager
from typing import Dict, Optional, Iterator, Tuple


class GradientMonitor:
    """Monitor gradients during backward passes using hooks."""

    def __init__(
        self, named_parameters: Iterator[Tuple[str, torch.nn.Parameter]]
    ):
        self._named_params = list(named_parameters)
        self._gradients: Dict[str, Dict[str, torch.Tensor]] = {}

    @contextmanager
    def __call__(self, loss_name: str):
        """Context manager to capture gradients for a specific loss."""
        self._gradients[loss_name] = {}

        handles = []
        for name, param in self._named_params:
            if param.requires_grad:

                def make_hook(param_name: str):
                    def hook(grad: torch.Tensor) -> torch.Tensor:
                        self._gradients[loss_name][param_name] = grad.clone()
                        return grad

                    return hook

                handles.append(param.register_hook(make_hook(name)))

        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    @staticmethod
    def _compute_stats(grad: torch.Tensor) -> Dict[str, float]:
        flat = grad.flatten().float()
        abs_flat = flat.abs()
        return {
            "norm": grad.norm().item(),
            "max": flat.max().item(),
            "min": flat.min().item(),
            "mean": flat.mean().item(),
            "median": flat.median().item(),
            "abs_mean": abs_flat.mean().item(),
            "abs_max": abs_flat.max().item(),
            "abs_median": abs_flat.median().item(),
        }

    def stats(self, loss_name: str) -> Dict[str, Dict[str, float]]:
        """Get per-parameter gradient statistics."""
        if loss_name not in self._gradients:
            raise ValueError(f"No gradients captured for loss '{loss_name}'")

        return {
            name: self._compute_stats(grad)
            for name, grad in self._gradients[loss_name].items()
        }

    def get_gradients(self, loss_name: str) -> Dict[str, torch.Tensor]:
        """Get raw gradient tensors."""
        return dict(self._gradients.get(loss_name, {}))

    def clear(self, loss_name: Optional[str] = None):
        """Clear captured gradients."""
        if loss_name is None:
            self._gradients.clear()
        else:
            self._gradients.pop(loss_name, None)


class GradientMixer:
    def __init__(
        self,
        named_parameters: Iterator[Tuple[str, torch.nn.Parameter]],
        loss_scale: Dict[str, float],
    ):
        self._named_params = list(named_parameters)
        self._scales_book = loss_scale
        self.hooks = []

    def attach(self):
        for _, param in self._named_params:
            if param.requires_grad:
                self.hooks.append(
                    param.register_hook(
                        self.make_grad_scaler_hook(self._scales_book)
                    )
                )

    def detach(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @contextmanager
    def __call__(self, loss_name: str):
        """Context manager to capture gradients for a specific loss."""
        # if loss_scale is None:
        self.detach()
        loss_scale = self._scales_book[loss_name]
        self._scales_book["loss_scale"] = loss_scale

        self.attach()

        try:
            yield self
        finally:
            self.detach()

    @staticmethod
    def make_grad_scaler_hook(scales_book: Dict[str, float]):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = (grad / (grad.norm() + 1e-12)).mul_(
                scales_book["loss_scale"]
            )
            return grad

        return hook
