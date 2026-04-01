"""Gradient monitoring and mixing utilities."""
from contextlib import contextmanager
from typing import Dict, Optional, Iterator, Tuple
import torch


class GradientMonitor:
    """Monitor gradients during backward passes."""

    def __init__(
        self,
        named_parameters: Iterator[
            Tuple[str, torch.nn.Parameter]
        ],
    ):
        """Initialize gradient monitor.

        Args:
            named_parameters: Model named parameters.
        """
        self._named_params = list(named_parameters)
        self._gradients: Dict[
            str, Dict[str, torch.Tensor]
        ] = {}

    @contextmanager
    def __call__(self, loss_name: str):
        """Capture gradients for a specific loss.

        Args:
            loss_name: Name of the loss.
        """
        self._gradients[loss_name] = {}

        handles = []
        for name, param in self._named_params:
            if param.requires_grad:

                def make_hook(param_name: str):
                    """Create gradient hook."""
                    def hook(
                        grad: torch.Tensor,
                    ) -> torch.Tensor:
                        """Capture gradient."""
                        self._gradients[
                            loss_name
                        ][param_name] = (
                            grad.clone()
                        )
                        return grad

                    return hook

                handles.append(
                    param.register_hook(
                        make_hook(name)
                    )
                )

        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()

    @staticmethod
    def _compute_stats(
        grad: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute gradient statistics."""
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

    def stats(
        self, loss_name: str
    ) -> Dict[str, Dict[str, float]]:
        """Get per-parameter gradient statistics.

        Args:
            loss_name: Name of the loss.

        Returns:
            Dict of param name to stat dict.
        """
        if loss_name not in self._gradients:
            raise ValueError(
                "No gradients captured for "
                f"loss '{loss_name}'"
            )

        return {
            name: self._compute_stats(grad)
            for name, grad in self._gradients[
                loss_name
            ].items()
        }

    def get_gradients(
        self, loss_name: str
    ) -> Dict[str, torch.Tensor]:
        """Get raw gradient tensors.

        Args:
            loss_name: Name of the loss.

        Returns:
            Dict of param name to gradient tensor.
        """
        return dict(
            self._gradients.get(loss_name, {})
        )

    def clear(
        self, loss_name: Optional[str] = None
    ):
        """Clear captured gradients.

        Args:
            loss_name: If given, clear only this loss.
        """
        if loss_name is None:
            self._gradients.clear()
        else:
            self._gradients.pop(loss_name, None)


class GradientMixer:
    """Mix gradients from multiple losses."""

    def __init__(
        self,
        named_parameters: Iterator[
            Tuple[str, torch.nn.Parameter]
        ],
        loss_scale: Dict[str, float],
    ):
        """Initialize gradient mixer.

        Args:
            named_parameters: Model named parameters.
            loss_scale: Dict of loss name to scale.
        """
        self._named_params = list(named_parameters)
        self._scales_book = loss_scale
        self.hooks = []

    def attach(self):
        """Attach gradient scaling hooks."""
        for _, param in self._named_params:
            if param.requires_grad:
                self.hooks.append(
                    param.register_hook(
                        self.make_grad_scaler_hook(
                            self._scales_book
                        )
                    )
                )

    def detach(self):
        """Remove all gradient hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @contextmanager
    def __call__(self, loss_name: str):
        """Context manager to scale gradients.

        Args:
            loss_name: Name of the loss.
        """
        self.detach()
        current_scale = self._scales_book[loss_name]
        self._scales_book["loss_scale"] = (
            current_scale
        )

        self.attach()

        try:
            yield self
        finally:
            self.detach()

    @staticmethod
    def make_grad_scaler_hook(
        scales_book: Dict[str, float],
    ):
        """Create gradient scaling hook.

        Args:
            scales_book: Dict with 'loss_scale' key.

        Returns:
            Hook function.
        """

        def hook(
            grad: torch.Tensor,
        ) -> torch.Tensor:
            """Scale gradient by norm."""
            grad = (
                grad / (grad.norm() + 1e-12)
            ).mul_(scales_book["loss_scale"])
            return grad

        return hook
