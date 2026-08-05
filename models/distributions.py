from __future__ import annotations

from numbers import Number

import torch
from tensordict.nn.utils import mappings
from torch import nn


class NormalParamExtractor(nn.Module):
    """A non-parametric nn.Module that splits its input into loc and scale parameters.

    The scale parameters are mapped onto positive values using the specified ``scale_mapping``.

    Args:
        scale_mapping (str, optional): positive mapping function to be used with the std.
            default = "biased_softplus_1.0" (i.e. softplus map with bias such that fn(0.0) = 1.0)
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        scale_lb (Number, optional): The minimum value that the variance can take. Default is 1e-4.

    Examples:
        >>> import torch
        >>> from tensordict.nn.distributions import NormalParamExtractor
        >>> from torch import nn
        >>> module = nn.Linear(3, 4)
        >>> normal_params = NormalParamExtractor()
        >>> tensor = torch.randn(3)
        >>> loc, scale = normal_params(module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([2]) torch.Size([2])
        >>> assert (scale > 0).all()
        >>> # with modules that return more than one tensor
        >>> module = nn.LSTM(3, 4)
        >>> tensor = torch.randn(4, 2, 3)
        >>> loc, scale, others = normal_params(*module(tensor))
        >>> print(loc.shape, scale.shape)
        torch.Size([4, 2, 2]) torch.Size([4, 2, 2])
        >>> assert (scale > 0).all()

    """

    def __init__(
            self,
            scale_mapping: str = "biased_softplus_1.0",
            scale_lb: Number = 1e-4,
    ) -> None:
        super().__init__()
        self.scale_mapping = scale_mapping
        self.scale_lb = scale_lb

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        loc, scale = tensor.chunk(2, -1)
        if self.scale_mapping != 'none':
            scale = mappings(self.scale_mapping)(scale)
        scale = scale.clamp_min(self.scale_lb)
        return (loc, scale, *others)
