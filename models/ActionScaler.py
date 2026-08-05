import torch
from torch import nn


class ActionScaler(nn.Module):
    def __init__(
        self,
        max_vel: float,
        max_ang_vel: float,
    ) -> None:
        super().__init__()
        self.max_vel = max_vel
        self.max_ang_vel = max_ang_vel

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        vel = tensor[..., None, 0] * self.max_vel
        ang_vel = tensor[..., None, 1] * self.max_ang_vel
        out = torch.cat((vel, ang_vel, tensor[..., 2:]), dim=-1)
        return out
