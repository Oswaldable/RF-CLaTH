import torch
from torch import nn


class QuantizationLoss(nn.Module):
    """Encourage soft hash codes to approach -1/+1."""

    def forward(self, u_a: torch.Tensor, u_b: torch.Tensor) -> torch.Tensor:
        q_a = ((u_a.abs() - 1.0) ** 2).mean()
        q_b = ((u_b.abs() - 1.0) ** 2).mean()
        return 0.5 * (q_a + q_b)


class BalanceLoss(nn.Module):
    """Encourage every bit to be balanced across a batch."""

    def forward(self, u_a: torch.Tensor, u_b: torch.Tensor) -> torch.Tensor:
        b_a = (u_a.mean(dim=0) ** 2).mean()
        b_b = (u_b.mean(dim=0) ** 2).mean()
        return 0.5 * (b_a + b_b)
