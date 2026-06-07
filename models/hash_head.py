import torch
from torch import nn


class HashHead(nn.Module):
    """Map video representation to soft hash code.

    Input:
        z: [B, D]
    Output:
        u: [B, K] in (-1, 1)
    """

    def __init__(self, d_model: int = 768, hash_bits: int = 64):
        super().__init__()
        self.hash_bits = hash_bits
        self.fc = nn.Linear(d_model, hash_bits)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(z))

    @staticmethod
    def binarize(u: torch.Tensor, binary_format: str = "pm1") -> torch.Tensor:
        b = torch.sign(u)
        b[b == 0] = 1
        if binary_format == "01":
            b = (b > 0).to(torch.int8)
        elif binary_format == "pm1":
            b = b.to(torch.int8)
        else:
            raise ValueError(f"Unsupported binary_format: {binary_format}")
        return b
