import torch
import torch.nn as nn
from torch import Tensor
from .linear import CastedLinear
from .bitlinear import BitLinear


def _make_linear(dim_in: int, dim_out: int, use_bitlinear: bool) -> nn.Linear:
    return BitLinear(dim_in, dim_out, bias=False) if use_bitlinear else CastedLinear(dim_in, dim_out, bias=False)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, use_bitlinear: bool = False):
        super().__init__()
        self.use_bitlinear = use_bitlinear
        hidden = mlp_mult * dim
        self.fc = _make_linear(dim, hidden, use_bitlinear)
        self.proj = _make_linear(hidden, dim, use_bitlinear)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        if self.use_bitlinear:
            bl_inf = self.fc._bitlinear_inference
            x = torch.relu(self.fc(x, inference=bl_inf))
            return self.proj(x.square(), inference=bl_inf)
        x = torch.relu(self.fc(x))
        return self.proj(x.square())
