import torch
import torch.nn as nn
from torch import Tensor
from .att import CausalSelfAttention
from .ff import MLP
from .layer_norm import RMSNorm


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        attn_bitlinear: bool = False,
        mlp_bitlinear: bool = False,
    ):
        super().__init__()
        self.attn_norm = nn.Identity() if attn_bitlinear else RMSNorm()
        self.mlp_norm = nn.Identity() if mlp_bitlinear else RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_bitlinear=attn_bitlinear)
        self.mlp = MLP(dim, mlp_mult, use_bitlinear=mlp_bitlinear)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x
