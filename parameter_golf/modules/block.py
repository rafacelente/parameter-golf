from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from .att import CausalSelfAttention
from .ff import MLP
from .layer_norm import RMSNorm
from .m2rnn import M2RNN


class BlockType(str, Enum):
    ATTENTION = "attention"
    M2RNN = "m2rnn"


class AttentionBlock(nn.Module):
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
        use_p_softmax_attention: bool = False,
    ):
        super().__init__()
        self.attn_norm = nn.Identity() if attn_bitlinear else RMSNorm()
        self.mlp_norm = nn.Identity() if mlp_bitlinear else RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            use_bitlinear=attn_bitlinear,
            use_p_softmax_attention=use_p_softmax_attention,
        )
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


class M2RNNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        num_forget_input_heads: int | None = None,
        num_weight_heads: int | None = None,
        m2rnn_gradient_clipping: float | None = None,
        mlp_bitlinear: bool = False,
    ):
        super().__init__()
        head_dim = dim // num_heads

        self.rnn_norm = RMSNorm()
        self.mlp_norm = nn.Identity() if mlp_bitlinear else RMSNorm()
        self.m2rnn = M2RNN(
            input_size=dim,
            key_head_dim=head_dim,
            value_head_dim=head_dim,
            output_size=dim,
            num_query_heads=num_heads,
            num_key_heads=num_kv_heads,
            num_value_heads=num_kv_heads,
            num_forget_input_heads=num_forget_input_heads if num_forget_input_heads is not None else num_kv_heads,
            num_weight_heads=num_weight_heads if num_weight_heads is not None else num_kv_heads,
            add_bias=False,
            gradient_clipping=m2rnn_gradient_clipping,
        )
        self.mlp = MLP(dim, mlp_mult, use_bitlinear=mlp_bitlinear)
        self.rnn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        rnn_out, _ = self.m2rnn(self.rnn_norm(x))
        x = x + self.rnn_scale.to(dtype=x.dtype)[None, None, :] * rnn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


def make_block(
    block_type: BlockType,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: int,
    rope_base: float = 10000.0,
    qk_gain_init: float = 1.5,
    attn_bitlinear: bool = False,
    mlp_bitlinear: bool = False,
    use_p_softmax_attention: bool = False,
    num_forget_input_heads: int | None = None,
    num_weight_heads: int | None = None,
    m2rnn_gradient_clipping: float | None = None,
) -> nn.Module:
    """Factory that returns an AttentionBlock or M2RNNBlock based on block_type."""
    if block_type == BlockType.ATTENTION:
        return AttentionBlock(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            rope_base=rope_base,
            qk_gain_init=qk_gain_init,
            attn_bitlinear=attn_bitlinear,
            mlp_bitlinear=mlp_bitlinear,
            use_p_softmax_attention=use_p_softmax_attention,
        )
    elif block_type == BlockType.M2RNN:
        return M2RNNBlock(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlp_mult=mlp_mult,
            num_forget_input_heads=num_forget_input_heads,
            num_weight_heads=num_weight_heads,
            m2rnn_gradient_clipping=m2rnn_gradient_clipping,
            mlp_bitlinear=mlp_bitlinear,
        )
    else:
        raise ValueError(f"Unknown block type: {block_type}")
