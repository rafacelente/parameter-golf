import torch
import torch.nn as nn
from torch import Tensor
from .rotary import Rotary, apply_rotary_emb
from .linear import CastedLinear
from .bitlinear import BitLinear
import torch.nn.functional as F
from ..kernels.p_softmax_attention import triton_p_softmax_attention


def _make_linear(dim_in: int, dim_out: int, use_bitlinear: bool) -> nn.Linear:
    return BitLinear(dim_in, dim_out, bias=False) if use_bitlinear else CastedLinear(dim_in, dim_out, bias=False)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        use_bitlinear: bool = False,
        use_p_softmax_attention: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.use_bitlinear = use_bitlinear
        self.use_p_softmax_attention = use_p_softmax_attention
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = _make_linear(dim, dim, use_bitlinear)
        self.c_k = _make_linear(dim, kv_dim, use_bitlinear)
        self.c_v = _make_linear(dim, kv_dim, use_bitlinear)
        self.proj = _make_linear(dim, dim, use_bitlinear)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        if self.use_bitlinear:
            bl_inf = self.c_q._bitlinear_inference
            q = self.c_q(x, inference=bl_inf).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.c_k(x, inference=bl_inf).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.c_v(x, inference=bl_inf).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.use_p_softmax_attention:
            y = triton_p_softmax_attention(q, k, v, p=2.0)
        else:
            y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        if self.use_bitlinear:
            return self.proj(y, inference=bl_inf)
        return self.proj(y)