# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.m2rnn.m2rnn import m2rnn
from ..kernels.utils import divide_if_divisible
from .layer_norm import RMSNorm


class M2RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        key_head_dim: int,
        value_head_dim: int,
        output_size: int,
        num_query_heads: int,
        num_key_heads: int,
        num_value_heads: int,
        num_forget_input_heads: int,
        num_weight_heads: int,
        add_bias: bool = False,
        gradient_clipping: float | None = None,
        conv_kernel_size: int = 4,
    ) -> None:
        super().__init__()

        self.gradient_clipping = gradient_clipping
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim

        self.num_query_heads = num_query_heads
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.num_forget_input_heads = num_forget_input_heads
        self.num_weight_heads = num_weight_heads

        self.num_heads = max(num_query_heads, num_key_heads, num_value_heads, num_forget_input_heads, num_weight_heads)
        self.state_size = self.num_heads * self.value_head_dim

        divide_if_divisible(self.num_heads, self.num_query_heads)
        divide_if_divisible(self.num_heads, self.num_key_heads)
        divide_if_divisible(self.num_heads, self.num_value_heads)
        divide_if_divisible(self.num_heads, self.num_forget_input_heads)
        divide_if_divisible(self.num_heads, self.num_weight_heads)

        q_dim = self.num_query_heads * self.key_head_dim
        k_dim = self.num_key_heads * self.key_head_dim
        v_dim = self.num_value_heads * self.value_head_dim
        f_dim = self.num_forget_input_heads
        g_dim = self.state_size

        # Eq. 14-17: separate linear projections for qkv, f, g
        self.qkv_proj = nn.Linear(input_size, q_dim + k_dim + v_dim, bias=add_bias)
        self.f_proj = nn.Linear(input_size, f_dim, bias=False)
        self.g_proj = nn.Linear(input_size, g_dim, bias=False)

        # Eq. 14-16: depthwise causal conv1d on q, k, v
        self.conv_kernel_size = conv_kernel_size
        self.q_conv = nn.Conv1d(q_dim, q_dim, conv_kernel_size, groups=q_dim, bias=True)
        self.k_conv = nn.Conv1d(k_dim, k_dim, conv_kernel_size, groups=k_dim, bias=True)
        self.v_conv = nn.Conv1d(v_dim, v_dim, conv_kernel_size, groups=v_dim, bias=True)

        # Eq. 19: state transition weight
        self.state_weight = nn.Parameter(torch.empty(self.num_weight_heads, self.value_head_dim, self.value_head_dim))

        # Eq. 21: learnable residual weight w_r (per value dim per head)
        self.w_r = nn.Parameter(torch.ones(self.state_size))

        # Eq. 22: RMSNorm before output projection
        self.out_norm = RMSNorm()

        # Eq. 23: output projection
        self.output_projection = nn.Linear(self.state_size, output_size, bias=False)

        self.reset_parameters()

    def _causal_conv1d(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Apply depthwise causal conv1d: (B, S, D) -> (B, S, D)."""
        x = x.transpose(1, 2)  # (B, D, S)
        x = F.pad(x, (self.conv_kernel_size - 1, 0))
        x = conv(x)
        return x.transpose(1, 2)  # (B, S, D)

    def forward(
        self,
        input: torch.Tensor,
        input_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_dim = self.num_query_heads * self.key_head_dim
        k_dim = self.num_key_heads * self.key_head_dim
        v_dim = self.num_value_heads * self.value_head_dim

        qkv = self.qkv_proj(input)
        q, k, v = qkv.split((q_dim, k_dim, v_dim), dim=-1)

        q = F.silu(self._causal_conv1d(q, self.q_conv))
        k = F.silu(self._causal_conv1d(k, self.k_conv))
        v = F.silu(self._causal_conv1d(v, self.v_conv))

        f = torch.sigmoid(self.f_proj(input))

        g = F.silu(self.g_proj(input))

        q = q.view(*q.size()[:-1], -1, self.key_head_dim)
        k = k.view(*k.size()[:-1], -1, self.key_head_dim)
        v = v.view(*v.size()[:-1], -1, self.value_head_dim)

        if input_state is not None:
            input_state = input_state.view(-1, self.num_heads, self.key_head_dim, self.value_head_dim)

        y, output_state = m2rnn(
            query=q,
            key=k,
            value=v,
            weight=self.state_weight,
            forget_input=f,
            input_state=input_state,
            gradient_clipping=self.gradient_clipping,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # residual
        y = y.flatten(-2, -1)
        v_flat = v.flatten(-2, -1)
        Gv = self.num_heads // self.num_value_heads
        if Gv > 1:
            v_flat = v_flat.repeat_interleave(Gv, dim=-1)
        y = y + self.w_r.to(dtype=y.dtype) * v_flat

        y = self.out_norm(y * g)

        output_state = output_state.flatten(-2, -1)
        y = self.output_projection(y)

        return y, output_state

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_weight, std=self.value_head_dim ** -0.5)
