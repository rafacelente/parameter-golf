# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch

from .forward import m2rnn_forward_triton
from .backward import m2rnn_backward_triton
from .utils import _get_num_heads


class _M2RNN(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        W: torch.Tensor,
        xf: torch.Tensor,
        h0: torch.Tensor | None,
        gradient_clipping: float | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=q, k=k, v=v, W=W, xf=xf, run_check=False)

        if cu_seqlens is None:
            B = k.size(0)
        else:
            B = cu_seqlens.size(0) - 1

        K = k.size(-1)
        V = v.size(-1)

        ht = torch.empty(B, N, K, V, device=k.device, dtype=k.dtype)

        y_shape = list(v.size())
        y_shape[-2] = N
        y = torch.empty(y_shape, device=q.device, dtype=q.dtype)

        m2rnn_forward_triton(
            q=q,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            h=None,
            ht=ht,
            y=y,
            cu_seqlens=cu_seqlens,
            Nq=Nq,
            Nk=Nk,
            Nv=Nv,
            Nw=Nw,
            Nxf=Nxf,
            N=N,
        )

        ctx.save_for_backward(q, k, v, W, xf, h0, cu_seqlens)
        ctx.gradient_clipping = gradient_clipping
        ctx.num_heads = Nq, Nk, Nv, Nw, Nxf, N

        y = y.type_as(v)

        return y, ht

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        q, k, v, W, xf, h0, cu_seqlens = ctx.saved_tensors
        Nq, Nk, Nv, Nw, Nxf, N = ctx.num_heads

        V = v.size(-1)

        if cu_seqlens is None:
            B, S, _, K = q.size()
            h = torch.empty(B, S, N, K, V, dtype=q.dtype, device=q.device)
        else:
            T, _, K = q.size()
            h = torch.empty(T, N, K, V, dtype=q.dtype, device=q.device)

        # recompute h (memory-saving: not cached during forward)
        m2rnn_forward_triton(
            q=None,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            h=h,
            ht=None,
            y=None,
            cu_seqlens=cu_seqlens,
            Nq=Nq,
            Nk=Nk,
            Nv=Nv,
            Nw=Nw,
            Nxf=Nxf,
            N=N,
        )

        def _empty_or_zeros(ref: torch.Tensor, n_head: int) -> torch.Tensor:
            if n_head == N:
                return torch.empty_like(ref, memory_format=torch.contiguous_format)
            return torch.zeros_like(ref, dtype=torch.float32, memory_format=torch.contiguous_format)

        dq = _empty_or_zeros(q, Nq)
        dk = _empty_or_zeros(k, Nk)
        dv = _empty_or_zeros(v, Nv)
        dW = torch.zeros_like(W, dtype=torch.float32, memory_format=torch.contiguous_format)
        dxf = _empty_or_zeros(xf, Nxf)
        dh0 = torch.empty_like(h0, memory_format=torch.contiguous_format) if h0 is not None and h0.requires_grad else None

        m2rnn_backward_triton(
            q=q,
            k=k,
            v=v,
            W=W,
            xf=xf,
            h0=h0,
            dy=dy,
            h=h,
            dq=dq,
            dk=dk,
            dv=dv,
            dW=dW,
            dxf=dxf,
            dh0=dh0,
            cu_seqlens=cu_seqlens,
            gradient_clipping=ctx.gradient_clipping,
        )

        dq = dq.type_as(q)
        dk = dk.type_as(k)
        dv = dv.type_as(v)
        dW = dW.type_as(W)
        dxf = dxf.type_as(xf)

        return dq, dk, dv, dW, dxf, dh0, None, None, None


def m2rnn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    forget_input: torch.Tensor,
    input_state: torch.Tensor | None = None,
    gradient_clipping: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """M2RNN recurrence with Triton-accelerated forward/backward.

    Args:
        query:  (B, S, Nq, K) or (T, Nq, K) with cu_seqlens
        key:    (B, S, Nk, K) or (T, Nk, K)
        value:  (B, S, Nv, V) or (T, Nv, V)
        weight: (Nw, V, V) state transition matrix
        forget_input: (B, S, Nxf) or (T, Nxf)
        input_state: (B, N, K, V) or None, where N = max(Nq, Nk, Nv, Nw, Nxf)
        gradient_clipping: clip hidden-state gradients in backward; None = no clip
        cu_seqlens: (B+1,) cumulative sequence lengths for variable-length mode
        max_seqlen: max sequence length (required when cu_seqlens is set)

    Returns:
        output: (B, S, N, V) or (T, N, V)
        output_state: (B, N, K, V)
    """
    if cu_seqlens is None:
        assert max_seqlen is None
        B, S, _, K = query.size()
    else:
        assert max_seqlen is not None
        assert cu_seqlens.dim() == 1
        B = cu_seqlens.size(0) - 1
        T, _, K = query.size()

    V = value.size(-1)
    Nq, Nk, Nv, Nw, Nxf, N = _get_num_heads(q=query, k=key, v=value, W=weight, xf=forget_input, run_check=True)

    if cu_seqlens is None:
        assert query.size() == (B, S, Nq, K)
        assert key.size() == (B, S, Nk, K)
        assert value.size() == (B, S, Nv, V)
        assert forget_input.size() == (B, S, Nxf)
    else:
        assert query.size() == (T, Nq, K)
        assert key.size() == (T, Nk, K)
        assert value.size() == (T, Nv, V)
        assert forget_input.size() == (T, Nxf)

    assert weight.size() == (Nw, V, V)

    if input_state is not None:
        assert input_state.size() == (B, N, K, V)

    if gradient_clipping is not None and gradient_clipping < 0:
        gradient_clipping = -gradient_clipping

    return _M2RNN.apply(
        query,
        key,
        value,
        weight,
        forget_input,
        input_state,
        gradient_clipping,
        cu_seqlens,
        max_seqlen,
    )
