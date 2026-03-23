import torch
import torch.nn as nn
from torch import Tensor
from .modules.block import BlockType, make_block, M2RNNBlock
from .modules.layer_norm import RMSNorm
from .modules.linear import CastedLinear
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        attn_bitlinear_layers: set[int] | None = None,
        mlp_bitlinear_layers: set[int] | None = None,
        use_p_softmax_attention: bool = False,
        m2rnn_layers: set[int] | None = None,
        m2rnn_num_forget_input_heads: int | None = None,
        m2rnn_num_weight_heads: int | None = None,
        m2rnn_gradient_clipping: float | None = None,
        tied_weights: list[set[int]] | None = None,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.use_p_softmax_attention = use_p_softmax_attention
        _attn_bl = attn_bitlinear_layers or set()
        _mlp_bl = mlp_bitlinear_layers or set()
        _m2rnn = m2rnn_layers or set()

        # Build a mapping from layer index -> canonical index for weight tying.
        # Layers sharing a tied group all point to the lowest index in that group.
        _tie_map: dict[int, int] = {}
        for group in (tied_weights or []):
            canonical = min(group)
            for idx in group:
                _tie_map[idx] = canonical

        built_blocks: dict[int, nn.Module] = {}
        block_list: list[nn.Module] = []
        for i in range(num_layers):
            canonical = _tie_map.get(i, i)
            if canonical not in built_blocks:
                built_blocks[canonical] = make_block(
                    block_type=BlockType.M2RNN if canonical in _m2rnn else BlockType.ATTENTION,
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_mult=mlp_mult,
                    rope_base=rope_base,
                    qk_gain_init=qk_gain_init,
                    attn_bitlinear=(canonical in _attn_bl),
                    mlp_bitlinear=(canonical in _mlp_bl),
                    use_p_softmax_attention=self.use_p_softmax_attention,
                    num_forget_input_heads=m2rnn_num_forget_input_heads,
                    num_weight_heads=m2rnn_num_weight_heads,
                    m2rnn_gradient_clipping=m2rnn_gradient_clipping,
                )
            block_list.append(built_blocks[canonical])
        self.blocks = nn.ModuleList(block_list)
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
        for block in self.blocks:
            if isinstance(block, M2RNNBlock):
                block.m2rnn.reset_parameters()

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")