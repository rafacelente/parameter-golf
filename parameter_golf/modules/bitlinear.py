import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_norm import RMSNorm
from ..quant.bitnet import weight_quant, activation_quant, activation_post_quant, quantize_weights_to_int8
from typing import Optional


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias)
        self.rms_norm = RMSNorm(in_features)
        self.weight_scale: torch.Tensor | None = None
        self._bitlinear_inference: bool = False
        self._skip_fake_quant: bool = False
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, inference: Optional[bool] = False) -> torch.Tensor:
        w = self.weight
        if not inference:
            x_norm = self.rms_norm(x)
            if self._skip_fake_quant:
                return F.linear(x_norm, w, self.bias)
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (weight_quant(w) - w).detach()
            return F.linear(x_quant, w_quant, self.bias)
        else:
            x_norm = self.rms_norm(x)
            x_quant, x_scale = activation_post_quant(x_norm)
            w_scale = self.weight_scale
            return F.linear(x_quant, w.float(), self.bias) / (x_scale * w_scale)


def set_bitlinear_eval_quantized(model: nn.Module) -> None:
    """After loading ternary-dequantized weights, skip the STE fake-quant during eval.

    The dequantized weights are already the ternary values the model saw during
    training.  Re-running weight_quant() on them recomputes the scale from the
    ternary distribution, which differs from the original training scale and
    silently shrinks all weights.  This flag makes the forward pass use the
    weights as-is (with RMSNorm on activations still applied).
    """
    for module in model.modules():
        if isinstance(module, BitLinear):
            module._skip_fake_quant = True


def clear_bitlinear_eval_quantized(model: nn.Module) -> None:
    """Re-enable STE fake quantization (for resuming training)."""
    for module in model.modules():
        if isinstance(module, BitLinear):
            module._skip_fake_quant = False