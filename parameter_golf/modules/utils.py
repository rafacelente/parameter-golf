import torch
import torch.nn as nn
from typing import List, Optional

from parameter_golf.quant.common import CONTROL_TENSOR_NAME_PATTERNS


def restore_low_dim_params_to_fp32(module: nn.Module, control_tensor_name_patterns: Optional[List[str]] = None) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    if control_tensor_name_patterns is None:
        control_tensor_name_patterns = CONTROL_TENSOR_NAME_PATTERNS
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in control_tensor_name_patterns)) and param.dtype != torch.float32:
                param.data = param.data.float()