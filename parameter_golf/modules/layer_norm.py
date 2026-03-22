import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            return F.rms_norm(x.to(dtype=torch.float32), (x.size(-1),), eps=self.eps)