import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, use_swiglu: bool = True, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int((8 / 3) * d_model)
            d_ff = ((d_ff + 63) // 64) * 64
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.w1 = nn.Parameter(
                torch.empty(d_ff, d_model, device=device, dtype=dtype)
            )
            self.w3 = nn.Parameter(
                torch.empty(d_ff, d_model, device=device, dtype=dtype)
            )
            self.w2 = nn.Parameter(
                torch.empty(d_model, d_ff, device=device, dtype=dtype)
            )
        else:
            self.w1 = nn.Parameter(
                torch.empty(d_ff, d_model, device=device, dtype=dtype)
            )
            self.w2 = nn.Parameter(
                torch.empty(d_model, d_ff, device=device, dtype=dtype)
            )
            self.register_parameter('w3', None)
        self.reset_parameters()

    def reset_parameters(self):
        std_1 = (2.0 / (self.d_model + self.d_ff)) ** 0.5
        init.trunc_normal_(self.w1, mean=0.0, std=std_1, a=-3 * std_1, b=3 * std_1)
        if self.use_swiglu:
            init.trunc_normal_(self.w3, mean=0.0, std=std_1, a=-3 * std_1, b=3 * std_1)
        std_2 = (2.0 / (self.d_ff + self.d_model)) ** 0.5
        init.trunc_normal_(self.w2, mean=0.0, std=std_2, a=-3 * std_2, b=3 * std_2)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        if self.use_swiglu:
            gate = F.silu(F.linear(x, self.w1))
            value = F.linear(x, self.w3)
            gated = gate * value
            output = F.linear(gated, self.w2)
            return output
        else:
            return F.silu(F.linear(x, self.w1)) @ self.w2.T