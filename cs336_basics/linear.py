import torch
import torch.nn as nn
from torch.nn import init

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.empty(out_features, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        sigma = (2.0 / (self.in_features + self.out_features)) ** 0.5
        init.trunc_normal_(self.weight,mean=0.0,std=sigma,a=-3*sigma,b=3*sigma)
        init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias