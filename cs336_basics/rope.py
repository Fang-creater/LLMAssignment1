import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int,max_seq_len: int,device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)

        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        output = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)

        return output