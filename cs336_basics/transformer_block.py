import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention_rope import CausalMultiHeadSelfAttention
from cs336_basics.positionwise_feedforward import PositionwiseFeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int, theta: float = 10000.0,
                 q_proj_weight: torch.Tensor = None,
                 k_proj_weight: torch.Tensor = None,
                 v_proj_weight: torch.Tensor = None,
                 o_proj_weight: torch.Tensor = None,
                 w1_weight: torch.Tensor = None,
                 w3_weight: torch.Tensor = None,
                 w2_weight: torch.Tensor = None,
                 norm1_weight: torch.Tensor = None,
                 norm2_weight: torch.Tensor = None,
                 rmsnorm_eps: float = 1e-5):
        super().__init__()

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            o_proj_weight=o_proj_weight,
        )
        self.norm1 = RMSNorm(d_model, eps=rmsnorm_eps)
        if norm1_weight is not None:
            self.norm1.weight.data = norm1_weight.clone()

        self.ffn = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
        )
        if w1_weight is not None:
            self.ffn.w1.data = w1_weight.clone()
        if w3_weight is not None:
            self.ffn.w3.data = w3_weight.clone()
        if w2_weight is not None:
            self.ffn.w2.data = w2_weight.clone()
        self.norm2 = RMSNorm(d_model, eps=rmsnorm_eps)
        if norm2_weight is not None:
            self.norm2.weight.data = norm2_weight.clone()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), token_positions=token_positions)
        x = x + self.ffn(self.norm2(x))
        return x