import torch
import torch.nn as nn

from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 q_proj_weight: torch.Tensor = None,
                 k_proj_weight: torch.Tensor = None,
                 v_proj_weight: torch.Tensor = None,
                 o_proj_weight: torch.Tensor = None):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.W_Q = nn.Parameter(q_proj_weight.clone())
        self.W_K = nn.Parameter(k_proj_weight.clone())
        self.W_V = nn.Parameter(v_proj_weight.clone())
        self.W_O = nn.Parameter(o_proj_weight.clone())

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = x @ self.W_Q.T
        K = x @ self.W_K.T
        V = x @ self.W_V.T
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.d_v)
        output = attn_output @ self.W_O.T

        return output