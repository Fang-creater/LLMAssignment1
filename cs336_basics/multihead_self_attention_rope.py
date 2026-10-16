import torch
import torch.nn as nn
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int,
                 theta: float = 10000.0,
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
        '''
        if q_proj_weight is not None:
            self.W_Q = nn.Parameter(q_proj_weight.clone())
        else:
            self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        if k_proj_weight is not None:
            self.W_K = nn.Parameter(k_proj_weight.clone())
        else:
            self.W_K = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        if v_proj_weight is not None:
            self.W_V = nn.Parameter(v_proj_weight.clone())
        else:
            self.W_V = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        if o_proj_weight is not None:
            self.W_O = nn.Parameter(o_proj_weight.clone())
        else:
            self.W_O = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        '''
        self.W_Q = nn.Parameter(q_proj_weight.clone())
        self.W_K = nn.Parameter(k_proj_weight.clone())
        self.W_V = nn.Parameter(v_proj_weight.clone())
        self.W_O = nn.Parameter(o_proj_weight.clone())
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=self.d_k,max_seq_len=max_seq_len
        )

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = x @ self.W_Q.T
        K = x @ self.W_K.T
        V = x @ self.W_V.T
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.d_v)
        output = attn_output @ self.W_O.T

        return output