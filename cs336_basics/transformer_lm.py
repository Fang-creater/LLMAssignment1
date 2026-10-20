import torch
import torch.nn as nn

from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 num_layers: int,
                 rope_theta: float = 10000.0,
                 token_embedding_weight: torch.Tensor = None,
                 layers_weights: list = None,
                 final_norm_weight: torch.Tensor = None,
                 lm_head_weight: torch.Tensor = None,
                 rmsnorm_eps: float = 1e-5):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.token_embedding = Embedding(vocab_size,d_model)
        if token_embedding_weight is not None:
            self.token_embedding.weight.data = token_embedding_weight.clone()

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            if layers_weights is not None and layer_idx < len(layers_weights):
                lw = layers_weights[layer_idx]
                block = TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    q_proj_weight=lw.get('q_proj_weight'),
                    k_proj_weight=lw.get('k_proj_weight'),
                    v_proj_weight=lw.get('v_proj_weight'),
                    o_proj_weight=lw.get('o_proj_weight'),
                    w1_weight=lw.get('w1_weight'),
                    w3_weight=lw.get('w3_weight'),
                    w2_weight=lw.get('w2_weight'),
                    norm1_weight=lw.get('norm1_weight'),
                    norm2_weight=lw.get('norm2_weight'),
                    rmsnorm_eps=rmsnorm_eps,
                )
            else:
                block = TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    rmsnorm_eps=rmsnorm_eps,
                )
            self.layers.append(block)

        self.final_norm = RMSNorm(d_model, eps=rmsnorm_eps)
        if final_norm_weight is not None:
            self.final_norm.weight.data = final_norm_weight.clone()

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if lm_head_weight is not None:
            self.lm_head.weight.data = lm_head_weight.clone()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits