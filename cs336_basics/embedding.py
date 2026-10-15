import torch
import torch.nn as nn
from torch.nn import init

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]