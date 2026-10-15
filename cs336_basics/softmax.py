import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_x = x.max(dim=dim, keepdim=True).values
    x_shifted = x - max_x
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)