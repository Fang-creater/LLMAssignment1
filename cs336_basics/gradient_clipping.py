from typing import Iterable

import torch

def gradient_clipping(
    parameters:Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
):
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return
    total_norm = torch.zeros(1,device=params_with_grad[0].grad.device)
    for p in params_with_grad:
        total_norm += p.grad.data.pow(2).sum()
    total_norm = torch.sqrt(total_norm)
    if total_norm > max_l2_norm:
        clipping_rate = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.data *= clipping_rate
