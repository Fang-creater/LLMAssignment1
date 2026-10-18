from collections.abc import Callable, Iterable
from typing import Optional,Tuple
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params:Iterable[torch.nn.Parameter],
        lr:float = 1e-3,
        betas:Tuple[float,float] = (0.9,0.999),
        weight_decay:float = 0.01,
        eps:float = 1e-8
    ):
        defaults = {
            "lr":lr,
            "betas":betas,
            "weight_decay":weight_decay,
            "eps":eps
        }
        super().__init__(params,defaults)

    def step(self,closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1,beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                t = state.get("t", 1)
                m = state.get("m",torch.zeros_like(p.data))
                v = state.get("v",torch.zeros_like(p.data))
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
