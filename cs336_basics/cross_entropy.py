import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = logits.max(dim=-1,keepdim=True).values
    logits_stable = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable),dim=-1))
    taget_logits = torch.gather(logits,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    loss = -taget_logits + log_sum_exp + max_logits.squeeze(-1)
    return loss.mean()

def perplexity(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    ce_loss = cross_entropy(logits, targets)
    return torch.exp(ce_loss)