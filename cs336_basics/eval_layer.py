import torch
import numpy as np
from tests.adapters import run_get_batch, run_cross_entropy

@torch.no_grad()
def evaluate(
    model,
    val_data: np.ndarray,
    args,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for _ in range(args.eval_batches):
        inputs, targets = run_get_batch(
            val_data, args.batch_size, args.context_length, args.device
        )
        logits = model(inputs)
        loss = run_cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()

    model.train()
    return total_loss / total_tokens