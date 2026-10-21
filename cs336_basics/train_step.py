import time
import numpy as np
from tests.adapters import run_gradient_clipping, run_get_batch
from model_layer import get_lr,compute_loss

def train_step(
    step: int,
    model,
    optimizer,
    train_data: np.ndarray,
    args,
) -> dict:
    t0 = time.time()
    lr = get_lr(step, args)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    total_loss = 0.0
    optimizer.zero_grad()

    for micro_step in range(accum_steps):
        inputs, targets = run_get_batch(
            train_data, args.batch_size, args.context_length, args.device
        )
        logits = model(inputs)
        loss = compute_loss(logits, targets)
        loss = loss / accum_steps
        loss.backward()
        total_loss += loss.item()

    run_gradient_clipping(model.parameters(), args.max_grad_norm)
    optimizer.step()
    elapsed = time.time() - t0
    tokens_per_sec = args.batch_size * args.context_length * accum_steps / elapsed

    return {
        "step": step,
        "loss": total_loss,
        "lr": lr,
        "tokens_per_sec": tokens_per_sec,
        "elapsed": elapsed,
    }