import math

def log_train(metrics: dict):
    print(
        f"Step {metrics['step']:>7d} | "
        f"loss {metrics['loss']:.4f} | "
        f"lr {metrics['lr']:.2e} | "
        f"{metrics['tokens_per_sec']:.0f} tok/s | "
        f"{metrics['elapsed']*1000:.1f} ms"
    )

def log_eval(step: int, val_loss: float):
    print(f"Step {step} | val_loss={val_loss:.4f} | "
          f"val_ppl={math.exp(val_loss):.2f}")