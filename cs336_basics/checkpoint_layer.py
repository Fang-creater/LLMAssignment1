import os
from tests.adapters import run_save_checkpoint, run_load_checkpoint

def save_ckpt(model, optimizer, step, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_step_{step}.pt")
    run_save_checkpoint(model, optimizer, step, path)
    print(f"[Checkpoint] Saved step {step} → {path}")


def load_ckpt(model, optimizer, checkpoint_path: str) -> int:
    step = run_load_checkpoint(checkpoint_path, model, optimizer)
    print(f"[Checkpoint] Resumed from {checkpoint_path}, step={step}")
    return step