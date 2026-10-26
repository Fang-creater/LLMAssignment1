import sys
import os
import json
import argparse
import subprocess

def run_single_lr(lr, args):
    lr_str = str(lr).replace(".", "p").replace("-", "minus")
    experiment_name = f"lr_sweep_{lr_str}"
    ckpt_dir = os.path.join(args.sweep_dir, experiment_name, "checkpoints")
    log_dir = os.path.join(args.sweep_dir, experiment_name, "logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable, "train_loop.py",
        "--train_data", args.train_data,
        "--val_data", args.val_data,
        "--vocab_size", "10000",
        "--d_model", "512",
        "--num_layers", "4",
        "--num_heads", "16",
        "--d_ff", "1344",
        "--context_length", "256",
        "--batch_size", str(args.batch_size),
        "--total_steps", str(args.steps),
        "--warmup_steps", str(args.warmup_steps),
        "--max_lr", str(lr),
        "--min_lr", str(lr * 0.1),
        "--weight_decay", "0.1",
        "--max_grad_norm", "1.0",
        "--eval_interval", str(args.eval_interval),
        "--save_interval", str(args.steps + 1),
        "--log_interval", "100",
        "--checkpoint_dir", ckpt_dir,
        "--log_dir", log_dir,
        "--experiment_name", experiment_name,
        "--device", args.device,
        "--seed", "42",
    ]

    print(f"\n{'=' * 60}")
    print(f"LR Sweep: lr={lr}")
    print(f"Cmd: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"  ❌ LR={lr} FAILED (return code {result.returncode})")
        return {"lr": lr, "status": "failed"}

    print(f"  ✅ LR={lr} completed")
    return {"lr": lr, "status": "success"}


def main():
    parser = argparse.ArgumentParser(description="7.2.3 LR Sweep")
    parser.add_argument("--lrs", type=str, default="1e-4,3e-4,5e-4,8e-4,1e-3,3e-3",
                        help="Comma-separated learning rates")
    parser.add_argument("--train_data", type=str,
                        default=r"/mnt/d/LLM/Assignment1/data/ts_train_tokens_uint16.npy")
    parser.add_argument("--val_data", type=str,
                        default=r"/mnt/d/LLM/Assignment1/data/ts_valid_tokens_uint16.npy")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sweep_dir", type=str,
                        default=r"/mnt/d/LLM/Assignment1/data/lr_sweep")

    args = parser.parse_args()

    os.makedirs(args.sweep_dir, exist_ok=True)

    lr_list = [float(x) for x in args.lrs.split(",")]

    print(f"LR Sweep Plan: {lr_list}")
    print(f"Steps: {args.steps}, Batch: {args.batch_size}")
    print(f"Total tokens per run: {args.steps * args.batch_size * 256:,}")

    results = []
    for lr in lr_list:
        result = run_single_lr(lr, args)
        results.append(result)

        with open(os.path.join(args.sweep_dir, "sweep_status.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  LR={r['lr']}: {r['status']}")


if __name__ == "__main__":
    main()

"""
uv run python train_lr_sweep.py \
  --lrs 1e-4,3e-4,5e-4,8e-4,1e-3,3e-3 \
  --steps 5000 \
  --batch_size 32 \
  --eval_interval 200
"""