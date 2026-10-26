import sys
import os
import json
import argparse
import subprocess


def run_single_batch(batch_size, lr, args):
    exp_name = f"batch_sweep_bs{batch_size}_lr{lr}"
    ckpt_dir = os.path.join(args.sweep_dir, exp_name, "checkpoints")
    log_dir = os.path.join(args.sweep_dir, exp_name, "logs")

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
        "--batch_size", str(batch_size),
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
        "--experiment_name", exp_name,
        "--device", args.device,
        "--seed", "42",
    ]

    print(f"\n{'=' * 60}")
    print(f"Batch Sweep: batch_size={batch_size}, lr={lr}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"  ❌ batch={batch_size}, lr={lr} FAILED")
        return {"batch_size": batch_size, "lr": lr, "status": "failed"}

    print(f"  ✅ batch={batch_size}, lr={lr} completed")
    return {"batch_size": batch_size, "lr": lr, "status": "success"}


def main():
    parser = argparse.ArgumentParser(description="7.2.4 Batch Size Experiment")
    parser.add_argument("--batch_sizes", type=str, default="1,4,16,32,64",
                        help="Comma-separated batch sizes")
    parser.add_argument("--train_data", type=str,
                        default=r"/mnt/d/LLM/Assignment1/data/ts_train_tokens_uint16.npy")
    parser.add_argument("--val_data", type=str,
                        default=r"/mnt/d/LLM/Assignment1/data/ts_valid_tokens_uint16.npy")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sweep_dir", type=str,
                        default=r"/mnt/d/LLM/Assignment1/data/batch_sweep")

    args = parser.parse_args()
    os.makedirs(args.sweep_dir, exist_ok=True)

    batch_list = [int(x) for x in args.batch_sizes.split(",")]

    lr_map = {
        1: 3e-4,
        4: 5e-4,
        16: 8e-4,
        32: 8e-4,
        64: 1e-3,
        128: 1.5e-3,
    }

    print(f"Batch Size Sweep Plan:")
    for bs in batch_list:
        lr = lr_map.get(bs, 8e-4)
        print(f"  batch={bs:>3d} → lr={lr}")

    results = []
    for bs in batch_list:
        lr = lr_map.get(bs, 8e-4)
        result = run_single_batch(bs, lr, args)
        results.append(result)

        with open(os.path.join(args.sweep_dir, "sweep_status.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("BATCH SWEEP COMPLETE")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  batch={r['batch_size']}, lr={r['lr']}: {r['status']}")


if __name__ == "__main__":
    main()

"""
  uv run python train_batch_sweep.py --batch_sizes 1,4,16,32,64 --steps 5000
"""