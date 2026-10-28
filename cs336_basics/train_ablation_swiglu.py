import os
import sys
import subprocess
import argparse
import json


def run_single(config, args):
    exp_name = config["name"]
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"  lr={config['lr']}, swiglu={config['swiglu']}, d_ff={config['d_ff']}")
    print(f"{'='*60}")

    exp_dir = os.path.join(args.sweep_dir, exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
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
        "--d_ff", str(config["d_ff"]),
        "--context_length", "256",
        "--batch_size", str(args.batch_size),
        "--total_steps", str(args.steps),
        "--warmup_steps", str(args.warmup_steps),
        "--max_lr", str(config["lr"]),
        "--min_lr", str(config["lr"] * 0.1),
        "--weight_decay", "0.1",
        "--max_grad_norm", "1.0",
        "--eval_interval", str(args.eval_interval),
        "--save_interval", str(args.save_interval),
        "--log_interval", "50",
        "--checkpoint_dir", ckpt_dir,
        "--log_dir", log_dir,
        "--experiment_name", exp_name,
        "--device", args.device,
        "--seed", "42",
    ]

    if not config["swiglu"]:
        cmd.append("--no_swiglu")

    print(f"Command: {' '.join(cmd)}")
    print("-" * 40)

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {exp_name} finished successfully (return code {result.returncode})")
        return {"name": exp_name, "status": "success", "returncode": result.returncode}
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {exp_name} failed with return code {e.returncode}")
        return {"name": exp_name, "status": f"failed (code {e.returncode})", "returncode": e.returncode}
    except KeyboardInterrupt:
        print(f"\n⚠ {exp_name} interrupted by user")
        return {"name": exp_name, "status": "interrupted", "returncode": -1}


def main():
    parser = argparse.ArgumentParser(description="7.5 Ablation 3: SwiGLU vs FFNSiLU")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--train_data", type=str,
                        default=r"/mnt/d/LLM/Assignment1/data/ts_train_tokens_uint16.npy")
    parser.add_argument("--val_data", type=str,
                        default=r"/mnt/d/LLM/Assignment1/data/ts_valid_tokens_uint16.npy")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sweep_dir", type=str,
                        default=r"/mnt/d/LLM/Assignment1/data/ablation_swiglu")

    args = parser.parse_args()
    os.makedirs(args.sweep_dir, exist_ok=True)

    configs = [
        {"name": "baseline_swiglu_lr8e4", "lr": 8e-4, "swiglu": True,  "d_ff": 1344},
        {"name": "ffnsilu_lr8e4",         "lr": 8e-4, "swiglu": False, "d_ff": 2048},
    ]

    if args.experiment:
        configs = [c for c in configs if c["name"] == args.experiment]
        if not configs:
            print(f"Experiment '{args.experiment}' not found!")
            print("Available experiments:")
            for c in [
                {"name": "baseline_swiglu_lr8e4", "lr": 8e-4, "swiglu": True,  "d_ff": 1344},
                {"name": "ffnsilu_lr8e4",         "lr": 8e-4, "swiglu": False, "d_ff": 2048},
            ]:
                print(f"  {c['name']}")
            return

    print("Ablation Plan:")
    for c in configs:
        ffn_type = "SwiGLU (gated)" if c["swiglu"] else "FFNSiLU (no gate)"
        print(f"  {c['name']:30s} | {ffn_type:20s} | d_ff={c['d_ff']} | lr={c['lr']}")

    results = []
    for config in configs:
        result = run_single(config, args)
        results.append(result)
        with open(os.path.join(args.sweep_dir, "status.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("ABLATION COMPLETE")
    for r in results:
        print(f"  {r['name']}: {r['status']}")


if __name__ == "__main__":
    main()