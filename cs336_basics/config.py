import argparse

def parse_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS336 Assignment1 Training")

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--no_rmsnorm", action="store_true", default=False,
                        help="Disable RMSNorm (default: False, i.e. RMSNorm is ON).")
    parser.add_argument('--no_prenorm', action='store_true',
                        help='Use post-norm instead of pre-norm')
    parser.add_argument('--no_rope', action='store_true',
                        help='Disable RoPE (use NoPE instead)')

    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--total_steps", type=int, default=80000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=r"/mnt/d/LLM/Assignment1/data/checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=r"/mnt/d/LLM/Assignment1/data/logs")
    parser.add_argument("--experiment_name", type=str, default="exp_default")
    parser.add_argument("--save_dir", type=str, default=r"/mnt/d/LLM/Assignment1/data/logs/plots")

    return parser.parse_args()