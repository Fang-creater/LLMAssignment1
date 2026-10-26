import torch
from tests.adapters import (
    get_adamw_cls,
    run_get_lr_cosine_schedule,
    run_cross_entropy,
)
from cs336_basics.transformer_lm import TransformerLM

def create_model(args, device: str) -> torch.nn.Module:
    d_ff = args.d_ff if args.d_ff is not None else int(8 / 3 * args.d_model)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=getattr(args, 'rope_theta', 10000.0),
    ).to(device)
    return model


def create_optimizer(model, args):
    AdamW = get_adamw_cls()
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    return optimizer


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return run_cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )

def get_lr(step: int, args) -> float:
    return run_get_lr_cosine_schedule(
        it=step,
        max_learning_rate=args.max_lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_steps,
        cosine_cycle_iters=args.total_steps,
    )