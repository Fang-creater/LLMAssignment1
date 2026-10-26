import random
import numpy as np
import torch
import math

from config import parse_config
from data_layer import create_dataloader
from model_layer import create_model, create_optimizer
from train_step import train_step
from eval_layer import evaluate
from checkpoint_layer import save_ckpt, load_ckpt
from log_layer import log_train, log_eval
from experiment_logger import ExperimentLogger

def main():
    args = parse_config()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = create_dataloader(args.train_data)
    val_data = create_dataloader(args.val_data) if args.val_data else None

    model = create_model(args, args.device)
    optimizer = create_optimizer(model, args)

    start_step = 0
    if args.resume_from is not None:
        start_step = load_ckpt(model, optimizer, args.resume_from)

    log_dir = getattr(args, 'log_dir', r"/mnt/d/LLM/Assignment1/data/logs")
    experiment_name = getattr(args, 'experiment_name', 'exp_default')
    logger = ExperimentLogger(log_dir, experiment_name)

    model.train()
    for step in range(start_step, args.total_steps):
        metrics = train_step(step, model, optimizer, train_data, args)

        if step % args.log_interval == 0:
            log_train(metrics)
            logger.log_train_step(
                step=step,
                loss=metrics['loss'],
                lr=metrics['lr'],
                tokens_per_sec=metrics['tokens_per_sec'],
            )

        if val_data is not None and step % args.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_data, args)
            log_eval(step, val_loss)
            logger.log_eval_step(
                step=step,
                val_loss=val_loss,
                val_ppl=math.exp(val_loss),
            )

        if step % args.save_interval == 0 and step > 0:
            save_ckpt(model, optimizer, step, args.checkpoint_dir)

    save_ckpt(model, optimizer, args.total_steps, args.checkpoint_dir)
    print("[Train] Done!")


if __name__ == "__main__":
    main()

'''
CPU：
python train_loop.py \
  --train_data /mnt/d/LLM/Assignment1/data/owt_train_tokens_uint16.npy \
  --val_data /mnt/d/LLM/Assignment1/data/ow_val_tokens_uint16.npy \
  --checkpoint_dir /mnt/d/LLM/Assignment1/data/ \
  --total_steps 50 \
  --log_interval 5 \
  --eval_interval 20 \
  --save_interval 30 \
  --batch_size 2 \
  --context_length 128 \
  --device cpu
'''

'''
cuda:0
python train_loop.py \
  --train_data /mnt/d/LLM/Assignment1/data/ts_train_tokens_uint16.npy \
  --val_data /mnt/d/LLM/Assignment1/data/ts_valid_tokens_uint16.npy \
  --vocab_size 10000 \
  --d_model 512 \
  --d_ff 1344 \
  --num_layers 4 \
  --num_heads 16 \
  --context_length 256 \
  --total_steps 80000 \
  --batch_size 16 \
  --device cuda:0
'''