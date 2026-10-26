import json
import os
import matplotlib.pyplot as plt

def plot_from_jsonl(log_path, save_dir=None):
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        print("No entries found!")
        return

    train_steps = []
    train_losses = []
    train_lrs = []
    train_times = []

    eval_steps = []
    eval_losses = []
    eval_ppls = []
    eval_times = []

    for e in entries:
        if e["split"] == "train":
            train_steps.append(e["step"])
            train_losses.append(e["loss"])
            train_lrs.append(e["lr"])
            train_times.append(e["elapsed_total"] / 60.0)
        elif e["split"] == "eval":
            eval_steps.append(e["step"])
            eval_losses.append(e["val_loss"])
            eval_ppls.append(e["val_ppl"])
            eval_times.append(e["elapsed_total"] / 60.0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    if train_steps:
        ax1.plot(train_steps, train_losses, alpha=0.3, color='blue', linewidth=0.5, label='train loss (raw)')
    if eval_steps:
        ax1.plot(eval_steps, eval_losses, 'r-o', markersize=4, linewidth=2, label='val loss')

    ax1.set_xlabel('Gradient Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss vs Steps')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'loss_vs_steps.png'), dpi=150)
        print(f"Saved: {os.path.join(save_dir, 'loss_vs_steps.png')}")
    plt.show()

    if train_times and eval_times:
        fig, ax2 = plt.subplots(figsize=(10, 6))

        if train_steps:
            ax2.plot(train_times, train_losses, alpha=0.3, color='blue', linewidth=0.5, label='train loss (raw)')
        if eval_times:
            ax2.plot(eval_times, eval_losses, 'r-o', markersize=4, linewidth=2, label='val loss')

        ax2.set_xlabel('Wall-clock Time (minutes)')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training & Validation Loss vs Wall-clock Time')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'loss_vs_time.png'), dpi=150)
            print(f"Saved: {os.path.join(save_dir, 'loss_vs_time.png')}")
        plt.show()

    if train_steps and train_lrs:
        fig, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(train_steps, train_lrs, 'g-', linewidth=1.5)
        ax3.set_xlabel('Gradient Steps')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'lr_schedule.png'), dpi=150)
            print(f"Saved: {os.path.join(save_dir, 'lr_schedule.png')}")
        plt.show()

    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    if eval_losses:
        best_val_loss = min(eval_losses)
        best_step = eval_steps[eval_losses.index(best_val_loss)]
        print(f"Best val_loss: {best_val_loss:.4f} at step {best_step}")
    if train_losses:
        print(f"Final train_loss: {train_losses[-1]:.4f}")
    if eval_times:
        print(f"Total wall-clock time: {eval_times[-1]:.1f} minutes")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot experiment logs")
    parser.add_argument("--log_dir", type=str, default=r"/mnt/d/LLM/Assignment1/data/logs")
    parser.add_argument("--experiment_name", type=str, default="exp_default")
    parser.add_argument("--save_dir", type=str, default=r"/mnt/d/LLM/Assignment1/data/logs/plots")

    args = parser.parse_args()
    log_path = os.path.join(args.log_dir, f"{args.experiment_name}.jsonl")
    plot_from_jsonl(log_path, args.save_dir)