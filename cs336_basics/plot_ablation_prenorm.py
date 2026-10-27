import json
import os
import matplotlib.pyplot as plt

sweep_dir = r"/mnt/d/LLM/Assignment1/data/ablation_prenorm"

configs = [
    ("baseline_prenorm_lr8e4", "Pre-norm (Baseline)"),
    ("postnorm_lr8e4",         "Post-norm"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for exp_name, label in configs:
    log_path = os.path.join(sweep_dir, exp_name, "logs", f"{exp_name}.jsonl")
    if not os.path.exists(log_path):
        print(f"Missing: {log_path}")
        continue

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    with open(log_path, "r") as f:
        for line in f:
            e = json.loads(line)
            if e["split"] == "train":
                train_steps.append(e["step"])
                train_losses.append(e["loss"])
            elif e["split"] == "eval":
                eval_steps.append(e["step"])
                eval_losses.append(e["val_loss"])

    window = 30
    smoothed = []
    for i in range(len(train_losses)):
        start = max(0, i - window + 1)
        smoothed.append(sum(train_losses[start:i+1]) / (i - start + 1))

    ax1.plot(train_steps, smoothed, label=label, linewidth=1.5)
    if eval_steps:
        ax2.plot(eval_steps, eval_losses, label=label, marker='o', markersize=4)

ax1.set_xlabel("Step")
ax1.set_ylabel("Train Loss (smoothed)")
ax1.set_title("Train Loss — Pre-norm vs Post-norm")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Step")
ax2.set_ylabel("Val Loss")
ax2.set_title("Validation Loss — Pre-norm vs Post-norm")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(sweep_dir, "prenorm_vs_postnorm_curves.png")
plt.savefig(save_path, dpi=150)
print(f"Saved to {save_path}")
plt.show()