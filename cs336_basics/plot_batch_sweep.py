import json
import os
import matplotlib.pyplot as plt

sweep_dir = r"/mnt/d/LLM/Assignment1/data/batch_sweep"
batch_list = [1, 4, 16, 32, 64]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for bs in batch_list:
    exp_dirs = [d for d in os.listdir(sweep_dir) if f"bs{bs}" in d]
    if not exp_dirs:
        print(f"No data for batch={bs}")
        continue

    log_path = os.path.join(sweep_dir, exp_dirs[0], "logs", f"{exp_dirs[0]}.jsonl")
    if not os.path.exists(log_path):
        print(f"Missing log: {log_path}")
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

    window = 50
    smoothed = []
    for i in range(len(train_losses)):
        start = max(0, i - window + 1)
        smoothed.append(sum(train_losses[start:i + 1]) / (i - start + 1))

    label = f"batch={bs}"
    ax1.plot(train_steps, smoothed, label=label, linewidth=1.5)

    if eval_steps:
        ax2.plot(eval_steps, eval_losses, label=label, marker='o', markersize=4)

ax1.set_xlabel("Step")
ax1.set_ylabel("Train Loss (smoothed)")
ax1.set_title("Train Loss — Batch Size Comparison")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Step")
ax2.set_ylabel("Val Loss")
ax2.set_title("Validation Loss — Batch Size Comparison")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(sweep_dir, "batch_sweep_curves.png"), dpi=150)
print(f"Saved to {sweep_dir}/batch_sweep_curves.png")
plt.show()