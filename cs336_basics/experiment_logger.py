import json
import time
import os
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class LogEntry:
    step: int
    timestamp: float
    elapsed_total: float
    loss: float = None
    lr: float = None
    val_loss: float = None
    val_ppl: float = None
    tokens_per_sec: float = None
    split: str = "train"


class ExperimentLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{experiment_name}.jsonl")
        self.start_time = time.time()
        self.entries: List[LogEntry] = []
        print(f"[Logger] Logging to {self.log_path}")

    def log_train_step(self, step: int, loss: float, lr: float, tokens_per_sec: float):
        entry = LogEntry(
            step=step,
            timestamp=time.time(),
            elapsed_total=time.time() - self.start_time,
            loss=loss,
            lr=lr,
            tokens_per_sec=tokens_per_sec,
            split="train",
        )
        self.entries.append(entry)
        self._append_jsonl(entry)

    def log_eval_step(self, step: int, val_loss: float, val_ppl: float):
        entry = LogEntry(
            step=step,
            timestamp=time.time(),
            elapsed_total=time.time() - self.start_time,
            val_loss=val_loss,
            val_ppl=val_ppl,
            split="eval",
        )
        self.entries.append(entry)
        self._append_jsonl(entry)

    def _append_jsonl(self, entry: LogEntry):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def get_entries(self) -> List[LogEntry]:
        return self.entries

    def load_entries(self) -> List[dict]:
        entries = []
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        return entries