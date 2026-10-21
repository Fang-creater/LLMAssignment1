import numpy as np
import os

SHARD_DIR = r"/mnt/d/LLM/Assignment1/data/token_shards"
NPY_OUT = r"/mnt/d/LLM/Assignment1/data/owt_train_tokens_uint16.npy"

# ---- 1. 算总 token 数 ----
print("Counting total tokens...")
total_tokens = 0
shard_files = sorted([f for f in os.listdir(SHARD_DIR) if f.endswith('.npy')])
print(f"Found {len(shard_files)} shards")

for fname in shard_files:
    path = os.path.join(SHARD_DIR, fname)
    arr = np.load(path, mmap_mode='r')
    total_tokens += arr.shape[0]
    del arr

print(f"Total tokens: {total_tokens:,}")

# ---- 2. 创建标准 npy（先写空数组，再逐段填充）----
print(f"Creating output file ({total_tokens * 2 / 1024**2:.1f} MB)...")
out = np.lib.format.open_memmap(NPY_OUT, mode='w+', dtype=np.uint16, shape=(total_tokens,))

offset = 0
for i, fname in enumerate(shard_files):
    path = os.path.join(SHARD_DIR, fname)
    shard = np.load(path, mmap_mode='r')
    n = shard.shape[0]
    out[offset:offset + n] = shard
    offset += n
    del shard
    if (i + 1) % 5 == 0 or i == len(shard_files) - 1:
        print(f"  Written {i+1}/{len(shard_files)} shards ({offset:,}/{total_tokens:,} tokens)")

out.flush()

# ---- 3. 验证：用 np.load 确认标准 npy 能正常打开 ----
print("Verifying with np.load...")
a = np.load(NPY_OUT, allow_pickle=False)
print(f"✅ shape={a.shape}, dtype={a.dtype}")
print(f"   max={int(a.max())}, min={int(a.min())}")
print(f"   Saved → {NPY_OUT}")