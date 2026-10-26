import os, time, json, numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

TOKENIZER_DIR = r"/mnt/d/LLM/Assignment1/data/hf_ts_train_tokenizer"
TEXT_IN = r"/mnt/d/LLM/Assignment1/data/TinyStoriesV2-GPT4-valid.txt"
NPY_OUT = r"/mnt/d/LLM/Assignment1/data/ts_valid_tokens_uint16.npy"
SHARD_DIR = r"/mnt/d/LLM/Assignment1/data/token_shards"
'''
TOKENIZER_DIR = r"/mnt/d/LLM/Assignment1/data/hf_owt_train_tokenizer"
TEXT_IN = r"/mnt/d/LLM/Assignment1/data/owt_valid.txt"
NPY_OUT = r"/mnt/d/LLM/Assignment1/data/ow_val_tokens_uint16.npy"
SHARD_DIR = r"/mnt/d/LLM/Assignment1/data/token_shards"
'''
os.makedirs(SHARD_DIR, exist_ok=True)

print("Loading vocab.json...")
with open(os.path.join(TOKENIZER_DIR, "vocab.json"), "r", encoding="utf-8") as f:
    vocab = json.load(f)

print("Loading merges.txt...")
merges = []
with open(os.path.join(TOKENIZER_DIR, "merges.txt"), "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2:
            merges.append(tuple(parts))

bpe = BPE(vocab=vocab, merges=merges)
tok = Tokenizer(bpe)
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
tok.decoder = ByteLevelDecoder()

CHUNK = 1 << 20
SHARD_SIZE = 100_000_000
file_size = os.path.getsize(TEXT_IN)

shard_idx = 0
current_shard = []
total_tokens = 0
processed = 0
t0 = time.time()

print(f"Encoding {TEXT_IN} ({file_size / 1024 ** 3:.1f} GB)...")

with open(TEXT_IN, "r", encoding="utf-8", errors="ignore") as f:
    while True:
        chunk = f.read(CHUNK)
        if not chunk:
            break

        enc = tok.encode(chunk)
        current_shard.extend(enc.ids)
        total_tokens += len(enc.ids)
        processed += len(chunk.encode("utf-8"))

        if len(current_shard) >= SHARD_SIZE:
            arr = np.array(current_shard, dtype=np.uint16)
            shard_path = os.path.join(SHARD_DIR, f"shard_{shard_idx:04d}.npy")
            np.save(shard_path, arr)
            print(f"  💾 Shard {shard_idx:04d}: {len(arr):,} tokens → {shard_path}")
            current_shard.clear()
            shard_idx += 1

        pct = processed / file_size * 100
        if int(pct) % 5 == 0 and int(pct) != int((processed - len(chunk.encode("utf-8"))) / file_size * 100):
            elapsed = time.time() - t0
            print(f"  {pct:5.1f}% | {total_tokens:,} tokens | {elapsed:.0f}s | shard {shard_idx}")

if current_shard:
    arr = np.array(current_shard, dtype=np.uint16)
    shard_path = os.path.join(SHARD_DIR, f"shard_{shard_idx:04d}.npy")
    np.save(shard_path, arr)
    print(f"  💾 Final shard {shard_idx:04d}: {len(arr):,} tokens → {shard_path}")
    shard_idx += 1

print(f"\nMerging {shard_idx} shards via memmap...")

total_tokens = 0
for i in range(shard_idx):
    shard_path = os.path.join(SHARD_DIR, f"shard_{i:04d}.npy")
    arr = np.load(shard_path)
    total_tokens += len(arr)
    del arr

print(f"  Total tokens: {total_tokens:,}")
print(f"  Allocating memmap: {NPY_OUT} ({total_tokens * 2 / 1024**2:.1f} MB)")

final = np.lib.format.open_memmap(NPY_OUT, mode='w+', dtype=np.uint16, shape=(total_tokens,))
offset = 0
for i in range(shard_idx):
    shard_path = os.path.join(SHARD_DIR, f"shard_{i:04d}.npy")
    shard = np.load(shard_path, mmap_mode='r')
    n = len(shard)
    final[offset:offset + n] = shard
    offset += n
    if (i + 1) % 10 == 0 or i == shard_idx - 1:
        print(f"  Merged {i+1}/{shard_idx} shards...")

final.flush()
print(f"✅ Saved → {NPY_OUT} ({total_tokens:,} tokens, {total_tokens * 2 / 1024**2:.1f} MB)")

final = np.load(NPY_OUT, mmap_mode='r',allow_pickle=False)
print(f"   Max token ID: {int(final.max()):,}")
print(f"   Min token ID: {int(final.min())}")