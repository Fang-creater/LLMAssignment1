"""
Run: cd /mnt/d/LLM/Assignment1/cs336_basics && source /mnt/d/LLM/Assignment1/.venv/bin/activate && python tokenizer_experiments.py
"""
import os
import pickle
import random
import sys
import time
import numpy as np

sys.path.insert(0, "/mnt/d/LLM/Assignment1/cs336_basics")
from tokenizer import Tokenizer

TS_VOCAB_PATH = "/mnt/d/LLM/Assignment1/data/ts_valid_vocab.pkl"
TS_MERGES_PATH = "/mnt/d/LLM/Assignment1/data/ts_valid_merges.pkl"
OW_VOCAB_PATH = "/mnt/d/LLM/Assignment1/data/owt_valid_vocab.pkl"
OW_MERGES_PATH = "/mnt/d/LLM/Assignment1/data/owt_valid_merges.pkl"
TS_CORPUS_PATH = "/mnt/d/LLM/Assignment1/data/TinyStoriesV2-GPT4-valid.txt"
OW_CORPUS_PATH = "/mnt/d/LLM/Assignment1/data/owt_valid.txt"
'''
OW_TRAIN_VOCAB_PATH  = "/mnt/d/LLM/Assignment1/data/owt_train_vocab.pkl"
OW_TRAIN_MERGES_PATH = "/mnt/d/LLM/Assignment1/data/owt_train_merges.pkl"
OW_TRAIN_CORPUS_PATH = "/mnt/d/LLM/Assignment1/data/owt_train.txt"
OW_TRAIN_NPY_PATH    = "/mnt/d/LLM/Assignment1/data/owt_train_tokens_uint16.npy"
'''
NUM_DOCS = 10
SEED = 42

def load_tokenizer(vocab_path, merges_path, special_tokens=None):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    return Tokenizer(vocab, merges, special_tokens)

print("Loading tokenizers...")
ts_tok = load_tokenizer(TS_VOCAB_PATH, TS_MERGES_PATH, special_tokens=["<|endoftext|>"])
ow_tok = load_tokenizer(OW_VOCAB_PATH, OW_MERGES_PATH, special_tokens=["<|endoftext|>"])
'''
ow_train_tok = load_tokenizer(OW_TRAIN_VOCAB_PATH, OW_TRAIN_MERGES_PATH,special_tokens=["<|endoftext|>"])
'''
print("  Done.")

def sample_documents(corpus_path, n, seed=42):
    random.seed(seed)
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if text.count("\n\n") > text.count("\n") * 0.3:
        docs = [d.strip() for d in text.split("\n\n") if len(d.strip()) > 50]
    else:
        docs = [d.strip() for d in text.split("\n") if len(d.strip()) > 50]

    print(f"  Total docs available: {len(docs):,}")
    sampled = random.sample(docs, min(n, len(docs)))
    combined = "\n\n".join(sampled)
    return combined, sampled


print(f"\nSampling {NUM_DOCS} docs from each corpus...")
ts_sample, ts_docs = sample_documents(TS_CORPUS_PATH, NUM_DOCS)
ow_sample, ow_docs = sample_documents(OW_CORPUS_PATH, NUM_DOCS)

ts_bytes = len(ts_sample.encode("utf-8"))
ow_bytes = len(ow_sample.encode("utf-8"))
print(f"  TS sample: {ts_bytes:,} bytes")
print(f"  OW sample: {ow_bytes:,} bytes")

def compression_ratio(text, tokenizer):
    ids = tokenizer.encode(text)
    num_tokens = len(ids)
    num_bytes = len(text.encode("utf-8"))
    return num_bytes / num_tokens, num_tokens


ts_cr, ts_ntok = compression_ratio(ts_sample, ts_tok)
ow_cr, ow_ntok = compression_ratio(ow_sample, ow_tok)

print(f"  TinyStories tokenizer on TS sample: {ts_cr:.2f} bytes/token ({ts_ntok:,} tokens)")
print(f"  OpenWebText tokenizer on OW sample: {ow_cr:.2f} bytes/token ({ow_ntok:,} tokens)")

cross_cr, cross_ntok = compression_ratio(ow_sample, ts_tok)
print(f"  TS tokenizer on OW sample: {cross_cr:.2f} bytes/token ({cross_ntok:,} tokens)")
print(f"  Native OW tokenizer on OW: {ow_cr:.2f} bytes/token ({ow_ntok:,} tokens)")
print(f"  → Drop from {ow_cr:.2f} to {cross_cr:.2f} bytes/token")

target_bytes = 5_000_000
repeat = max(1, target_bytes // ts_bytes)
bench_text = ts_sample * repeat
bench_bytes = len(bench_text.encode("utf-8"))

_ = ts_tok.encode(ts_sample[:2000])

start = time.perf_counter()
ids = ts_tok.encode(bench_text)
elapsed = time.perf_counter() - start

throughput = bench_bytes / elapsed
pile_bytes = 825 * 1024**3
pile_time = pile_bytes / throughput
pile_hours = pile_time / 3600
pile_days = pile_hours / 24

print(f"  Benchmark text: {bench_bytes / 1e6:.1f} MB ({repeat}x repeat)")
print(f"  Encoded {len(ids):,} tokens in {elapsed:.2f}s")
print(f"  Throughput: {throughput / 1e6:.2f} MB/s")
print(f"  Pile (825 GB) ETA: {pile_days:.1f} days ({pile_hours:.0f} hours)")
print(f"  (single CPU core, pure Python)")

print("  Encoding TS validation set (full)...")
with open(TS_CORPUS_PATH, "r", encoding="utf-8", errors="ignore") as f:
    ts_full = f.read()
ts_val_ids = ts_tok.encode(ts_full)

print("  Encoding OW validation set (full)...")
with open(OW_CORPUS_PATH, "r", encoding="utf-8", errors="ignore") as f:
    ow_full = f.read()
ow_val_ids = ow_tok.encode(ow_full)
'''
ow_train_ids = []
chunk_size = 1024 * 1024
file_size = os.path.getsize(OW_TRAIN_CORPUS_PATH)
processed = 0
t0 = time.time()

with open(OW_TRAIN_CORPUS_PATH, "r", encoding="utf-8", errors="ignore") as f:
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        ids = ow_train_tok.encode(chunk)
        ow_train_ids.extend(ids)
        processed += len(chunk.encode("utf-8"))
        pct = processed / file_size * 100
        print(f"    {pct:6.1f}%  | {len(ow_train_ids):>12,} tokens | {time.time()-t0:.0f}s")
'''
ts_arr = np.array(ts_val_ids, dtype=np.uint16)
ow_arr = np.array(ow_val_ids, dtype=np.uint16)
'''
ow_train_arr = np.array(ow_train_ids, dtype=np.uint16)
'''
print(f"  TS val: {len(ts_val_ids):,} tokens → {ts_arr.nbytes / 1e6:.1f} MB (uint16)")
print(f"  OW val: {len(ow_val_ids):,} tokens → {ow_arr.nbytes / 1e6:.1f} MB (uint16)")
print(f"  Max TS token ID: {max(ts_val_ids):,}  (uint16 max: 65,535)")
print(f"  Max OW token ID: {max(ow_val_ids):,}  (uint16 max: 65,535)")
print(f"  Vocab sizes: TS={len(ts_tok.vocab):,}, OW={len(ow_tok.vocab):,}")

# Save
np.save("/mnt/d/LLM/Assignment1/data/ts_val_tokens_uint16.npy", ts_arr)
np.save("/mnt/d/LLM/Assignment1/data/ow_val_tokens_uint16.npy", ow_arr)
'''
np.save("/mnt/d/LLM/Assignment1/data/ow_train_tokens_uint16.npy", ow_train_arr)
'''
print("  Saved .npy files.")