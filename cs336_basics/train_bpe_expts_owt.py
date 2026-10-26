import os
import pickle
import time
import psutil

from train_bpe2 import train_bpe

input_path = r"/mnt/d/LLM/Assignment1/data/owt_valid_100mb.txt"

start = time.time()
vocab,merges = train_bpe(
    input_path = input_path,
    vocab_size = 10000,
    special_tokens = ["<|endoftext|>"]
)
elapsed = time.time() - start

print(f"Time: {elapsed:.1f}s")
print(f"Vocab size: {len(vocab)}")

longest = max(vocab.values(),key = lambda x:len(x))
print(f"Longest token: {longest} ({len(longest)} bytes, decoded: {longest.decode('utf-8', errors='replace')})")

with open(r"/mnt/d/LLM/Assignment1/data/owt_valid_vocab.pkl","wb") as f:
    pickle.dump(vocab,f)
with open(r"/mnt/d/LLM/Assignment1/data/owt_valid_merges.pkl", "wb") as f:
    pickle.dump(merges,f)

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024**3:.2f} GB")

print("Saved to disk.")