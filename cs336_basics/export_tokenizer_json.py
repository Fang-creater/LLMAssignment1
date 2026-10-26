import os, json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

TOKENIZER_DIR = r"/mnt/d/LLM/Assignment1/data/hf_ts_train_tokenizer"

with open(os.path.join(TOKENIZER_DIR, "vocab.json"), "r", encoding="utf-8") as f:
    vocab = json.load(f)
with open(os.path.join(TOKENIZER_DIR, "merges.txt"), "r", encoding="utf-8") as f:
    merges = []
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

output_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")
tok.save(output_path)
print(f"Saved tokenizer.json → {output_path}")