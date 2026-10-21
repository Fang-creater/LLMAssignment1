# pkl_to_hf_tokenizer.py
import pickle
import json
import os
'''
VOCAB_PKL  = r"/mnt/d/LLM/Assignment1/data/owt_train_vocab.pkl"
MERGES_PKL = r"/mnt/d/LLM/Assignment1/data/owt_train_merges.pkl"
OUT_DIR    = r"/mnt/d/LLM/Assignment1/data/hf_owt_train_tokenizer"
'''
VOCAB_PKL  = r"/mnt/d/LLM/Assignment1/data/owt_valid_vocab.pkl"
MERGES_PKL = r"/mnt/d/LLM/Assignment1/data/owt_valid_merges.pkl"
OUT_DIR    = r"/mnt/d/LLM/Assignment1/data/hf_owt_valid_tokenizer"
os.makedirs(OUT_DIR, exist_ok=True)

with open(VOCAB_PKL, "rb") as f:
    vocab = pickle.load(f)       # {int_id: bytes}
with open(MERGES_PKL, "rb") as f:
    merges = pickle.load(f)      # [(bytes_a, bytes_b), ...]

# ============ GPT-2 ByteLevel 编码 ============
# 规则：空格(0x20) → "Ġ"(U+0120)，其他字节原样转字符
def bytes_to_bytelevel(b):
    """bytes → GPT-2 ByteLevel 字符串（空格变 Ġ）"""
    if not isinstance(b, bytes):
        return str(b)
    result = []
    for byte in b:
        if byte == 0x20:
            result.append("Ġ")
        elif byte < 128:
            result.append(chr(byte))
        else:
            # 非 ASCII 字节，用 Unicode 私有区映射（GPT-2 做法）
            result.append(chr(byte + 256))
    return "".join(result)

# -------- 1. vocab.json --------
vocab_json = {}
for tid, b in vocab.items():
    key = bytes_to_bytelevel(b)
    vocab_json[key] = tid

vocab_path = os.path.join(OUT_DIR, "vocab.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab_json, f, ensure_ascii=False, indent=2)
print(f"✅ vocab.json: {len(vocab_json):,} entries → {vocab_path}")

# -------- 2. merges.txt --------
merges_lines = ["#version: 0.2\n"]
for a_bytes, b_bytes in merges:
    a = bytes_to_bytelevel(a_bytes)
    b = bytes_to_bytelevel(b_bytes)
    merges_lines.append(f"{a} {b}\n")

merges_path = os.path.join(OUT_DIR, "merges.txt")
with open(merges_path, "w", encoding="utf-8") as f:
    f.writelines(merges_lines)
print(f"✅ merges.txt: {len(merges):,} entries → {merges_path}")

# -------- 3. special_tokens_map.json --------
special = {
    "eos_token": "<|endoftext|>",
    "bos_token": "<|endoftext|>",
    "unk_token": None,
    "pad_token": None
}
with open(os.path.join(OUT_DIR, "special_tokens_map.json"), "w") as f:
    json.dump(special, f, indent=2)

with open(os.path.join(OUT_DIR, "tokenizer_config.json"), "w") as f:
    json.dump({"model_max_length": 1024, **special}, f, indent=2)

print(f"\n✅ All HF tokenizer files written to {OUT_DIR}")