import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
'''
PAT = r"""(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""(use this one)
'''
def train_bpe(
    input_path : str,
    vocab_size : int,
    special_tokens : list[str],
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:

    chunk_size = 1024 * 1024

    if special_tokens:
        delimiter = "|".join(re.escape(tok) for tok in special_tokens)
    else:
        delimiter = None

    pretoken_seqs = []
    leftover = ""

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            text = leftover + chunk

            last_newline = text.rfind('\n')
            if last_newline != -1:
                process_text = text[:last_newline]
                leftover = text[last_newline:]
            else:
                if len(text) > 1024 * 1024:
                    process_text = text
                    leftover = ""
                else:
                    leftover = text
                    continue

            if delimiter:
                documents = re.split(delimiter, process_text)
            else:
                documents = [process_text]

            for doc in documents:
                if not doc:
                    continue
                for match in re.finditer(PAT, doc):
                    token_str = match.group()
                    byte_seq = list(token_str.encode("utf-8"))
                    if len(byte_seq) > 0:
                        pretoken_seqs.append(byte_seq)

    if leftover:
        if delimiter:
            documents = re.split(delimiter, leftover)
        else:
            documents = [leftover]
        for doc in documents:
            if not doc:
                continue
            for match in re.finditer(PAT, doc):
                token_str = match.group()
                byte_seq = list(token_str.encode("utf-8"))
                if len(byte_seq) > 0:
                    pretoken_seqs.append(byte_seq)

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    for tok in special_tokens:
        tok_bytes = tok.encode("utf-8")
        vocab[len(vocab)] = tok_bytes

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab,merges

    pair_counts = defaultdict(int)
    pair_to_seqs = defaultdict(set)

    for seq_idx, seq in enumerate(pretoken_seqs):
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1
            pair_to_seqs[pair].add(seq_idx)

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]))[0]
        a, b = best_pair

        merges.append((vocab[a], vocab[b]))

        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]

        affected = pair_to_seqs.pop(best_pair, set())
        pair_counts.pop(best_pair, None)

        for seq_idx in affected:
            seq = pretoken_seqs[seq_idx]
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] -= 1
                pair_to_seqs[pair].discard(seq_idx)
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                    pair_to_seqs.pop(pair, None)

        for seq_idx in affected:
            seq = pretoken_seqs[seq_idx]
            i = 0
            while i < len(seq) - 1:
                if seq[i] == a and seq[i + 1] == b:
                    seq[i] = new_id
                    seq.pop(i + 1)
                else:
                    i += 1

        for seq_idx in affected:
            seq = pretoken_seqs[seq_idx]
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1
                pair_to_seqs[pair].add(seq_idx)

    return vocab,merges