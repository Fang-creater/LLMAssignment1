import pickle
import regex as re
from collections.abc import Iterable, Iterator

PAT = r"""(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v:k for k,v in vocab.items()}
        self.merge_ranks = {}

        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.byte_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = tok_bytes
                self.byte_to_id[tok_bytes] = new_id

        for rank, (a_bytes, b_bytes) in enumerate(merges):
            merged_bytes = a_bytes + b_bytes
            if merged_bytes in self.byte_to_id:
                a_id = self.byte_to_id[a_bytes]
                b_id = self.byte_to_id[b_bytes]
                self.merge_ranks[(a_id, b_id)] = rank

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)
        return cls(vocab,merges,special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=lambda x: len(x), reverse=True)
            pattern = re.compile("|".join(re.escape(tok) for tok in sorted_tokens))
            result = []
            last_end = 0
            for m in pattern.finditer(text):
                if m.start() > last_end:
                    result.extend(self._encode_chunk(text[last_end:m.start()]))
                result.append(self.byte_to_id[m.group().encode("utf-8")])
                last_end = m.end()
            if last_end < len(text):
                result.extend(self._encode_chunk(text[last_end:]))
            return result
        else:
            return self._encode_chunk(text)

    def _encode_chunk(self, text: str) -> list[int]:
        token_ids = []
        for match in re.finditer(PAT, text):
            token_str = match.group()
            byte_seq = token_str.encode("utf-8")
            ids = [self.byte_to_id[bytes([b])] for b in byte_seq]
            ids = self._merge_ids(ids)
            token_ids.extend(ids)

        return token_ids

    def _merge_ids(self, ids: list[int]) -> list[int]:
        while len(ids) > 1:
            best_rank = float('inf')
            best_i = -1
            for i in range(len(ids) - 1):
                key = (ids[i], ids[i + 1])
                if key in self.merge_ranks:
                    if self.merge_ranks[key] < best_rank:
                        best_rank = self.merge_ranks[key]
                        best_i = i

            if best_i == -1:
                break

            a_id = ids[best_i]
            b_id = ids[best_i + 1]
            merged_bytes = self.vocab[a_id] + self.vocab[b_id]
            merged_id = self.byte_to_id[merged_bytes]
            ids = ids[:best_i] + [merged_id] + ids[best_i + 2:]

        return ids

    '''Simplify version'''
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            for tid in token_ids:
                yield tid

    def decode(self, ids: list[int]) -> str:
        byte_seq = b''.join(self.vocab[tid] for tid in ids)
        return byte_seq.decode('utf-8', errors='replace')