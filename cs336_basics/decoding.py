import torch
import torch.nn.functional as F
from typing import List

def decode(
    model,
    prompt_tokens: List[int],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_of_text_token_id: int = 50256,
    context_length: int = 512,
    device: str = "cuda",
) -> List[int]:
    model.eval()
    generated = list(prompt_tokens)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(generated) > context_length:
                context = generated[-context_length:]
            else:
                context = generated
            input_ids = torch.tensor(
                [context], dtype=torch.long, device=device
            )
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            if temperature <= 0:
                next_token_id = torch.argmax(next_token_logits).item()
            else:
                next_token_logits = next_token_logits / temperature
                if top_p < 1.0:
                    next_token_id = _top_p_sample(next_token_logits, top_p)
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token_id)
            if next_token_id == end_of_text_token_id:
                break
    return generated


def _top_p_sample(logits: torch.Tensor, top_p: float) -> int:
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=0)
    cutoff = torch.searchsorted(cumsum_probs, top_p, right=False).item()
    cutoff = max(1, cutoff + 1)
    filtered_probs = sorted_probs[:cutoff]
    filtered_indices = sorted_indices[:cutoff]
    filtered_probs = filtered_probs / filtered_probs.sum()
    sampled_idx = torch.multinomial(filtered_probs, num_samples=1).item()
    return filtered_indices[sampled_idx].item()