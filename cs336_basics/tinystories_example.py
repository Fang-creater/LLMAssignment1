import torch
from tokenizers import Tokenizer
from decoding import decode
from model_layer import create_model

def main():
    TOKENIZER_PATH = r"/mnt/d/LLM/Assignment1/data/hf_ts_train_tokenizer/tokenizer.json"
    CKPT_PATH = r"/mnt/d/LLM/Assignment1/data/checkpoints/ckpt_step_80000.pt"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"EOT token ID: {eot_id}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    class Args:
        vocab_size = tokenizer.get_vocab_size()
        d_model = 512
        num_layers = 4
        num_heads = 16
        d_ff = 1344
        context_length = 512
        device = DEVICE
    model = create_model(Args(), DEVICE)

    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE)
    model.eval()
    step = ckpt.get('step', ckpt.get('global_step', '?'))
    if step != '?':
        print(f"Model loaded (step {step})")
    else:
        print(f"Model loaded (keys: {list(ckpt.keys())})")

    prompt = (
        "Once upon a time there was a little boy named Ben. "
        "Ben loved to explore the world around him. "
        "He saw many amazing things, like beautiful vases that were on display in a store."
    )
    print(f"\nPrompt: {prompt}")
    prompt_tokens = tokenizer.encode(prompt).ids
    print(f"Prompt token count: {len(prompt_tokens)}")
    generated = decode(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=300,
        temperature=0.8,
        top_p=0.95,
        end_of_text_token_id=eot_id,
        context_length=512,
        device=DEVICE,
    )

    generated_text = tokenizer.decode(generated)
    print(f"\n--- Generated text ---")
    print(generated_text)

    print("\n\n" + "=" * 60)
    print("PARAMETER COMPARISON")
    print("=" * 60)

    test_cases = [
        ("Greedy (τ=0)", 0.0, 1.0),
        ("τ=0.5, top_p=1.0", 0.5, 1.0),
        ("τ=0.8, top_p=0.95", 0.8, 0.95),
        ("τ=1.0, top_p=0.9", 1.0, 0.9),
        ("τ=1.2, top_p=0.8", 1.2, 0.8),
    ]

    comparison_prompt = "Once upon a time there was a little boy named Ben."
    comparison_tokens = tokenizer.encode(comparison_prompt).ids

    for desc, temp, p in test_cases:
        gen = decode(
            model=model,
            prompt_tokens=comparison_tokens,
            max_new_tokens=120,
            temperature=temp,
            top_p=p,
            end_of_text_token_id=eot_id,
            context_length=512,
            device=DEVICE,
        )
        text = tokenizer.decode(gen)
        print(f"\n--- {desc} ---")
        print(text)


if __name__ == "__main__":
    main()