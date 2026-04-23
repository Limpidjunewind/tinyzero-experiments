"""
Logit distribution analysis for collapsed vs. pre-collapse checkpoints.
Measures entropy of token distribution at generation start, after <think>.

Usage:
    python logit_analysis.py \
        --step50 checkpoints/TinyZero/ppo-3b-bs128-mini16/actor/global_step_50 \
        --step100 checkpoints/TinyZero/ppo-3b-bs128-mini16/actor/global_step_100

Output: top-10 token probabilities + entropy H(π) for each checkpoint.
"""

import argparse
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    "Using the numbers [92, 13, 71], create an equation that equals 34. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.",
    "Using the numbers [25, 4, 17], create an equation that equals 32. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.",
    "Using the numbers [56, 23, 67], create an equation that equals 12. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.",
]

SYSTEM = ("A conversation between User and Assistant. The user asks a question, and the "
          "Assistant solves it. The assistant first thinks about the reasoning process in "
          "the mind and then provides the user with the answer.")

def build_prefix(tokenizer, user_msg):
    """Build prompt up to and including <think>, so we measure the first generated token."""
    text = (f"{SYSTEM}\nUser: {user_msg}\nAssistant: Let me solve this step by step.\n<think>")
    return text

def compute_entropy(probs):
    """H(π) = -Σ p log p in nats, converted to bits."""
    probs = probs.clamp(min=1e-10)
    return -(probs * probs.log()).sum().item() / math.log(2)

def analyze_checkpoint(ckpt_path, tokenizer, prompts, device, top_k=10):
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    entropies = []

    for i, user_msg in enumerate(prompts):
        prefix = build_prefix(tokenizer, user_msg)
        input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids).logits  # [1, seq_len, vocab]

        # Take logits at last position (what comes right after <think>)
        next_token_logits = logits[0, -1, :]  # [vocab]
        probs = F.softmax(next_token_logits.float(), dim=-1)

        entropy = compute_entropy(probs)
        entropies.append(entropy)

        top_probs, top_ids = probs.topk(top_k)
        top_tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]

        print(f"\nPrompt {i+1}: {user_msg[:60]}...")
        print(f"  Entropy H(π) = {entropy:.4f} bits")
        print(f"  Top-{top_k} tokens after <think>:")
        for tok, prob in zip(top_tokens, top_probs.tolist()):
            bar = '█' * int(prob * 40)
            print(f"    {repr(tok):12s}  {prob:.4f}  {bar}")

        # Also generate a short response to see actual output
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=60,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
        print(f"  Generated: {repr(generated[:120])}")

    print(f"\n  Mean entropy across {len(prompts)} prompts: {sum(entropies)/len(entropies):.4f} bits")
    del model
    torch.cuda.empty_cache()
    return entropies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step50', required=True, help='Path to global_step_50 checkpoint')
    parser.add_argument('--step100', required=True, help='Path to global_step_100 checkpoint')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.step50)

    e50 = analyze_checkpoint(args.step50, tokenizer, PROMPTS, args.device)
    e100 = analyze_checkpoint(args.step100, tokenizer, PROMPTS, args.device)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Step 50  mean entropy: {sum(e50)/len(e50):.4f} bits")
    print(f"Step 100 mean entropy: {sum(e100)/len(e100):.4f} bits")
    print(f"Drop: {sum(e50)/len(e50) - sum(e100)/len(e100):.4f} bits")


if __name__ == '__main__':
    main()
