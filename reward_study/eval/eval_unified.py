"""
Unified exact-match evaluation for v2/v3/v4a/v4b checkpoints.

Design:
- Load test.parquet (1024 prompts)
- Use vLLM to generate one response per prompt (greedy)
- Parse output with STRICT rules — no partial credit:
    * exact_match: equation uses exactly the given number set AND result == target
    * right_value_wrong_numbers: result == target BUT numbers don't match
    * wrong_value: result != target
    * no_equation: can't extract <answer>...</answer> equation
- Also record auxiliary metrics: avg response length, */ usage
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
from vllm import LLM, SamplingParams


def extract_equation(text):
    if "Assistant:" in text:
        text = text.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[1]
    m = re.findall(r'<answer>(.*?)</answer>', text, flags=re.DOTALL)
    if not m:
        return None
    return m[-1].strip()


def numbers_used(equation_str):
    try:
        return [int(n) for n in re.findall(r'\d+', equation_str)]
    except Exception:
        return []


def evaluate_equation(equation_str):
    try:
        if not re.match(r'^[\d+\-*/().\s]+$', equation_str):
            return None
        result = eval(equation_str, {"__builtins__": None}, {})
        if not isinstance(result, (int, float)):
            return None
        return result
    except Exception:
        return None


def classify(equation, target, numbers):
    if equation is None:
        return "no_equation", None
    used = numbers_used(equation)
    result = evaluate_equation(equation)
    if result is None:
        return "no_equation", None
    value_matches = abs(result - target) < 1e-5
    numbers_match = sorted(used) == sorted(numbers)
    if value_matches and numbers_match:
        return "exact_match", result
    if value_matches and not numbers_match:
        return "right_value_wrong_numbers", result
    return "wrong_value", result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to actor checkpoint")
    ap.add_argument("--tag", required=True, help="Name tag (v2/v3/v4a/v4b)")
    ap.add_argument("--test_parquet", default="/data/fangda/data/countdown/test.parquet")
    ap.add_argument("--out_dir", default="/data/fangda/tinyzero/report_0414/eval_results")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.test_parquet)
    prompts = [row[0]["content"] for row in df["prompt"]]
    targets = [int(r["ground_truth"]["target"]) for r in df["reward_model"]]
    numbers_list = [[int(x) for x in r["ground_truth"]["numbers"]] for r in df["reward_model"]]

    print(f"[{args.tag}] Loading {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=True,
        dtype="bfloat16",
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    print(f"[{args.tag}] Generating for {len(prompts)} prompts")
    outputs = llm.generate(prompts, sampling)

    records = []
    counts = {"exact_match": 0, "right_value_wrong_numbers": 0, "wrong_value": 0, "no_equation": 0}
    total_tokens = 0
    used_complex = 0
    used_complex_correct = 0

    for i, out in enumerate(outputs):
        response = out.outputs[0].text
        response_tokens = len(out.outputs[0].token_ids)
        total_tokens += response_tokens

        equation = extract_equation("Assistant:" + response)
        category, result = classify(equation, targets[i], numbers_list[i])
        counts[category] += 1

        has_complex = bool(equation and re.search(r'[*/]', equation))
        if has_complex:
            used_complex += 1
            if category == "exact_match":
                used_complex_correct += 1

        records.append({
            "idx": i,
            "target": targets[i],
            "numbers": numbers_list[i],
            "equation": equation,
            "result": result,
            "category": category,
            "has_complex": has_complex,
            "response_tokens": response_tokens,
            "response": response,
        })

    n = len(prompts)
    wrong_n = counts["right_value_wrong_numbers"] + counts["wrong_value"] + counts["no_equation"]
    # "wrong with deviation==0" (from previous report) corresponds to right_value_wrong_numbers
    wrong_dev_zero_ratio = counts["right_value_wrong_numbers"] / wrong_n if wrong_n > 0 else 0.0

    summary = {
        "tag": args.tag,
        "model": args.model,
        "n_samples": n,
        "exact_match_accuracy": counts["exact_match"] / n,
        "right_value_wrong_numbers_ratio": counts["right_value_wrong_numbers"] / n,
        "wrong_value_ratio": counts["wrong_value"] / n,
        "no_equation_ratio": counts["no_equation"] / n,
        "wrong_with_dev_zero_ratio_among_wrong": wrong_dev_zero_ratio,
        "avg_response_tokens": total_tokens / n,
        "complex_ops_attempt_rate": used_complex / n,
        "complex_ops_correct_rate": used_complex_correct / n,
        "raw_counts": counts,
    }

    print(f"\n[{args.tag}] Summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save
    with open(f"{args.out_dir}/{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{args.out_dir}/{args.tag}_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()
