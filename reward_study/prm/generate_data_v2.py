"""
Improved data generation for PRM training (v2).

Key improvements:
1. Integer-only arithmetic (no floating point division) - matches Countdown task
2. Harder negative samples: off-by-1, off-by-2, sign errors, carry errors
3. More diverse expressions: 2-5 operands with parentheses
4. Balanced difficulty distribution
"""

import json
import random
import os
import argparse
from typing import Tuple, Optional


def safe_eval(expr_str: str) -> Optional[int]:
    """Safely evaluate an integer arithmetic expression."""
    try:
        allowed = set("0123456789+-*/() ")
        if not all(c in allowed for c in expr_str):
            return None
        result = eval(expr_str, {"__builtins__": None}, {})
        if isinstance(result, float):
            if abs(result - round(result)) < 1e-9:
                return int(round(result))
            return None  # Skip non-integer results
        return int(result)
    except:
        return None


def gen_simple_expr(min_val=1, max_val=100) -> Tuple[str, int]:
    """Generate a + b, a - b, a * b (2 operands, no division)."""
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    op = random.choice(['+', '-', '*'])

    if op == '*':
        # Keep multiplication smaller to avoid huge numbers
        b = random.randint(1, 15)

    expr = f"{a} {op} {b}"
    result = safe_eval(expr)
    return expr, result


def gen_three_operand(min_val=1, max_val=100) -> Tuple[str, int]:
    """Generate expressions with 3 operands, sometimes with parentheses."""
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    c = random.randint(min_val, max_val)
    ops = ['+', '-', '*']
    op1 = random.choice(ops)
    op2 = random.choice(ops)

    if op1 == '*':
        b = random.randint(1, 15)
    if op2 == '*':
        c = random.randint(1, 15)

    patterns = [
        f"{a} {op1} {b} {op2} {c}",
        f"({a} {op1} {b}) {op2} {c}",
        f"{a} {op1} ({b} {op2} {c})",
    ]
    expr = random.choice(patterns)
    result = safe_eval(expr)
    return expr, result


def gen_four_operand(min_val=1, max_val=100) -> Tuple[str, int]:
    """Generate expressions with 4 operands."""
    nums = [random.randint(min_val, max_val) for _ in range(4)]
    ops = [random.choice(['+', '-']) for _ in range(3)]

    # Occasionally use multiplication
    if random.random() < 0.3:
        idx = random.randint(0, 2)
        ops[idx] = '*'
        nums[idx + 1] = random.randint(1, 12)

    patterns = [
        f"{nums[0]} {ops[0]} {nums[1]} {ops[1]} {nums[2]} {ops[2]} {nums[3]}",
        f"({nums[0]} {ops[0]} {nums[1]}) {ops[1]} {nums[2]} {ops[2]} {nums[3]}",
        f"({nums[0]} {ops[0]} {nums[1]}) {ops[1]} ({nums[2]} {ops[2]} {nums[3]})",
        f"{nums[0]} {ops[0]} ({nums[1]} {ops[1]} {nums[2]}) {ops[2]} {nums[3]}",
    ]
    expr = random.choice(patterns)
    result = safe_eval(expr)
    return expr, result


def make_hard_wrong(correct: int) -> int:
    """Generate a challenging incorrect result (close to correct)."""
    strategy = random.choices(
        ['off_by_1', 'off_by_small', 'off_by_medium', 'sign_error',
         'digit_swap', 'off_by_10', 'random_close'],
        weights=[25, 20, 15, 10, 10, 10, 10]
    )[0]

    if strategy == 'off_by_1':
        return correct + random.choice([-1, 1])
    elif strategy == 'off_by_small':
        return correct + random.choice([-3, -2, 2, 3])
    elif strategy == 'off_by_medium':
        return correct + random.randint(-10, 10)
    elif strategy == 'sign_error':
        return -correct if correct != 0 else random.randint(1, 20)
    elif strategy == 'digit_swap':
        s = str(abs(correct))
        if len(s) >= 2:
            s = s[-1] + s[1:-1] + s[0] if len(s) > 2 else s[::-1]
            result = int(s) * (1 if correct >= 0 else -1)
            return result if result != correct else correct + 1
        return correct + random.choice([-1, 1])
    elif strategy == 'off_by_10':
        return correct + random.choice([-10, 10])
    else:  # random_close
        return correct + random.randint(-20, 20)


def generate_sample() -> Optional[dict]:
    """Generate a single training sample."""
    # Choose complexity
    complexity = random.choices(
        ['simple', 'three', 'four'],
        weights=[0.35, 0.40, 0.25]
    )[0]

    if complexity == 'simple':
        expr, result = gen_simple_expr()
    elif complexity == 'three':
        expr, result = gen_three_operand()
    else:
        expr, result = gen_four_operand()

    if result is None:
        return None

    # Filter extreme values
    if abs(result) > 100000:
        return None

    # Positive or negative sample
    is_positive = random.random() < 0.5

    if is_positive:
        text = f"{expr} = {result}"
        label = 1
    else:
        wrong = make_hard_wrong(result)
        # Ensure wrong != correct
        while wrong == result:
            wrong = result + random.choice([-1, 1, -2, 2])
        text = f"{expr} = {wrong}"
        label = 0

    return {"text": text, "label": label}


def generate_dataset(num_samples: int):
    data = []
    attempts = 0
    while len(data) < num_samples and attempts < num_samples * 5:
        attempts += 1
        sample = generate_sample()
        if sample is not None:
            data.append(sample)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/data/fangda/tinyzero/prm/data")
    parser.add_argument("--train_size", type=int, default=300000)
    parser.add_argument("--test_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.train_size} training samples (integer-only, hard negatives)...")
    train_data = generate_dataset(args.train_size)
    train_path = os.path.join(args.output_dir, "train_v2.jsonl")
    with open(train_path, 'w') as f:
        for s in train_data:
            f.write(json.dumps(s) + '\n')
    print(f"Saved {len(train_data)} to {train_path}")

    print(f"Generating {args.test_size} test samples...")
    test_data = generate_dataset(args.test_size)
    test_path = os.path.join(args.output_dir, "test_v2.jsonl")
    with open(test_path, 'w') as f:
        for s in test_data:
            f.write(json.dumps(s) + '\n')
    print(f"Saved {len(test_data)} to {test_path}")

    # Stats
    train_pos = sum(1 for d in train_data if d['label'] == 1)
    test_pos = sum(1 for d in test_data if d['label'] == 1)
    print(f"\nTrain: {train_pos} pos / {len(train_data) - train_pos} neg")
    print(f"Test:  {test_pos} pos / {len(test_data) - test_pos} neg")

    has_mult = sum(1 for d in train_data if '*' in d['text'].split('=')[0])
    has_paren = sum(1 for d in train_data if '(' in d['text'])
    print(f"With multiplication: {has_mult}/{len(train_data)} ({100*has_mult/len(train_data):.1f}%)")
    print(f"With parentheses:    {has_paren}/{len(train_data)} ({100*has_paren/len(train_data):.1f}%)")

    print("\nExamples:")
    for s in train_data[:8]:
        print(f"  {s['text']:45s} label={s['label']}")


if __name__ == "__main__":
    main()
