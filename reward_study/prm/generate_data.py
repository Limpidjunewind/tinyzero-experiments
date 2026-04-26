"""
Data generation script for PRM training.

Generates arithmetic expression claims with correct/incorrect labels.
Output format: JSONL with {"text": "expr = result", "label": 0 or 1}

Usage:
    python generate_data.py --output_dir prm/data --train_size 100000 --test_size 10000
"""

import json
import random
import os
import argparse
from typing import Tuple


def generate_expression(num_operands: int, min_val: int = 1, max_val: int = 100) -> Tuple[str, float]:
    """Generate a random arithmetic expression and its correct result.

    Args:
        num_operands: number of operands (2-5)
        min_val: minimum operand value
        max_val: maximum operand value

    Returns:
        (expression_string, correct_result)
    """
    operators = ['+', '-', '*', '/']

    # Generate operands
    nums = [random.randint(min_val, max_val) for _ in range(num_operands)]

    # Build expression
    expr_parts = [str(nums[0])]
    for i in range(1, num_operands):
        op = random.choice(operators)

        # Avoid division by zero and keep results reasonable
        if op == '/':
            # Make sure divisor divides evenly for cleaner data
            if nums[i] == 0:
                nums[i] = 1
            # Simplify: just use the number as-is, might not divide evenly
        elif op == '*':
            # Keep multiplication operands smaller to avoid huge numbers
            nums[i] = random.randint(1, min(20, max_val))

        expr_parts.append(op)
        expr_parts.append(str(nums[i]))

    expr_str = ' '.join(expr_parts)

    # Evaluate
    try:
        result = eval(expr_str)
        # Round to avoid floating point display issues
        if isinstance(result, float):
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            else:
                result = round(result, 2)
    except:
        return None, None

    return expr_str, result


def generate_expression_with_parens(num_operands: int, min_val: int = 1, max_val: int = 100) -> Tuple[str, float]:
    """Generate expression with parentheses for more complexity."""
    if num_operands < 3:
        return generate_expression(num_operands, min_val, max_val)

    operators = ['+', '-', '*']
    nums = [random.randint(min_val, max_val) for _ in range(num_operands)]

    # Randomly decide parentheses grouping
    # e.g., (a + b) - c  or  a * (b + c)
    op1 = random.choice(operators)
    op2 = random.choice(operators)

    if num_operands == 3:
        patterns = [
            f"({nums[0]} {op1} {nums[1]}) {op2} {nums[2]}",
            f"{nums[0]} {op1} ({nums[1]} {op2} {nums[2]})",
        ]
    elif num_operands == 4:
        op3 = random.choice(operators)
        patterns = [
            f"({nums[0]} {op1} {nums[1]}) {op2} ({nums[2]} {op3} {nums[3]})",
            f"(({nums[0]} {op1} {nums[1]}) {op2} {nums[2]}) {op3} {nums[3]}",
            f"({nums[0]} {op1} {nums[1]} {op2} {nums[2]}) {op3} {nums[3]}",
        ]
    else:
        # Fallback to simple expression for 5+ operands
        return generate_expression(num_operands, min_val, max_val)

    expr_str = random.choice(patterns)

    try:
        result = eval(expr_str)
        if isinstance(result, float):
            if abs(result - round(result)) < 1e-9:
                result = int(round(result))
            else:
                result = round(result, 2)
    except:
        return None, None

    return expr_str, result


def make_wrong_result(correct_result: float) -> float:
    """Generate a plausible but incorrect result.

    Uses various perturbation strategies to create realistic errors.
    """
    strategies = [
        # Small perturbation (off by 1-5)
        lambda r: r + random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]),
        # Medium perturbation (off by 5-20)
        lambda r: r + random.randint(-20, 20),
        # Sign flip
        lambda r: -r if r != 0 else random.randint(1, 50),
        # Digit swap (e.g., 45 -> 54)
        lambda r: int(str(abs(int(r)))[::-1]) if abs(r) >= 10 and isinstance(r, (int, float)) and r == int(r) else r + random.randint(1, 10),
        # Off by factor of 10
        lambda r: r * 10 if abs(r) < 100 else r // 10,
        # Random number in similar range
        lambda r: random.randint(int(r) - 50, int(r) + 50) if isinstance(r, (int, float)) else r + 1,
    ]

    strategy = random.choice(strategies)
    wrong = strategy(correct_result)

    # Make sure it's actually wrong
    if isinstance(correct_result, float):
        if abs(wrong - correct_result) < 1e-5:
            wrong = correct_result + random.choice([-3, -2, -1, 1, 2, 3])
    else:
        if wrong == correct_result:
            wrong = correct_result + random.choice([-3, -2, -1, 1, 2, 3])

    # Format nicely
    if isinstance(wrong, float) and abs(wrong - round(wrong)) < 1e-9:
        wrong = int(round(wrong))

    return wrong


def generate_sample(use_parens_prob=0.3):
    """Generate a single training sample.

    Returns:
        dict with 'text' and 'label' keys, or None if generation failed
    """
    num_operands = random.choices([2, 3, 4], weights=[0.3, 0.5, 0.2])[0]

    if random.random() < use_parens_prob and num_operands >= 3:
        expr_str, result = generate_expression_with_parens(num_operands)
    else:
        expr_str, result = generate_expression(num_operands)

    if expr_str is None or result is None:
        return None

    # Filter out overly large or complex results
    if isinstance(result, float) and (abs(result) > 100000 or result != result):  # NaN check
        return None
    if isinstance(result, int) and abs(result) > 100000:
        return None

    # Decide if this is a positive or negative sample
    is_positive = random.random() < 0.5

    if is_positive:
        text = f"{expr_str} = {result}"
        label = 1
    else:
        wrong = make_wrong_result(result)
        text = f"{expr_str} = {wrong}"
        label = 0

    return {"text": text, "label": label}


def generate_dataset(num_samples, use_parens_prob=0.3):
    """Generate a dataset of arithmetic claims."""
    data = []
    attempts = 0
    max_attempts = num_samples * 3

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        sample = generate_sample(use_parens_prob)
        if sample is not None:
            data.append(sample)

    return data


def main():
    parser = argparse.ArgumentParser(description="Generate PRM training data")
    parser.add_argument("--output_dir", type=str, default="/data/fangda/tinyzero/prm/data")
    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--test_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate training data
    print(f"Generating {args.train_size} training samples...")
    train_data = generate_dataset(args.train_size)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    with open(train_path, 'w') as f:
        for sample in train_data:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved {len(train_data)} training samples to {train_path}")

    # Generate test data
    print(f"Generating {args.test_size} test samples...")
    test_data = generate_dataset(args.test_size)
    test_path = os.path.join(args.output_dir, "test.jsonl")
    with open(test_path, 'w') as f:
        for sample in test_data:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved {len(test_data)} test samples to {test_path}")

    # Print stats
    train_pos = sum(1 for d in train_data if d['label'] == 1)
    test_pos = sum(1 for d in test_data if d['label'] == 1)
    print(f"\nTrain: {train_pos} positive, {len(train_data) - train_pos} negative")
    print(f"Test:  {test_pos} positive, {len(test_data) - test_pos} negative")

    # Print examples
    print("\nExamples:")
    for sample in train_data[:6]:
        print(f"  {sample['text']:40s}  label={sample['label']}")


if __name__ == "__main__":
    main()
