"""
Reward function for v2-no-proximity ablation experiment.

Causal ablation of the proximity score component in v2's reward.

Hypothesis being tested (C3 from report_0413/experiment_report.md):
    The continuous proximity score in v2 created an "addition/subtraction
    highway" that suppressed multiplication/division exploration, dropping
    the mul/div attempt rate from v1's 4.4% to v2's 1.5%.

This file is byte-identical to countdown_v2.py EXCEPT that the proximity
score component is set to 0 (line 5/section 5 below). All other reward
components — format, number usage, exact match bonus, complexity bonus,
think-answer consistency, self-verification — are unchanged.

If C3 is causally correct, removing the proximity score should restore
mul/div attempt rate toward v1's level (~4.4%), at the cost of overall
accuracy (since proximity also drove the +26pp accuracy gain v1 → v2).

Score range: 0.0 ~ 0.8 (vs v2's 0.0 ~ 1.2)
"""

import re
import random


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def extract_think(solution_str):
    """Extract the thinking process from the solution string."""
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        return numbers_in_eq == available_numbers
    except:
        return False


def count_number_overlap(equation_str, available_numbers):
    """Count how many available numbers are correctly used in the equation."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        available_sorted = sorted(available_numbers)
        used_sorted = sorted(numbers_in_eq)

        i, j, matched = 0, 0, 0
        while i < len(available_sorted) and j < len(used_sorted):
            if available_sorted[i] == used_sorted[j]:
                matched += 1
                i += 1
                j += 1
            elif available_sorted[i] < used_sorted[j]:
                i += 1
            else:
                j += 1
        return matched, len(available_numbers)
    except:
        return 0, len(available_numbers)


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation."""
    try:
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            return None
        result = eval(equation_str, {"__builtins__": None}, {})
        if not isinstance(result, (int, float)):
            return None
        return result
    except:
        return None


def has_complex_ops(equation_str):
    """Check if equation uses multiplication or division."""
    return bool(re.search(r'[*/]', equation_str))


def check_think_answer_consistency(think_str, answer_str):
    """Check if the final expression in <think> matches <answer>."""
    if think_str is None or answer_str is None:
        return False
    think_nums = re.findall(r'\d+', think_str.split('\n')[-1] if think_str else "")
    answer_nums = re.findall(r'\d+', answer_str)
    if think_nums and answer_nums:
        return think_nums == answer_nums
    return False


def check_self_verification(think_str):
    """Check if the thinking process contains self-verification patterns."""
    if think_str is None:
        return False
    verification_patterns = [
        r'let me check',
        r'let me verify',
        r'verify:',
        r'check:',
        r'= \d+,?\s*(which is|that\'s|correct|right|equals)',
        r'this (gives|equals|is)\s+\d+',
    ]
    for pattern in verification_patterns:
        if re.search(pattern, think_str, re.IGNORECASE):
            return True
    return False


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """
    v2-no-proximity reward — identical to v2 except proximity_score = 0.

    Score breakdown:
    - Format score (has <think> + <answer>):           0.05
    - Number usage score (correct numbers used):       0.05
    - Proximity score (how close to target):           0.0  (REMOVED — was 0~0.4 in v2)
    - Exact match bonus:                               +0.5
    - Operation complexity bonus (* or / used):        +0.1  (only if correct)
    - Think-answer consistency bonus:                  +0.05 (only if correct)
    - Self-verification bonus:                         +0.05 (only if correct)

    Total range: 0.0 ~ 0.8 (vs v2's 0.0 ~ 1.2)
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']

    do_print = random.randint(1, 64) == 1

    equation = extract_solution(solution_str=solution_str)
    think_str = extract_think(solution_str)

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")

    if equation is None:
        if do_print:
            print(f"No equation found → score=0")
        return 0

    total_score = 0.0
    has_think = think_str is not None and len(think_str) > 0
    has_answer = equation is not None
    if has_think and has_answer:
        total_score += 0.05

    numbers_valid = validate_equation(equation, numbers)
    if numbers_valid:
        total_score += 0.05
    else:
        matched, total = count_number_overlap(equation, numbers)
        if total > 0:
            total_score += 0.05 * (matched / total) * 0.5

    result = evaluate_equation(equation)

    if result is None:
        if do_print:
            print(f"Could not evaluate → score={total_score:.3f}")
        return total_score

    # --- ABLATED: Proximity score (was 0~0.4 in v2; set to 0 here) ---
    # In v2:
    #   if target != 0:
    #       relative_error = abs(result - target) / abs(target)
    #   else:
    #       relative_error = abs(result - target)
    #   proximity = max(0.0, 1.0 - relative_error)
    #   proximity_score = 0.4 * proximity
    #   total_score += proximity_score
    # In v2-no-proximity: proximity contribution removed entirely.
    # We still compute relative_error below for diagnostic logging only.
    if target != 0:
        relative_error = abs(result - target) / abs(target)
    else:
        relative_error = abs(result - target)
    proximity = max(0.0, 1.0 - relative_error)  # logged but not added to score

    is_correct = abs(result - target) < 1e-5
    if is_correct and numbers_valid:
        total_score += 0.5

        if has_complex_ops(equation):
            total_score += 0.1
            if do_print:
                print(f"Complexity bonus: used */")

        if check_think_answer_consistency(think_str, equation):
            total_score += 0.05

        if check_self_verification(think_str):
            total_score += 0.05
            if do_print:
                print(f"Self-verification detected")

    if do_print:
        status = "CORRECT" if (is_correct and numbers_valid) else "WRONG"
        print(f"{status}: result={result}, target={target}, "
              f"proximity={proximity:.3f} (NOT scored), total_score={total_score:.3f}")

    return total_score
