"""
Reward function v5b: Structural incentive for multiplication/division,
discounted by number-usage legitimacy.

Based on v2, with a new "meaningful multiplication/division bonus" that
is awarded REGARDLESS of correctness BUT scaled by how many of the
given numbers appear in the equation. This addresses two problems:

1. v2's +0.1 complexity bonus (only on correct answers) provides no
   signal during exploration of multiplicative paths. v5b gives a
   partial signal even to wrong answers, overcoming v2's "addition/
   subtraction highway" effect.

2. An unscaled +0.5 bonus creates a severe reward-hacking vector:
   the model could ignore the given numbers entirely and output any
   multiplication whose product happens to be near the target (e.g.
   "5 * 10 = 50" for target=50 with numbers=[3,7,9]). The legitimacy
   discount (matched_numbers / total_numbers) removes this hack —
   zero numbers matched = zero bonus.

Anti-hacking rules (two layers):
  L1 — Syntactic: the multiplication/division must actually transform
       the value. `x * 1`, `x * 0`, `x / 1`, `x / x` are excluded.
  L2 — Semantic: the bonus is multiplied by (matched_given_numbers /
       total_given_numbers). Ignoring the problem gets 0 × 0.5 = 0.

Effective bonus range: 0.0 (no given numbers used) to 0.5 (all given
numbers used correctly). Partial credit is linear.

Changes vs v2:
- NEW: meaningful_mul_bonus = 0.5 × (matched / total), awarded to any
  answer that (a) contains at least one syntactically-meaningful * or /,
  and (b) uses at least some of the given numbers.
- All other dimensions unchanged (number usage weight stays at v2's 0.05).
"""

import re
import random


MUL_WEIGHT = 0.5


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
    """Check if equation uses multiplication or division (any form)."""
    return bool(re.search(r'[*/]', equation_str))


def has_meaningful_complex_ops(equation_str):
    """
    Check if equation uses at least one MEANINGFUL multiplication or division.

    An operation is meaningful if it actually transforms the value:
    - operand is not 0 or 1
    - division does not have identical digit operands (x / x)
    - subexpressions in parens count as non-trivial

    Returns True if at least one meaningful * or / is present.
    """
    if not re.search(r'[*/]', equation_str):
        return False

    # Match each multiplication/division with its direct left and right operands.
    # Operands can be: a number, a closing paren (subexpr on left),
    # or an opening paren (subexpr on right).
    pattern = r'(\d+|\))\s*([*/])\s*(\d+|\()'
    matches = re.findall(pattern, equation_str)

    if not matches:
        return False

    for left, op, right in matches:
        left_trivial = left in ('0', '1')
        right_trivial = right in ('0', '1')

        if left_trivial or right_trivial:
            continue

        # Self-divide (x / x with same literal digits) is trivial
        if op == '/' and left.isdigit() and right.isdigit() and left == right:
            continue

        return True

    return False


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
    Multi-granularity reward for countdown task, with structural incentive
    for meaningful multiplication/division use.

    Score breakdown:
    - Format score (has <think> + <answer>):                        0.05
    - Number usage score (correct numbers used):                    0.05
    - Proximity score (how close to target):                        0.0 ~ 0.4
    - Meaningful mul/div bonus × (matched/total):                   0.0 ~ 0.5  (NEW in v5b)
    - Exact match bonus:                                            +0.5
    - Operation complexity bonus (* or / used):                     +0.1  (only if correct)
    - Think-answer consistency bonus:                               +0.05 (only if correct)
    - Self-verification bonus:                                      +0.05 (only if correct)

    Total range: 0.0 ~ 1.7
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']

    do_print = random.randint(1, 64) == 1

    # --- Extract components ---
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

    # --- Number usage score (v2 weight: 0.05) ---
    numbers_valid = validate_equation(equation, numbers)
    num_matched, num_total = count_number_overlap(equation, numbers)
    num_legitimacy = (num_matched / num_total) if num_total > 0 else 0.0

    if numbers_valid:
        total_score += 0.05
    else:
        if num_total > 0:
            total_score += 0.05 * num_legitimacy * 0.5

    # --- NEW in v5b: Meaningful multiplication/division bonus ---
    # Awarded to any answer (correct or wrong) that uses * or / meaningfully,
    # scaled by number-usage legitimacy. This gives exploration a positive
    # signal even when the numeric result is far from target, while preventing
    # the reward-hacking strategy of ignoring given numbers entirely.
    uses_meaningful_mul = has_meaningful_complex_ops(equation)
    if uses_meaningful_mul:
        mul_bonus = MUL_WEIGHT * (1.0 if numbers_valid else num_legitimacy)
        total_score += mul_bonus

    # --- Evaluate equation ---
    result = evaluate_equation(equation)

    if result is None:
        if do_print:
            print(f"Could not evaluate → score={total_score:.3f}")
        return total_score

    # --- Proximity score (0 ~ 0.4) ---
    if target != 0:
        relative_error = abs(result - target) / abs(target)
    else:
        relative_error = abs(result - target)

    proximity = max(0.0, 1.0 - relative_error)
    proximity_score = 0.4 * proximity
    total_score += proximity_score

    # --- Exact match bonus ---
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
        if uses_meaningful_mul:
            effective_mul = MUL_WEIGHT * (1.0 if numbers_valid else num_legitimacy)
            mul_tag = f" [MUL+{effective_mul:.2f}]"
        else:
            mul_tag = ""
        print(f"{status}{mul_tag}: result={result}, target={target}, "
              f"proximity={proximity:.3f}, total_score={total_score:.3f}")

    return total_score
