"""
Reward function v5c: 乘除法激励的修正版，防 reward hacking。

动机
----
v5b 试图通过"对错都给 meaningful mul bonus"来鼓励乘除法探索，但模型发现了一个
reward hacking 模板：(given_a - given_b) / given_c + TARGET —— 把 target 值直接
塞入等式，套上除法模板，就能在每个样本上拿到 ~0.925 的分数，但真实 exact-match
正确率为 0%。

v5c 的设计原则
--------------
1. 硬约束优先，软激励其次：违反硬约束的答案直接 0 分
2. 检查负空间：不仅检查"是否用了给定数字"，更要检查"是否用了题外数字"
3. mul bonus 必须锚定合规前提：numbers_valid=True 才给
4. 整数精确匹配：避免浮点近似刷分
5. 上线前对抗测试：10 个 hacking 模板全部 ≤ 0.1 分才能启动

关键改动 vs v5b
---------------
- 硬约束：equation 中出现非给定数字（extra_numbers ≠ ∅）→ 直接 0 分
- mul bonus 前提：必须 numbers_valid=True（与 v2 的 complexity bonus 一致）
- mul bonus 权重：0.5 → 0.3（更保守，留出 hacking 缓冲）
- Exact match：|result - target| < 1e-5 → 严格整数相等（round(result) == target）

得分结构
--------
硬约束通过后：
  Format                        0.05
  Number usage 正确             0.1   (提高权重，参考 v4 实验)
  Proximity                     0 ~ 0.4
  Exact match                   +0.5  (严格整数)
  Complexity bonus (仅正确时)   +0.1
  Meaningful mul bonus           +0.3  (仅 numbers_valid 时)
  Self-verification (仅正确时)  +0.05

总分范围：0.0 ~ 1.6
"""

import re
import random


# ---------- 基础工具（与 v2/v5b 一致）----------

def extract_solution(solution_str):
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
        return matches[-1].group(1).strip()
    return None


def extract_think(solution_str):
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def evaluate_equation(equation_str):
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
    """是否包含 * 或 /（不区分是否有意义）"""
    return bool(re.search(r'[*/]', equation_str))


def has_meaningful_complex_ops(equation_str):
    """语法层过滤：是否包含有意义的乘除法（两侧操作数都 ≥ 2，且除法不是 x/x）"""
    if not re.search(r'[*/]', equation_str):
        return False

    pattern = r'(\d+|\))\s*([*/])\s*(\d+|\()'
    matches = re.findall(pattern, equation_str)
    if not matches:
        return False

    for left, op, right in matches:
        left_trivial = left in ('0', '1')
        right_trivial = right in ('0', '1')
        if left_trivial or right_trivial:
            continue
        if op == '/' and left.isdigit() and right.isdigit() and left == right:
            continue
        return True
    return False


def check_think_answer_consistency(think_str, answer_str):
    if think_str is None or answer_str is None:
        return False
    think_nums = re.findall(r'\d+', think_str.split('\n')[-1] if think_str else "")
    answer_nums = re.findall(r'\d+', answer_str)
    if think_nums and answer_nums:
        return think_nums == answer_nums
    return False


def check_self_verification(think_str):
    if think_str is None:
        return False
    verification_patterns = [
        r'let me check', r'let me verify', r'verify:', r'check:',
        r'= \d+,?\s*(which is|that\'s|correct|right|equals)',
        r'this (gives|equals|is)\s+\d+',
    ]
    for pattern in verification_patterns:
        if re.search(pattern, think_str, re.IGNORECASE):
            return True
    return False


# ---------- v5c 新增：硬约束 & multiset 级别的 numbers_valid ----------

def get_numbers_in_eq(equation_str):
    """提取等式中所有数字（作为整数列表）"""
    try:
        return [int(n) for n in re.findall(r'\d+', equation_str)]
    except:
        return []


def find_extra_numbers(numbers_in_eq, available_numbers):
    """
    返回等式中出现的"非给定数字"（考虑多重集合）。
    比如 given=[3,7,9]，equation 里有 [3,7,9,19] → extra = [19]
    比如 given=[3,7,9]，equation 里有 [3,3,7] → extra = [3]（3 用了两次，多出一次）
    """
    available_counter = {}
    for n in available_numbers:
        available_counter[n] = available_counter.get(n, 0) + 1
    extra = []
    for n in numbers_in_eq:
        if available_counter.get(n, 0) > 0:
            available_counter[n] -= 1
        else:
            extra.append(n)
    return extra


def count_number_overlap(equation_str, available_numbers):
    """
    多重集合层面统计匹配数（保留原 v2 行为）。
    """
    numbers_in_eq = get_numbers_in_eq(equation_str)
    if not numbers_in_eq:
        return 0, len(available_numbers)
    available_counter = {}
    for n in available_numbers:
        available_counter[n] = available_counter.get(n, 0) + 1
    matched = 0
    for n in numbers_in_eq:
        if available_counter.get(n, 0) > 0:
            available_counter[n] -= 1
            matched += 1
    return matched, len(available_numbers)


def validate_equation(equation_str, available_numbers):
    """是否严格等于给定数字集合（每个数字恰好用一次）"""
    try:
        numbers_in_eq = sorted(get_numbers_in_eq(equation_str))
        return numbers_in_eq == sorted(available_numbers)
    except:
        return False


# ---------- 主评分函数 ----------

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """
    v5c reward: 硬约束 + 软激励两层设计。

    Layer 1 (硬约束，违反 → 0 分):
      - 等式必须能解析和求值
      - 等式中不能出现"题外数字"（给定数字以外的任何整数）

    Layer 2 (软激励):
      Format / Number usage / Proximity / Exact match / Mul bonus / Verification
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

    # -------- 硬约束 Layer 1 --------
    if equation is None:
        if do_print:
            print(f"No equation found → 0")
        return 0

    numbers_in_eq = get_numbers_in_eq(equation)
    if not numbers_in_eq:
        if do_print:
            print(f"No numbers in equation → 0")
        return 0

    extra_numbers = find_extra_numbers(numbers_in_eq, numbers)
    if extra_numbers:
        # 硬约束违反：用了题外数字
        if do_print:
            print(f"HARD CONSTRAINT VIOLATION: extra numbers {extra_numbers} → 0")
        return 0

    result = evaluate_equation(equation)
    if result is None:
        if do_print:
            print(f"Could not evaluate → 0")
        return 0

    # -------- 软激励 Layer 2 --------
    total_score = 0.0

    # 1. Format
    has_think = think_str is not None and len(think_str) > 0
    if has_think:
        total_score += 0.05

    # 2. Number usage（v4 实验证明更高权重有效，v5c 采用 0.1）
    numbers_valid = validate_equation(equation, numbers)
    if numbers_valid:
        total_score += 0.1
    else:
        # 部分使用（给定数字没用全）
        matched, total = count_number_overlap(equation, numbers)
        if total > 0:
            total_score += 0.1 * (matched / total) * 0.5

    # 3. Proximity（连续接近度）
    if target != 0:
        relative_error = abs(result - target) / abs(target)
    else:
        relative_error = abs(result - target)
    proximity = max(0.0, 1.0 - relative_error)
    total_score += 0.4 * proximity

    # 4. Exact match（严格整数匹配，避免浮点刷分）
    is_exact_integer_match = False
    try:
        if abs(result - round(result)) < 1e-9 and int(round(result)) == target:
            is_exact_integer_match = True
    except:
        pass

    if is_exact_integer_match and numbers_valid:
        total_score += 0.5

        if has_complex_ops(equation):
            total_score += 0.1  # v2 的原 complexity bonus
            if do_print:
                print(f"Complexity bonus: used */")

        if check_self_verification(think_str):
            total_score += 0.05
            if do_print:
                print(f"Self-verification detected")

    # 5. Meaningful mul bonus（关键锚定：numbers_valid=True 才给）
    uses_meaningful_mul = has_meaningful_complex_ops(equation)
    if uses_meaningful_mul and numbers_valid:
        total_score += 0.3
        if do_print:
            print(f"Meaningful mul bonus: +0.3 (numbers_valid=True)")

    if do_print:
        status = "CORRECT" if is_exact_integer_match and numbers_valid else "WRONG"
        print(f"{status}: result={result}, target={target}, "
              f"proximity={proximity:.3f}, total_score={total_score:.3f}")

    return total_score
