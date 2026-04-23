"""
PPO 训练日志分析：思维涌现 + 正确率
对比 0.5B / 1.5B / 3B 在 countdown 任务上的表现

Reward 设计（来自 TinyZero_Local/verl/utils/reward_score/countdown.py）:
  - extract_solution: 只看 Assistant 之后的 **最后一行** 的 <answer> 标签
  - 三档打分:
    0   → "No equation found"    没有 <answer> 标签
    0.1 → "Invalid equation"     有标签但数字不合法（未用完/重复/捏造）
    0.1 → "Wrong result"         方程合法但计算结果 != target
    0.1 → "Could not evaluate"   方程无法 eval
    1.0 → "Correct equation"     完全正确
  - validate_equation: 提取方程中所有数字，排序后与 available_numbers 严格比对
  - evaluate_equation: 白名单 eval，只允许数字/运算符/括号

采样说明:
  - 日志以 1/64 概率随机打印每个样本（random.randint(1,64)==1）
  - 每隔 50 步有一次更大批量的打印（可能是 eval 阶段），贡献约 24% 样本
  - eval step 和普通 step 的正确率差异不大（<2%），不影响趋势分析
  - 总样本约 600-936 条，统计误差约 ±3%

注意:
  - 本脚本的 <answer> 提取搜索 assistant 全文输出，而 reward 函数只看最后一行
    实测差异约 8 条样本（<1%），不影响结论
  - result/reward 字段直接取自日志判定行，与 reward 函数一致，是最权威的真值
"""

import re
from pathlib import Path


# --------------- 解析 ---------------

def parse_log(filepath):
    """
    解析一个日志文件，返回 list[dict]。
    每个 dict 代表一个样本（由 Target 行开头）。
    """
    text = Path(filepath).read_text(encoding="utf-8")
    # 去掉时间戳前缀 (2026-03-19 06:34:23 )
    lines = []
    for raw in text.splitlines():
        m = re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (.*)", raw)
        lines.append(m.group(1) if m else raw)

    samples = []
    cur_step = 0

    i = 0
    while i < len(lines):
        # 跟踪 step
        sm = re.match(r"epoch \d+, step (\d+)", lines[i])
        if sm:
            cur_step = int(sm.group(1))
            i += 1
            continue

        # 样本起始：Target 行
        tm = re.match(r"Target: (\d+) \| Numbers: \[([^\]]+)\]", lines[i])
        if not tm:
            i += 1
            continue

        target = int(tm.group(1))
        nums = tm.group(2).split()
        num_count = len(nums)

        # 从 Target 行开始，收集到下一个 Target 行或 epoch 行
        block = [lines[i]]
        j = i + 1
        while j < len(lines):
            if re.match(r"Target: \d+ \|", lines[j]):
                break
            if re.match(r"epoch \d+, step", lines[j]):
                break
            if lines[j].strip() == "--------------------------------":
                break
            block.append(lines[j])
            j += 1

        sample = _parse_sample(block, cur_step, target, num_count)
        samples.append(sample)
        i = j
        continue

    return samples


def _parse_sample(block, step, target, num_count):
    """解析一个样本块"""
    text = "\n".join(block)

    # --- 提取 Assistant 输出 ---
    parts = text.split("Assistant:", 1)
    assistant_output = parts[1] if len(parts) > 1 else ""

    # --- 判定结果 + reward 映射 ---
    # 日志判定行直接对应 reward 函数的返回值
    if "Correct equation" in text:
        result = "correct"
        reward = 1.0
    elif "Wrong result" in text:
        result = "wrong"
        reward = 0.1
    elif "Invalid equation" in text:
        result = "invalid"
        reward = 0.1
    elif "No equation found" in text:
        result = "no_equation"
        reward = 0.0
    else:
        result = "unknown"
        reward = -1  # 不应出现

    # --- 分析 <think> ---
    think_matches = re.findall(r"<think>(.*?)</think>", assistant_output, re.DOTALL)
    has_think_pair = len(think_matches) > 0
    think_all = " ".join(t.strip() for t in think_matches).strip()
    think_empty = has_think_pair and len(think_all) == 0

    # --- 分析 <answer> ---
    answer_matches = re.findall(r"<answer>(.*?)</answer>", assistant_output, re.DOTALL)
    has_answer_pair = len(answer_matches) > 0
    answer_content = answer_matches[-1].strip() if answer_matches else ""  # 取最后一个，和 reward 函数一致

    # --- Think 质量 ---
    has_nl_reasoning = bool(re.search(
        r"\b(subtract|add|multiply|divide|get|gives?|then|first|next|take|plus|minus|result|equal|need)\b",
        think_all, re.IGNORECASE
    )) if think_all else False

    has_arithmetic = bool(re.search(r"\d+\s*[+\-*/]\s*\d+", think_all)) if think_all else False

    has_backtrack = bool(re.search(
        r"\b(try again|another|instead|wait|actually|however|not equal|doesn'?t|does not|hmm|let'?s try|wrong|incorrect|too)\b",
        think_all, re.IGNORECASE
    )) if think_all else False

    # --- Answer 内容分类（互斥，与 has_answer_pair 组合覆盖全部样本）---
    # has_answer_pair=False → 无标签
    # has_answer_pair=True  → 按内容分: 含方程 / 纯数字 / 空内容 / 其他
    answer_has_equation = bool(re.search(r"[+\-*/]", answer_content)) if answer_content else False
    answer_is_just_number = (
        bool(re.fullmatch(r"\d+(\.\d+)?", answer_content))
        if answer_content and not answer_has_equation
        else False
    )
    answer_is_empty = has_answer_pair and len(answer_content) == 0
    answer_is_other = (
        has_answer_pair
        and not answer_has_equation
        and not answer_is_just_number
        and not answer_is_empty
        and bool(answer_content)
    )

    # --- 输出行数（多行 = reward 函数可能找不到 answer）---
    output_lines = len(assistant_output.strip().split("\n"))

    return {
        "step": step,
        "target": target,
        "num_count": num_count,
        "result": result,
        "reward": reward,
        "has_think_pair": has_think_pair,
        "think_content": think_all,
        "think_empty": think_empty,
        "has_nl_reasoning": has_nl_reasoning,
        "has_arithmetic": has_arithmetic,
        "has_backtrack": has_backtrack,
        "has_answer_pair": has_answer_pair,
        "answer_content": answer_content,
        "answer_is_just_number": answer_is_just_number,
        "answer_has_equation": answer_has_equation,
        "answer_is_empty": answer_is_empty,
        "answer_is_other": answer_is_other,
        "output_lines": output_lines,
    }


# --------------- 统计 ---------------

def bin_metrics(samples, n_bins=10):
    """按 step 分段统计"""
    if not samples:
        return []
    steps = [s["step"] for s in samples]
    lo, hi = min(steps), max(steps)
    bin_size = max(1, (hi - lo + 1) // n_bins)

    results = []
    for b_start in range(lo, hi + 1, bin_size):
        b_end = b_start + bin_size
        bs = [s for s in samples if b_start <= s["step"] < b_end]
        if not bs:
            continue
        n = len(bs)

        # 基础
        correct = sum(1 for s in bs if s["result"] == "correct")
        wrong = sum(1 for s in bs if s["result"] == "wrong")
        invalid = sum(1 for s in bs if s["result"] == "invalid")
        no_eq = sum(1 for s in bs if s["result"] == "no_equation")

        # Reward
        avg_reward = sum(s["reward"] for s in bs if s["reward"] >= 0) / n

        # Think
        has_think = sum(1 for s in bs if s["has_think_pair"])
        think_nonempty = sum(1 for s in bs if s["has_think_pair"] and not s["think_empty"])
        think_nl = sum(1 for s in bs if s["has_nl_reasoning"])
        think_arith = sum(1 for s in bs if s["has_arithmetic"])
        think_back = sum(1 for s in bs if s["has_backtrack"])

        # Answer
        has_answer = sum(1 for s in bs if s["has_answer_pair"])
        ans_just_num = sum(1 for s in bs if s["answer_is_just_number"])
        ans_equation = sum(1 for s in bs if s["answer_has_equation"])

        # 输出行数
        avg_lines = sum(s["output_lines"] for s in bs) / n

        results.append({
            "range": f"{b_start}-{b_end - 1}",
            "n": n,
            "correct": correct / n,
            "wrong": wrong / n,
            "invalid": invalid / n,
            "no_eq": no_eq / n,
            "avg_reward": avg_reward,
            "think_rate": has_think / n,
            "think_nonempty": think_nonempty / n,
            "think_nl": think_nl / n,
            "think_arith": think_arith / n,
            "think_back": think_back / n,
            "answer_rate": has_answer / n,
            "ans_just_num": ans_just_num / n,
            "ans_equation": ans_equation / n,
            "avg_lines": avg_lines,
        })
    return results


# --------------- 输出 ---------------

def report(name, samples):
    n = len(samples)
    max_step = max(s["step"] for s in samples)

    print(f"\n{'=' * 70}")
    print(f"  {name}  |  样本: {n}  |  步数: {max_step}")
    print(f"{'=' * 70}")

    # 整体
    correct = sum(1 for s in samples if s["result"] == "correct")
    wrong = sum(1 for s in samples if s["result"] == "wrong")
    invalid = sum(1 for s in samples if s["result"] == "invalid")
    no_eq = sum(1 for s in samples if s["result"] == "no_equation")
    avg_reward = sum(s["reward"] for s in samples if s["reward"] >= 0) / n

    print(f"\n  [结果分布]  (平均 reward: {avg_reward:.3f})")
    print(f"    Correct  (r=1.0): {correct:>4} ({correct/n:.1%})")
    print(f"    Wrong    (r=0.1): {wrong:>4} ({wrong/n:.1%})")
    print(f"    Invalid  (r=0.1): {invalid:>4} ({invalid/n:.1%})")
    print(f"    No eq    (r=0.0): {no_eq:>4} ({no_eq/n:.1%})")

    # 3数字 vs 4数字
    s3 = [s for s in samples if s["num_count"] == 3]
    s4 = [s for s in samples if s["num_count"] == 4]
    c3 = sum(1 for s in s3 if s["result"] == "correct")
    c4 = sum(1 for s in s4 if s["result"] == "correct")
    print(f"\n  [按数字个数]")
    print(f"    3数字: {c3}/{len(s3)} = {c3/len(s3):.1%}" if s3 else "    3数字: 无")
    print(f"    4数字: {c4}/{len(s4)} = {c4/len(s4):.1%}" if s4 else "    4数字: 无")

    # 格式
    tp = sum(1 for s in samples if s["has_think_pair"])
    te = sum(1 for s in samples if s["think_empty"])
    tn = sum(1 for s in samples if s["has_think_pair"] and not s["think_empty"])
    ap = sum(1 for s in samples if s["has_answer_pair"])
    ajn = sum(1 for s in samples if s["answer_is_just_number"])
    aeq = sum(1 for s in samples if s["answer_has_equation"])
    aem = sum(1 for s in samples if s["answer_is_empty"])
    aot = sum(1 for s in samples if s["answer_is_other"])
    no_ap = n - ap

    print(f"\n  [格式分析]")
    print(f"    有<think>对:     {tp:>4} ({tp/n:.1%})")
    print(f"      - think非空:  {tn:>4} ({tn/n:.1%})")
    print(f"      - think为空:  {te:>4} ({te/n:.1%})")
    print(f"    有<answer>对:    {ap:>4} ({ap/n:.1%})")
    print(f"      - 含方程:     {aeq:>4} ({aeq/n:.1%})")
    print(f"      - 纯数字:     {ajn:>4} ({ajn/n:.1%})")
    print(f"      - 空内容:     {aem:>4} ({aem/n:.1%})")
    print(f"      - 其他:       {aot:>4} ({aot/n:.1%})  (非ASCII运算符/文本/不完整)")
    print(f"    无<answer>对:    {no_ap:>4} ({no_ap/n:.1%})")
    total_check = aeq + ajn + aem + aot + no_ap
    assert total_check == n, f"分类不完整: {aeq}+{ajn}+{aem}+{aot}+{no_ap}={total_check} != {n}"

    # Think 质量
    nl = sum(1 for s in samples if s["has_nl_reasoning"])
    ar = sum(1 for s in samples if s["has_arithmetic"])
    bk = sum(1 for s in samples if s["has_backtrack"])
    print(f"\n  [Think 质量]（占总样本比例）")
    print(f"    含自然语言推理:  {nl:>4} ({nl/n:.1%})")
    print(f"    含算术表达式:    {ar:>4} ({ar/n:.1%})")
    print(f"    含回溯/自我纠正: {bk:>4} ({bk/n:.1%})")

    # 按阶段
    bins = bin_metrics(samples, n_bins=10)
    print(f"\n  [按训练阶段]")
    print(f"    {'步数':>10}  {'N':>4}  {'Reward':>7}  {'正确':>6}  {'错答':>6}  {'无效':>6}  {'无eq':>5}  |  {'Think非空':>9}  {'NL推理':>7}  {'算术':>6}  |  {'Ans方程':>7}  {'Ans数字':>7}  {'行数':>5}")
    print(f"    {'-' * 115}")
    for b in bins:
        print(f"    {b['range']:>10}  {b['n']:>4}"
              f"  {b['avg_reward']:>7.3f}"
              f"  {b['correct']:>5.0%}  {b['wrong']:>5.0%}  {b['invalid']:>5.0%}  {b['no_eq']:>4.0%}"
              f"  |  {b['think_nonempty']:>8.0%}  {b['think_nl']:>6.0%}  {b['think_arith']:>5.0%}"
              f"  |  {b['ans_equation']:>6.0%}  {b['ans_just_num']:>6.0%}  {b['avg_lines']:>5.1f}")


def cross_compare(all_data):
    print(f"\n\n{'=' * 70}")
    print(f"  跨模型对比")
    print(f"{'=' * 70}")

    print(f"\n  {'模型':<12} {'样本':>5} {'Reward':>7} {'正确':>6} {'3数字':>6} {'4数字':>6}  |  {'Think非空':>9} {'Ans方程':>7} {'NL推理':>7}")
    print(f"  {'-' * 80}")

    for name, samples in all_data.items():
        n = len(samples)
        correct = sum(1 for s in samples if s["result"] == "correct")
        avg_r = sum(s["reward"] for s in samples if s["reward"] >= 0) / n
        s3 = [s for s in samples if s["num_count"] == 3]
        s4 = [s for s in samples if s["num_count"] == 4]
        c3 = sum(1 for s in s3 if s["result"] == "correct")
        c4 = sum(1 for s in s4 if s["result"] == "correct")

        tn = sum(1 for s in samples if s["has_think_pair"] and not s["think_empty"])
        aeq = sum(1 for s in samples if s["answer_has_equation"])
        nl = sum(1 for s in samples if s["has_nl_reasoning"])

        r3 = f"{c3/len(s3):.0%}" if s3 else "-"
        r4 = f"{c4/len(s4):.0%}" if s4 else "-"
        print(f"  {name:<12} {n:>5} {avg_r:>7.3f} {correct/n:>5.0%} {r3:>6} {r4:>6}"
              f"  |  {tn/n:>8.0%} {aeq/n:>6.0%} {nl/n:>6.0%}")

    # 后期对比（最后20%步数）
    print(f"\n  [后期 (最后20%步数)]")
    print(f"  {'模型':<12} {'Reward':>7} {'正确':>6} {'Think非空':>9} {'NL推理':>7} {'Ans方程':>7}")
    print(f"  {'-' * 55}")
    for name, samples in all_data.items():
        max_step = max(s["step"] for s in samples)
        cutoff = max_step * 0.8
        late = [s for s in samples if s["step"] >= cutoff]
        if not late:
            continue
        nl = len(late)
        correct = sum(1 for s in late if s["result"] == "correct")
        avg_r = sum(s["reward"] for s in late if s["reward"] >= 0) / nl
        tn = sum(1 for s in late if s["has_think_pair"] and not s["think_empty"])
        nlr = sum(1 for s in late if s["has_nl_reasoning"])
        aeq = sum(1 for s in late if s["answer_has_equation"])
        print(f"  {name:<12} {avg_r:>7.3f} {correct/nl:>5.0%} {tn/nl:>8.0%} {nlr/nl:>6.0%} {aeq/nl:>6.0%}")

    # === 0.5B 失败原因深挖 ===
    print(f"\n  [0.5B 失败原因分析]")
    if "PPO 0.5B" in all_data:
        s05 = all_data["PPO 0.5B"]
        n = len(s05)

        # 互斥分类：每个样本恰好属于一类
        no_ans = sum(1 for s in s05 if not s["has_answer_pair"])
        ans_empty = sum(1 for s in s05 if s["answer_is_empty"])
        ans_num = sum(1 for s in s05 if s["answer_is_just_number"])
        ans_eq_invalid = sum(1 for s in s05 if s["answer_has_equation"] and s["result"] == "invalid")
        ans_eq_wrong = sum(1 for s in s05 if s["answer_has_equation"] and s["result"] == "wrong")
        ans_eq_correct = sum(1 for s in s05 if s["answer_has_equation"] and s["result"] == "correct")
        ans_eq_other_result = sum(1 for s in s05 if s["answer_has_equation"] and s["result"] not in ("invalid", "wrong", "correct"))
        ans_other = sum(1 for s in s05 if s["answer_is_other"])

        total_check = no_ans + ans_empty + ans_num + ans_eq_invalid + ans_eq_wrong + ans_eq_correct + ans_eq_other_result + ans_other
        print(f"    ┌─ 无<answer>标签:        {no_ans:>4} ({no_ans/n:.1%})  → reward=0")
        print(f"    ├─ <answer>空内容:        {ans_empty:>4} ({ans_empty/n:.1%})  → reward=0.1 (Invalid)")
        print(f"    ├─ answer是纯数字:        {ans_num:>4} ({ans_num/n:.1%})  → reward=0.1 (Invalid)")
        print(f"    ├─ answer含方程但Invalid:  {ans_eq_invalid:>4} ({ans_eq_invalid/n:.1%})  → reward=0.1 (用错数字)")
        print(f"    ├─ answer含方程且Wrong:    {ans_eq_wrong:>4} ({ans_eq_wrong/n:.1%})  → reward=0.1 (数字对但算错)")
        print(f"    ├─ answer含方程且Correct:  {ans_eq_correct:>4} ({ans_eq_correct/n:.1%})  → reward=1.0")
        print(f"    ├─ answer含方程其他:       {ans_eq_other_result:>4} ({ans_eq_other_result/n:.1%})  → reward函数未判定")
        print(f"    └─ answer其他内容:         {ans_other:>4} ({ans_other/n:.1%})  → (非ASCII运算符÷/文本等)")
        print(f"    合计: {total_check} / {n}  {'✓' if total_check == n else '✗ 分类有遗漏!'}")


if __name__ == "__main__":
    files = {
        "PPO 0.5B": "logs/ppo0.5B.txt",
        "PPO 1.5B": "logs/ppo1.5B.txt",
        "PPO 3B":   "logs/ppo3B.txt",
    }

    all_data = {}
    for name, path in files.items():
        samples = parse_log(path)
        all_data[name] = samples
        report(name, samples)

    cross_compare(all_data)
