"""
对抗测试 for countdown_v5c.

目标：在启动 PPO 训练前，验证 reward function 能拒绝已知的 hacking 模板。
所有 hacking 案例必须拿到 ≤ 0.1 分（基本为 0）。
所有合规答案必须拿到期望范围内的分数。

运行：
    python -m verl.utils.reward_score.test_countdown_v5c
"""

from verl.utils.reward_score.countdown_v5c import compute_score


def wrap(equation):
    """把纯等式包装成完整 solution_str 格式"""
    return f"User: xxx\nAssistant: Let me solve this.\n<think>working...</think>\n<answer> {equation} </answer>"


def run_test(name, equation, gt, expected_max=None, expected_min=None):
    score = compute_score(wrap(equation), gt)
    ok = True
    msg = ""
    if expected_max is not None and score > expected_max:
        ok = False
        msg = f"EXPECTED ≤ {expected_max}"
    if expected_min is not None and score < expected_min:
        ok = False
        msg = f"EXPECTED ≥ {expected_min}"
    status = "✓" if ok else "✗ FAIL"
    print(f"  [{status}] {name:<50} eq={equation!r:<40} score={score:.3f}  {msg}")
    return ok


def main():
    all_pass = True

    print("\n========== Hacking 模板（必须 ≤ 0.1 分）==========")
    # 场景：target=19, numbers=[33, 66, 38]
    gt1 = {'target': 19, 'numbers': [33, 66, 38]}
    hacks = [
        ("v5b 暴露的模板 (a-b)/c + target", "(33 - 38) / 66 + 19"),
        ("Target 直接塞入加法",             "33 + 19 - 33"),
        ("纯 target 输出",                   "19"),
        ("target + 0 恒等",                  "19 + 0"),
        ("target * 1 恒等",                  "19 * 1"),
        ("target + 归零凑数",                "19 + 33 - 33"),
        ("自除 + 凑接近 target",             "33 / 33 + 18"),
        ("复杂混合 hacking",                  "33 * 1 + 19 - 33"),
        ("完全无视给定数字",                  "100 - 81"),
        ("target 出现两次",                   "(19 + 19) / 2"),
    ]
    for name, eq in hacks:
        all_pass &= run_test(name, eq, gt1, expected_max=0.1)

    print("\n========== 合规探索（错但方向可接受）==========")
    # 用给定数字，但答错——reward 应该反映"接近度 + 合规 + 可能的 mul bonus"
    # 关键对比：合规 mul 尝试（0.45）应该 > 合规加减尝试（0.15），才能鼓励乘除法探索
    gt2 = {'target': 19, 'numbers': [33, 66, 38]}
    legitimate_wrong = [
        # 合规 mul 尝试：format(0.05) + number(0.1) + proximity(~0) + mul_bonus(0.3) = 0.45
        ("合规 mul 尝试（偏离远）",     "33 * 66 - 38",    {'max': 0.50, 'min': 0.35}),   # result=2140
        ("合规 mul 尝试（中等偏离）",    "66 / 33 + 38",    {'max': 0.50, 'min': 0.35}),   # result=40
        # 合规加减探索：format(0.05) + number(0.1) + proximity(~0) = 0.15
        ("合规加减 wrong（偏远）",      "33 + 66 - 38",    {'max': 0.25, 'min': 0.10}),   # result=61
        # 下一个：合规 mul 结果恰好接近 target（更直观的 target 近）
    ]
    # 另一个场景：target=20, numbers=[3, 7, 2] —— 可以验证 mul 探索拿到的分数
    gt_mul_easy = {'target': 20, 'numbers': [3, 7, 2]}
    legitimate_wrong.append(
        ("合规 mul 接近 target", "3 * 7 + 2",   {'max': 0.85, 'min': 0.60})  # result=23, 接近 20; 0.05+0.1+0.4*(17/20)+0.3=0.79
    )
    for name, eq, bounds in legitimate_wrong[:3]:  # 前 3 个用 gt2
        all_pass &= run_test(name, eq, gt2, expected_max=bounds['max'], expected_min=bounds['min'])
    # 最后一个用 gt_mul_easy
    name, eq, bounds = legitimate_wrong[3]
    all_pass &= run_test(name, eq, gt_mul_easy, expected_max=bounds['max'], expected_min=bounds['min'])

    print("\n========== 合规正解（必须拿到高分）==========")
    # 目标 52，给定 [16, 73, 18, 13]
    # 正解之一：73 - 13 - 18 + 16 + ... 太难算。换个例子
    # target=6, numbers=[3,2,1] → 3*2*1=6 or 3+2+1=6
    gt3 = {'target': 6, 'numbers': [3, 2, 1]}
    correct_answers = [
        # 正解使用乘法（应该拿满分：format + numbers + proximity + exact + complexity + mul_bonus）
        ("正解带乘法",    "3 * 2 * 1",  {'min': 1.30, 'max': 1.60}),   # 0.05+0.1+0.4+0.5+0.1+0.3 = 1.45
        # 正解用加法
        ("正解带加法",    "3 + 2 + 1",  {'min': 0.95, 'max': 1.20}),   # 0.05+0.1+0.4+0.5 = 1.05
    ]
    for name, eq, bounds in correct_answers:
        all_pass &= run_test(name, eq, gt3, expected_max=bounds['max'], expected_min=bounds['min'])

    print("\n========== 边界情况 ==========")
    gt4 = {'target': 6, 'numbers': [3, 2, 1]}
    edge_cases = [
        # 数字用错一个（部分合法）
        ("漏用一个给定数字", "3 * 2",  {'max': 0.50, 'min': 0.10}),   # result=6, exact but numbers_invalid
        # 没有 think 标签（ORM 风格）
        # 这个得分配 wrap 手工构造更严格测试
    ]
    for name, eq, bounds in edge_cases:
        all_pass &= run_test(name, eq, gt4, expected_max=bounds['max'], expected_min=bounds['min'])

    print("\n" + "=" * 60)
    if all_pass:
        print("✅ ALL ADVERSARIAL TESTS PASSED — v5c reward is safe to train.")
    else:
        print("❌ SOME TESTS FAILED — DO NOT train with this reward.")
    print("=" * 60)
    return all_pass


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
