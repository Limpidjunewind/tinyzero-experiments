# Day 6 — 实验报告写作

## 今天做了什么

所有实验跑完，把四组结果整理成了完整实验报告：`TinyZero 实验报告：小语言模型的强化学习推理能力.md`

---

## 实验结果汇总

| | PPO | GRPO |
|--|--|--|
| Qwen2-0.5B | score 卡在 0.10，无 `<think>` | — |
| Qwen2.5-1.5B | score 0.3–0.5，空 `<think>` | score 卡在 0.10，150 步未收敛 |
| Qwen2.5-3B | score 0.3–0.4，`<think>` 有真实内容，step ~250 entropy 坍塌 | — |

三个规模跨越三个门槛：
```
0.5B → 1.5B → 3B
无<think> → 空<think> → <think>有真实内容
```

---

## 今天真正搞懂的机制

**pg_loss 和 entropy 的因果顺序（之前搞反了）**

不是 entropy collapse 导致训练停止，因果链是：
```
step ~100：模型找到"空 think + format reward"捷径 → pg_loss 归零（策略停止更新）
step ~150：输出固定后 entropy 随之坍塌
step 150+：score 继续从 0.3 涨到 0.55，但不是在学新东西
           ← rollout 方差降低，原来就会的题开始稳定答对
```

**score 涨 ≠ 在学习**

entropy ≈ 0、pg_loss ≈ 0 之后 score 还在涨，是方差降低的假象，不是策略改进。entropy 和 pg_loss 是比 score 更诚实的诊断信号。

**Critic 没有过拟合（证伪了之前的假说）**

`score ≈ rewards` 只说明 KL 惩罚极小，不代表 advantage ≈ 0。实测 values ≈ 0、returns ≈ 0.3，advantage 始终有值，Critic 正常工作。

**3B 比 1.5B 更快坍塌，反直觉但合理**

能力越强 → 越快找到稳定策略 → PPO 越快强化它 → entropy 越快锁死。能力强在这里是更快失去探索性的代价，不是优势。

---

## 报告结构

讨论章节从"话题堆砌"重构成一条因果链，围绕一个核心问题：**为什么纯 RL 在小模型上会失败？**

```
第一层：模型容量决定探索能否起步
第二层：reward 设计决定模型学到什么
第三层：entropy collapse 决定训练何时终止
第四层：cold start 的必要性
```

每层回答上一层留下的问题，最后四条 takeaway 是整条链的自然收尾。

---

## 核心结论

**outcome reward 的有效性依赖模型的初始能力。** 模型需要在随机探索阶段就能偶尔答对，reward 信号才能起作用。小模型做不到，RL 就无法激励推理链，cold start 是必要条件，不是可选优化。
