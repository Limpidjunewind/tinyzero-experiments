# Day 3-2 — PPO 训练主循环

## 核心文件
`verl/trainer/ppo/ray_trainer.py` — `fit()` 函数（第547行）

---

## fit() 主循环骨架

两层循环：`for epoch` → `for batch`

每个 batch 走完 9 步：

```
prompt batch
  → ① generate sequences (Actor rollout，生成 token 序列)
  → ② repeat align (同一 prompt 重复 n 次，对齐 n 条生成回答)
  → ③ compute valid tokens
  → ④ compute values (Critic 估计 V(t)，GRPO 跳过)
  → ⑤ compute scores (rule-based 打分，0 / 0.1 / 1.0)
  → ⑥ compute rewards (score - kl_coef * KL，加入 KL 惩罚)
  → ⑦ compute advantages (GAE 或 GRPO，由 adv_estimator 决定)
  → ⑧ update critic (GRPO 跳过)
  → ⑨ update actor (等 critic warmup 之后才开始)
```

---

## 关键细节

### repeat align 为什么需要
GRPO 每个 prompt 采样 n 条回答，`generate_sequences` 生成了 n 条 response，
但原始 batch 每个 prompt 只有 1 条。`batch.repeat(n)` 把 prompt 复制 n 份，才能和 n 条 response 一一对应。

### score vs reward 的区别
- `token_level_scores`：纯任务分（0 / 0.1 / 1.0）
- `token_level_rewards`：score - kl_coef * KL(t)

KL(t) = log(新模型概率) - log(参考模型概率)，衡量新策略跑偏多远。
`kl_coef` 是惩罚力度系数，由 `AdaptiveKLController` 自动调节。

**为什么不直接用 score？** score 只管答对，不管跑偏。加 KL 惩罚让模型既要答对，又不能偏离参考模型太远，防止 reward hacking 和语言能力退化。

### critic warmup
Critic 刚初始化时 V(t) 估计很差，advantage 质量低。
先让 Critic 训几步稳定后，再开始更新 Actor，避免用噪声 advantage 把 Actor 训歪。

### use_rm vs rule-based
- `if self.use_rm`：用神经网络 Reward Model 打分
- `self.reward_fn`：Rule-based 打分（TinyZero 用这个）
两者可以叠加使用。

---

## 自测

- PPO 训练循环的 9 步是什么？
- score 和 reward 有什么区别？为什么要减 KL？
- critic warmup 是什么，为什么需要？
- GRPO 模式下哪几步会跳过？
