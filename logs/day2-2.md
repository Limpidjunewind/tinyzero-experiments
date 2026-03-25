# Day 2-2：为什么要 SFT → RL，逻辑链完整复盘

---

## 核心问题：Base Model 能直接用吗？

Base Model 只做了一件事：在海量文本上预测下一个 token。它不知道自己是 assistant，不知道要回答问题，也没有任何"对话"的概念。

所以需要后训练（Post-training）。后训练分两阶段：**SFT → RL**。每一阶段解决不同的问题。

---

## 第一阶段：SFT 做了什么

**训练方式：** Teacher Forcing。给模型看完整的 (输入, 输出) 对，每一个 token 都有正确标签，用 cross-entropy loss 监督。

**SFT 解决的问题：**
- 让模型学会 chat template 格式（`<|im_start|>user ... <|im_start|>assistant`）
- 学会基本的指令遵循：不答非所问、能按指定格式输出
- 建立对话角色的基本概念

**SFT 的本质：模仿。** 模型学的是"训练数据长什么样子"，能力上限被数据集质量死死卡住。

**SFT 的两个关键副作用：**

1. **Chat Template 成为硬依赖**
   Instruct 模型见过几十亿次 `<|im_start|>assistant\n` 这个序列，权重里编码了"看到这个 token → 应该生成回答"的条件概率。推理时如果格式不对，输入分布偏离训练分布（distribution shift），模型不知道自己处于"回答模式"，输出质量大幅下降。这不是 bug，是 SFT 的必然结果。

2. **继续加数据会过拟合**
   SFT 数据加到一定量后，模型开始机械背诵训练集，在未见过的问题上泛化性变差（幻觉增多）。这是 SFT 的天花板信号。

---

## SFT 的边界：为什么训不出真正的推理

以 `<think>` 格式为例。你可以在 SFT 数据里加上 `<think>` 标签，让模型学会输出这个格式。但这里有一个根本问题：

> **SFT 里，think 内容和 answer 对错没有因果连接。**

SFT 的 loss 是 token 级别的，每个 token 都被独立监督。模型学到的是：
- "遇到数学题，输出 `<think>` 标签"
- "输出一些看起来像推理的文字"
- "输出 `<answer>` 标签"

但它不知道 think 里写了什么会影响 answer 对不对。因为 SFT 的标注数据里，无论 think 写得好不好，answer 都已经是固定的正确答案，模型没有办法通过 think 内容去"推断"答案。

这就是 SFT 只能训出"表面 think"的原因。

---

## 第二阶段：RL 解决什么问题

**训练方式：** 模型自己生成完整输出（rollout），只看最终结果对不对（reward），通过 PPO 反向强化好的行为。

**RL 的本质：探索 + 试错。** 不告诉模型"应该怎么做"，只告诉它"做了之后结果好不好"。

**为什么 RL 能训出真正的推理：**

Transformer 生成 token 时，attention 可以回看之前所有的 token。`<think>` 里写的内容，模型生成 `<answer>` 时是真的能"看到"并利用的。这让 think tokens 成为真实的**草稿纸**（scratchpad）。

RL 训练过程中：
- 模型随机探索，碰巧在 think 里写了有用的中间步骤
- 有中间步骤的 rollout，answer 正确率更高 → reward 更高
- 这个行为被统计强化 → 模型越来越倾向于先写推理再作答

这就是 **Aha Moment** 的机制：不是模型"理解"了推理的意义，而是"认真写 think → 答对 → 拿 reward"这条因果链被强化了。DeepSeek-R1 里模型自己涌现出回顾、反思的行为，本质也是同一个机制。

---

## 为什么必须先 SFT 再 RL

RL 需要一个**靠谱的初始策略**才能有效探索。

如果直接从 Base Model 开始 RL：
- 模型的输出是随机的文本续写
- 绝大多数 rollout 都是乱码，reward 全是 0
- reward 信号极其稀疏，梯度几乎无法传递，训练无法收敛

SFT 先让模型学会基本的对话格式和指令遵循，RL 的探索才有意义的起点。

**实证：DeepSeek R1-Zero** 直接跳过 SFT 从 Base Model 开始 RL，推理能力确实涌现了，但输出混乱（中英文混杂、格式不稳定、可读性差）。正式版 R1 加了 cold-start SFT 阶段才稳定。

---

## 何时从 SFT 切换到 RL（实践判断）

两个信号同时成立才切：

**信号一：SFT 已经充分**
- 模型能稳定遵循指令、格式不再混乱
- SFT loss 降到低位不再波动

**信号二：RL 有提升空间**
- `pass@k 明显高于 pass@1`：模型回答 k 次时通过率远大于回答 1 次，说明能力上限在那，但单次生成不稳定。RL 的目标就是把 pass@k 的能力"蒸馏"到 pass@1。

---

## 完整 Pipeline

```
Base Model
    ↓
冷启动 SFT（几千条高质量数据）
  → 目的：学会基本格式、稳定指令遵循
    ↓
RL 训练（PPO / GRPO）
  → 目的：探索推理策略、提升 pass@1
    ↓
Reject Sampling Fine-tuning（可选）
  → 收集 RL 阶段模型自己生成的、通过验证的高质量样本
  → 回头再做一轮 SFT，巩固 RL 学到的推理行为
    ↓
循环迭代（螺旋式上升）
```

---

## 逻辑链一句话总结

> Base model 不会对话 → SFT 教会格式和规矩（保下限）→ SFT 上限被数据卡死 → RL 用 reward 代替标注，让模型自己探索"怎么赢"（提上限）→ think 的因果链在 RL 里真实存在（attention scratchpad）→ 推理能力涌现。
