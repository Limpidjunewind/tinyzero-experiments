# Day 3 — GAE 核心算法

## 今天搞懂的东西

### Actor vs Critic
- **Actor**：生成 token 的模型（就是我们在训练的那个）
- **Critic**：价值评估器，给定当前状态预测未来能拿多少 reward，输出 V(t)

### TD Error δ(t)
```
δ(t) = reward(t) + γ·V(t+1) - V(t)
```
`V(t+1) - V(t)` = 多走这一步，未来预期涨了还是跌了。δ(t) > 0 说明这步比 Critic 预期好。

### GAE Advantage
```
A(t) = δ(t) + γλ·A(t+1)
```
把多个 TD error 加权求和。λ 控制看多远：λ=0 只看一步，λ=1 看到最后。

### 为什么要倒序算
必须先知道 A(t+1) 才能算 A(t)，所以从最后一个 token 往前算。

---

## core_algos.py — compute_gae_advantage_return（第70行）

```python
lastgaelam = 0          # 初始化，第一轮没有 A(t+1)
advantages_reversed = [] # 存结果的列表

for t in reversed(range(gen_len)):
    nextvalues = values[:, t+1] if t < gen_len-1 else 0.0  # 最后一步无未来，用0
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]  # δ(t)
    lastgaelam = delta + gamma * lam * lastgaelam            # A(t)
    advantages_reversed.append(lastgaelam)

advantages = torch.stack(advantages_reversed[::-1], dim=1)  # 翻转回正序
```

**关键理解：**
- `values[:, t]` 是整个 batch 第 t 步的 V 值，batch 里所有样本同时计算
- 最后一步 `nextvalues = 0.0`，因为序列结束后没有未来价值
- `advantages_reversed[::-1]` 把倒序列表翻转回正序




### GAE 退化情况
- λ=0：单步TD
- λ=1：蒙特卡洛

### PPO Clip Loss
核心逻辑：算两个 loss，一个用原始 ratio，一个用 clip 到 [1-ε, 1+ε] 的 ratio，取 max，保证 ratio 越界时梯度被压制，每次更新小步走，防止策略跑偏。

```python
ratio = exp(log_prob - old_log_prob)
pg_losses  = -advantages * ratio
pg_losses2 = -advantages * clamp(ratio, 1-ε, 1+ε)
pg_loss = max(pg_losses, pg_losses2)   # 取更保守的
```

### pg_clipfrac
是什么：这个 batch 里 ratio 越界（被 clip）的 token 占比。监控指标，太高说明新旧策略差距大，训练不稳定。

### Day 3 自测

- λ 控制用几步来计算当前的预期，λ=0 退化成 TD(0)，λ=1 退化成蒙特卡洛
- ε=0.2 意味着允许 ratio 浮动 ±0.2，改成 0.5 会更宽松，训练更激进但不稳定
- GRPO vs PPO advantage：GRPO 用组内均值当 baseline（不需要 Critic），PPO 用 Critic 提供 V(t) + GAE 计算

---

## 手算 GAE 验证

```
序列 3 个 token，γ=1，λ=0.5
rewards = [0, 0, 1]
values  = [0.5, 0.3, 0.1]

δ(2) = 1 + 0 - 0.1 = 0.9
δ(1) = 0 + 0.1 - 0.3 = -0.2
δ(0) = 0 + 0.3 - 0.5 = -0.2

A(2) = 0.9
A(1) = -0.2 + 0.5 * 0.9 = 0.25
A(0) = -0.2 + 0.5 * 0.25 = 0.075
```

**直觉**：δ(0)=-0.2 看起来这步很差，但 A(0)=0.075 说明整体略好于预期。
GAE 的本质是**功劳往前传**——最后答对了，前面铺垫的 token 也有功劳，不该被惩罚。

## 遗留问题
- 无
