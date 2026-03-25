# Day 4 — 训练主循环

## main_ppo.py

### RewardManager 两条路径

```python
if 'rm_scores' in data.batch.keys():
    return data.batch['rm_scores']   # model-based 路径：直接返回
# 否则走 rule-based 路径
```

- **model-based RM**：神经网络打分，分数已由 RewardModelWorker 写入 `rm_scores`，直接取用。适合开放式任务（答案无唯一标准）。
- **rule-based RM**：按 `data_source` 字符串路由到对应规则函数（如 `countdown.compute_score`），代码直接验证答案。适合有确定答案的任务。TinyZero 用这条路径。

### Sparse Reward

```python
reward_tensor[i, valid_response_length - 1] = score
```

reward 只放在 response 最后一个有效 token，其余位置全为 0。
原因：整条 rollout 只有序列结束时才知道答案对错（terminal reward），中间每步没有即时信号。
**与 GAE 的关系**：正因为 reward 稀疏，GAE 才需要通过 λ 加权把终点信号往前传播，让每个 token 都能感知最终结果的贡献。

### Ray 与角色分工

Ray 是分布式调度框架，让 Actor、Ref、Critic 三个模型分别部署在不同 GPU 上，主进程通过 RPC 统一调度。

| 角色 | 职责 | 是否更新 |
|------|------|---------|
| Actor | 生成 response；接受梯度更新 | ✅ 有 optimizer |
| Ref | 提供参考 log prob，计算 KL | ❌ 冻结 |
| Critic | 估计每步状态价值 V(s) | ✅ 独立更新 |

**为什么 Actor 和 Ref 用同一个类 `ActorRolloutRefWorker`**：
两者都需要跑模型推理，代码大量重叠。初始化时传入 `role` 字符串，类内部设置布尔标志：
```python
self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
self._is_ref   = self.role in ['ref', 'actor_rollout_ref']
```
`_is_actor=True` 时初始化 optimizer、允许梯度更新；`_is_ref=True` 时只做推理，不更新参数。

---

## ray_trainer.py — fit() 8 步循环

| 步骤 | 名称 | 输入 | 输出 |
|------|------|------|------|
| 1 | generate | prompts (input_ids) | responses + actor log_prob |
| 2 | ref_log_prob | responses | 每个 token 在 Ref 下的 log_prob |
| 3 | values | prompts + responses | 每步 V(s)（Critic 估计） |
| 4 | reward | prompts + responses | token_level_scores（0/0.1/1.0） |
| 5 | KL penalty | actor log_prob + ref log_prob | token_level_rewards = scores - kl_coef * KL |
| 6 | advantage | rewards + values | A(t)（GAE） |
| 7 | update_critic | batch + A | 更新后的 Critic 参数 |
| 8 | update_actor | batch + A | 更新后的 Actor 参数 |

### On-policy 约束

每次 iteration 必须重新 generate，不能复用上一轮数据。
原因：update_actor 之后模型权重改变，旧数据的分布和新模型不匹配，用旧数据更新会引入分布偏移。
**PPO 的妥协**：同一批数据可跑多个 mini-batch epoch，但 clip 限制每次更新幅度（ratio 不超出 [1-ε, 1+ε]），保证新旧策略不偏离太远。

### KL penalty 为什么加进 reward 而不是 loss

- **加进 reward**：KL 是 per-token 的，经过 GAE 传播后每步都有个性化惩罚——跑偏的 token 多扣，没跑偏的不扣。
- **加进 loss**：整个 batch 一个 KL 标量，所有 token 一刀切，无法区分哪步跑偏。

加进 reward 粒度更细，credit assignment 更准确。

---

## 遗留问题
- 无
