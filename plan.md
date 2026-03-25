
# TinyZero 1 周冲刺计划

> 前提：CS285已过，TinyZero架构已分析，SoC GPU环境已配好。
> 目标：跑通训练 + 能用自己话讲清楚 PPO 全链路 + 包装为面试证据。
> 带着问题读代码，比按顺序精读效率高 10 倍。遇到报错/疑问再回头看细节。
---

## Day 1 ✅：提交第一个 SLURM job

**完成**：critic/score/mean 从 0.02 → 0.08，Step 22 后明显上升，Aha Moment 出现。

1. 数据准备（如果还没做）
```bash
cd /home/j/jingwenl/TinyZero
conda activate tinyzero
python examples/data_preprocess/countdown.py --local_dir ./data/countdown
```

2. 写 sbatch 脚本 → 本地写好，保存同步到集群
3. 提交：`sbatch train.slurm`
4. 跑着的同时：读 `examples/data_preprocess/countdown.py`（20分钟）

**完成标志**：job 状态变 R，wandb 有 log 写入。

---

## Day 2：数据 + 奖励函数精读

**目标**：能完整描述"数据从哪来，怎么打分"。

| 文件 | 重点 | 预计时间 |
|------|------|----------|
| `examples/data_preprocess/countdown.py` | prompt 模板构造，parquet 格式 | 1h |
| `verl/protocol.py` | DataProto 三层结构（batch / non_tensor_batch / meta_info） | 1.5h |
| `verl/utils/reward_score/countdown.py` | extract → validate → evaluate → score | 1h |

**动手**：修改 countdown.py 参数重新生成数据，打开 parquet 看内容。


---

## Day 3：PPO 算法核心

**目标**：能白板讲清楚 clip loss + GAE，不需要看代码。

| 文件 | 重点 | 预计时间 |
|------|------|----------|
| `verl/trainer/ppo/core_algos.py` | compute_policy_loss / compute_gae_advantage_return / compute_grpo_outcome_advantage / AdaptiveKLController | 3h |

**动手**：用简单数字手算 GAE（3个token的序列），对照代码验证。
**面试题**：能用1-2句话回答"PPO为什么要clip？不clip会怎样？"

**Day 3 结束自测**（不看代码回答）：
- GAE 的 λ 参数控制什么？λ=0 和 λ=1 分别退化成什么？
- clip ratio 的 ε 设成 0.2 是什么意思？改成 0.5 会怎样？
- GRPO 和 PPO 的 advantage 计算有什么区别？

---

## Day 4：训练主循环（最重要）

**目标**：能画出 PPO dataflow 的 8 步循环，每步知道输入输出。

| 文件 | 重点 | 预计时间 |
|------|------|----------|
| `verl/trainer/main_ppo.py` | RewardManager 路由逻辑、Ray 初始化 | 1h |
| `verl/trainer/ppo/ray_trainer.py` fit() | generate → ref → critic → reward → kl → advantage → update_critic → update_actor | 3h |

**动手**：在 fit() 里逐步加注释，画出你自己的 dataflow 图。

---

## Day 5：Worker + 分布式（知道就行，不用深入）

**目标**：能解释"一个 Worker 怎么同时做 Actor 和 vLLM Rollout"。

| 文件 | 重点 | 预计时间 |
|------|------|----------|
| `verl/workers/fsdp_workers.py` | ActorRolloutRefWorker：init_model / generate_sequences / update_actor | 2h |
| `verl/trainer/config/ppo_trainer.yaml` | 四大配置块 + 每个参数在代码哪里被用 | 1h |

不需要完全理解 FSDP 细节，但要能说出："FSDP 把参数分片到多卡，前向时 all-gather，这样单卡显存压力小"。

---

## Day 6：对比实验

**目标**：动手改参数，有自己跑出来的实验结论。

| 实验 | 改什么 | 观察什么 |
|------|--------|----------|
| PPO vs GRPO | `algorithm.adv_estimator: gae → grpo` | 去掉 Critic 后收敛曲线对比 |
| KL 系数影响 | `algorithm.kl_ctrl.kl_coef: 0.001 → 0.01` | score 稳定性变化 |

第二个实验可选，时间不够跳过。

---

## Day 7：包装 + 复盘

**目标**：TinyZero 变成面试可以讲的项目。

1. 用 STAR 结构写出 TinyZero 项目描述（300字以内）
2. 准备追问答案（根据测验暴露的漏洞重点准备）：
   - "为什么用 PPO 而不是 SFT？"
   - "奖励函数怎么设计的？有没有 reward hacking？"
   - "GRPO 和 PPO 的区别？"
   - "为什么 format_score 设成 0.1 而不是 0 或 0.5？" ← 测验暴露的弱点
   - "prompt template 为什么要区分 base 和 instruct？" ← 测验暴露的弱点
3. 模拟面试：让 Claude 扮演面试官，追问 5 轮，录下来回听
4. 更新 GMS weakness.md 和 todo.md

---

## 核心文件优先级

| 优先级 | 文件 | 天数 |
|--------|------|------|
| P0 | ray_trainer.py fit() | Day 4 |
| P0 | core_algos.py | Day 3 |
| P0 | main_ppo.py | Day 4 |
| P1 | protocol.py | Day 2 |
| P1 | countdown.py (reward) | Day 2 |
| P1 | countdown.py (data) | Day 1-2 |
| P2 | fsdp_workers.py | Day 5 |
| P3 | single_controller/ | 跳过（了解概念即可） |
