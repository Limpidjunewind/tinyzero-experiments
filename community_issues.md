# TinyZero 社区踩坑汇总

> 整理自 GitHub Issues、verl 文档、社区论坛，结合本地实验记录

---

## 坑1：flash-attention 安装失败

**现象**：`pip install flash-attn` 编译失败或耗时极长。

**解法**：直接从预编译 whl 安装，跳过本地编译：
1. 去 [Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases) 找对应 CUDA/Python 版本的 whl
2. `pip install flash_attn-xxx.whl`

---

## 坑2：3B PPO 在 2 张 H100-96 上 OOM

**现象**：训练启动后在 Critic 更新阶段 OOM 崩溃，报错：
```
torch.OutOfMemoryError: CUDA out of memory.
GPU 0 has a total capacity of 93.09 GiB of which ~32 MiB is free.
```

**根因**：Actor + Critic + Ref 三份 3B 权重 + optimizer state + activation 同时占显存，两卡 192GB 不够。

**解法**：
```bash
# 1. Actor 和 Critic 都开 gradient checkpointing（用算力换显存）
actor_rollout_ref.model.enable_gradient_checkpointing=True
critic.model.enable_gradient_checkpointing=True

# 2. Ref 参数卸载到 CPU
actor_rollout_ref.ref.fsdp_config.param_offload=True

# 3. 降低 vLLM 显存占用
actor_rollout_ref.rollout.gpu_memory_utilization=0.25

# 4. 缩小 micro_batch
actor_rollout_ref.actor.ppo_micro_batch_size=2
critic.ppo_micro_batch_size=2

# 5. 防止内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

社区参考：[TinyZero Issue #30：Qwen 3B OOMs on 2 H100s](https://github.com/Jiayi-Pan/TinyZero/issues/30)、[TinyZero Issue #74：Ray OOM](https://github.com/Jiayi-Pan/TinyZero/issues/74)

---

## 坑3：两卡通信失败，NCCL Duplicate GPU 报错

**现象**：训练启动卡死，报错：
```
Duplicate GPU detected: rank 0 and rank 1 both on CUDA device 82000
ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
```

**根因**：H100 节点有 NVLink，Ray + FSDP 初始化多个进程组时触发 NCCL P2P 冲突。

**解法**：slurm 脚本里同时加两个变量（只加一个不够）：
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

社区参考：[verl Issue #1096](https://github.com/verl-project/verl/issues/1096)、[TRL Issue #3158](https://github.com/huggingface/trl/issues/3158)、[NVIDIA NCCL Issue #1241](https://github.com/NVIDIA/nccl/issues/1241)

---

## 坑4：训练输出全变感叹号，entropy collapse

**现象**：训练进行正常（reward 0.4+，grad_norm 正常，KL 无异常），某一 step 突然退化：
```
<think>!!!!!!!!!!!!!!!!!!!</think>
```
或直接输出：
```
<think><|endoftext|>
```
reward 归零，无法恢复。**崩溃前没有任何可见的预警信号。**

崩溃是**概率性**的——同样的参数，有时 step 74 崩，有时跑过 100 步都不崩。

**根因（最可能的解释）：随机游走 + 吸收态**

PPO 训练本质是随机过程，策略在参数空间里随机游走。`!!!` 是一个**吸收态**：
- 一旦模型输出 `!!!`，reward = 0，所有样本 reward 相同
- advantage ≈ 0，梯度 ≈ 0，模型无法自我纠正
- 每次 rollout 都有一定概率随机游走到这个状态，跑得越久累积概率越高

这解释了为什么：崩溃无预兆（不是渐变，是某步刚好走偏）、随机出现（取决于采样到的数据）、无法恢复。

**另一个可能机制（verl 特有）：训练-推理数值不一致**

vLLM（生成阶段）和 FSDP（训练阶段）对同一个 token 算出的概率有微小差异。特定 token 序列下差异积累，导致 `old_log_prob` 出现 `-inf`，PPO ratio 变 NaN，梯度爆炸。参考 [verl Issue #721](https://github.com/volcengine/verl/issues/721)。

**本地实验结论（2026-04）**

| 配置 | entropy_coeff | 结果 |
|------|--------------|------|
| bs=256, mini=128 | 0.001 | step ~50 崩，输出 `!!!` |
| bs=256, mini=128 | 0.01 | 存活至 step 400+，test score ~0.45 |
| bs=256, mini=128 | 0.05 | step ~30 崩，entropy 反向飙升 |

entropy_coeff=0.01 能阻止崩溃，说明**保持策略多样性**可以降低随机游走撞上吸收态的概率。

**解法**：
- **最直接**：提高 `entropy_coeff`（从 0.001 → 0.01），实测有效
- 换用 Instruct 模型（有 SFT 先验，初始分布更合理）
- 使用 DAPO 的 Dynamic Sampling，过滤 std=0 的 batch（所有 reward 相同时跳过更新）
- 提高 `kl_coef` 防止策略偏移过大

**不要做的事**：
- 不要调小 `kl_coef`（默认 0.001 已经很小了）
- 不要用 `format_score`（给懒惰输出额外奖励，会加速 reward hacking）

社区参考：[TinyZero Issue #63：outputs become ! after think](https://github.com/Jiayi-Pan/TinyZero/issues/63)、[verl Issue #721](https://github.com/volcengine/verl/issues/721)、[verl Entropy Mechanism 文档](https://verl.readthedocs.io/en/latest/algo/entropy.html)

---

## 坑5：use_remove_padding=True 与 flash attention 冲突导致 OOM

**现象**：崩在 flash attention 的 `_upad_input` → `torch.gather`：
```
torch.OutOfMemoryError: CUDA out of memory.
File "flash_attn/bert_padding.py", line 17, in forward
    return torch.gather(
```

**根因**：`use_remove_padding=True` 触发 flash attention 内部的 padding 去除逻辑，在显存已接近上限时 gather 操作申请不到显存。

**解法**：降低整体显存压力（gradient checkpointing + ref offload + 降低 gpu_memory_utilization），不必关闭 `use_remove_padding`。

---

## 坑6：hydra 参数未覆盖默认值，adv_estimator 实际跑成 gae

**现象**：脚本里写了 `algorithm.adv_estimator=grpo`，但日志显示实际是 `gae`，Critic 被启动，OOM。

**根因**：hydra 默认配置 `ppo_trainer.yaml` 里 `adv_estimator: gae`，命令行参数未能覆盖。

**解法**：用 `++` 强制覆盖：
```bash
++algorithm.adv_estimator=grpo
```

---

## 相关资源

- [TinyZero GitHub Issues](https://github.com/Jiayi-Pan/TinyZero/issues)
- [verl Issue #1096：Duplicate GPU NCCL](https://github.com/verl-project/verl/issues/1096)
- [verl Entropy Mechanism 文档](https://verl.readthedocs.io/en/latest/algo/entropy.html)
- [TinyZero Issue #30：3B OOM on H100](https://github.com/Jiayi-Pan/TinyZero/issues/30)
- [TinyZero Issue #63：entropy collapse / ! outputs](https://github.com/Jiayi-Pan/TinyZero/issues/63)
- [TinyZero Issue #74：Ray OOM](https://github.com/Jiayi-Pan/TinyZero/issues/74)
- [Dao-AILab flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases)
