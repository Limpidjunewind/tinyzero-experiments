# TinyZero 7B GRPO 实验记录

## 概览

| 项目 | 详情 |
|---|---|
| 模型 | Qwen/Qwen2.5-7B |
| 算法 | GRPO (`adv_estimator=grpo`) |
| 实验名 | `7b-grpo-h100-96` |
| W&B 项目 | `tinyzero-7b-grpo` |
| Checkpoint 路径 | `/mnt/scratch/j/jingwenl/TinyZero_checkpoints/tinyzero-7b-grpo/7b-grpo-h100-96/actor/global_step_50` |
| 备份路径 | `/mnt/scratch/j/jingwenl/checkpoints_backup/tinyzero-7b-grpo/7b-grpo-h100-96/` |

---

## 训练参数（完整）

### 数据

| 参数 | 值 |
|---|---|
| `data.train_files` | `data/countdown/train.parquet` |
| `data.val_files` | `data/countdown/test.parquet` |
| `data.train_batch_size` | 128 |
| `data.val_batch_size` | 128 |
| `data.max_prompt_length` | 256 |
| `data.max_response_length` | 512 |

### Actor / Rollout

| 参数 | 值 |
|---|---|
| `actor.optim.lr` | 1e-6 |
| `actor.ppo_mini_batch_size` | 32 |
| `actor.ppo_micro_batch_size` | 2 |
| `actor.entropy_coeff` | 0.001 |
| `actor.clip_ratio` | 0.2 |
| `actor.ppo_epochs` | 1 |
| `actor.use_kl_loss` | False |
| `rollout.n` | **1**（注意：1.5B GRPO 用的是 8，此处仅 1 个样本/prompt） |
| `rollout.temperature` | 1.0 |
| `rollout.tensor_model_parallel_size` | 2 |
| `rollout.gpu_memory_utilization` | 0.5 |
| `rollout.enforce_eager` | True |

### Critic

| 参数 | 值 |
|---|---|
| `critic.optim.lr` | 1e-5 |
| `critic.model.path` | Qwen/Qwen2.5-7B |
| `critic.ppo_micro_batch_size` | 2 |
| `critic.enable_gradient_checkpointing` | True |
| `critic.cliprange_value` | 0.5 |

### 算法

| 参数 | 值 |
|---|---|
| `algorithm.adv_estimator` | grpo |
| `algorithm.kl_ctrl.kl_coef` | 0.001 |
| `algorithm.kl_ctrl.type` | fixed |
| `algorithm.gamma` | 1.0 |
| `algorithm.lam` | 1.0 |

### 训练调度

| 参数 | 值 |
|---|---|
| `trainer.total_epochs` | 15 |
| `trainer.total_training_steps`（自动计算）| 38400 |
| `trainer.n_gpus_per_node` | 2 |
| `trainer.nnodes` | 1 |
| `trainer.save_freq` | 50 |
| `trainer.test_freq` | 50 |

---

## 两次运行记录

| | Run 1 | Run 2 |
|---|---|---|
| SLURM Job ID | 462726 | 466119 |
| 开始时间 | 2026-03-23 14:53 | 2026-03-23 18:56 |
| 结束时间 | 2026-03-23 16:44 | 2026-03-23 20:47 |
| 实际时长 | 1h51min | 1h50min |
| GPU | 2x H100 NVL | 2x H100 NVL |
| SLURM 状态 | FAILED (exit code 1) | FAILED (exit code 1) |
| 完成 steps | ~50 | ~50 |

---

## 问题分析

### 为什么只跑了 50 steps？

7B 模型在 2xH100 NVL 上每个 step 约需 **2 分钟**，50 steps ≈ 100 分钟，加上初始化约 10 分钟，总计 ~1h50min，与实测一致。两次均以 exit code 1 **崩溃退出**（非超时），可能是在 step 50 保存 checkpoint 后下一步崩溃。

### Entropy 为什么接近 0？

**根本原因：`rollout.n=1`**

- GRPO 需要对同一 prompt 采样多个响应（group），再计算 group-relative advantage：`A = (r - mean(r_group)) / std(r_group)`
- 1.5B 的 GRPO 配置用的是 `n=8`（每个 prompt 生成 8 个响应）
- 7B 这里用的是 `n=1`，group size=1，std=0，advantage 退化
- 结果是策略梯度信号近乎为 0，模型无法从探索中学习，迅速坍缩成确定性策略 → entropy → 0

### 建议修复

1. **加 `rollout.n=8`**（最重要，必须修）
2. **增加 `actor.entropy_coeff`**（从 0.001 → 0.01）防止 entropy collapse
3. **延长时间**：`#SBATCH --time=24:00:00` 或 `48:00:00`，让训练跑到几百 steps
4. **加 resume**：`trainer.resume_mode=auto`，从 global_step_50 继续

---

## 崩溃原因分析（GPU OOM）

### 排除 CPU 内存

`sacct` 显示 CPU RAM 峰值约 122GB，申请了 480GB，远未打满，**不是 CPU OOM**。

### GPU OOM（推断）

崩溃时序：step 50 训练 → **保存 checkpoint** → step 50 validation 生成（output log 最后一行是 val 输出）→ **崩溃**

原因：FSDP 保存 checkpoint 时需要在各 rank 上 **gather 完整模型参数**（7B × bf16 ≈ 14GB），两张 H100 NVL（每张 94GB）上显存峰值瞬间飙升。保存完毕后 vLLM 尝试 reload 推理引擎（`free_cache_engine=True` 意味着每步都要重新分配 KV cache），此时显存已经碎片化/不足，触发 OOM 崩溃。

两次 run 均在同一位置崩溃（step 50 后），且 `save_freq=50`——完全一致，进一步佐证是 save 触发的显存问题。

### 如果重跑需注意

改成 `rollout.n=8` 后每步显存压力更大（8 个序列并行生成），建议：
- 换 **4x GPU**（`--gres=gpu:h100-96:4`，`n_gpus_per_node=4`）
- 或开启 `actor.fsdp_config.param_offload=True`（慢但省显存）
- `rollout.gpu_memory_utilization` 从 0.5 降到 0.4

---

## 集群文件结构

```
~/TinyZero/
├── data/
│   ├── countdown/
│   │   ├── train.parquet       # 训练集（Countdown 数学游戏）
│   │   └── test.parquet        # 测试集
│   └── sft_coldstart.json      # SFT 冷启动数据
│
├── verl/                       # 核心框架（fork 自 verl）
│   ├── trainer/
│   │   ├── main_ppo.py         # 训练入口（GRPO/PPO 共用，靠 adv_estimator 区分）
│   │   ├── config/             # Hydra 默认配置
│   │   └── ppo/                # PPO/GRPO trainer 主逻辑
│   ├── workers/
│   │   ├── actor/              # Actor 模型（策略网络）
│   │   ├── critic/             # Critic 模型（价值网络，PPO 专用）
│   │   ├── rollout/vllm_rollout/  # vLLM 推理生成
│   │   ├── reward_model/       # Reward model（本项目用规则 reward）
│   │   └── sharding_manager/   # FSDP 分片管理
│   ├── utils/
│   │   ├── reward_score/       # 规则 reward 计算（验证算式是否等于目标数）
│   │   ├── dataset/            # 数据加载
│   │   └── logger/             # wandb/console 日志
│   └── single_controller/      # Ray 分布式控制器
│
├── examples/
│   ├── data_preprocess/        # 各数据集预处理脚本
│   └── grpo_trainer/           # 官方 GRPO 示例（含正确的 n=8 配置，可参考）
│
├── checkpoints/                # 软链接，实体在 scratch
│   ├── tinyzero-7b-grpo/
│   ├── sft-coldstart-1.5b/
│   └── sft-coldstart-1.5b-merged/
│
├── eval_results/               # 各模型评估结果 JSON
├── logs/                       # SLURM .out / .err 日志
├── wandb/                      # wandb 本地缓存
│
├── submit_7b_grpo.sh           # 7B GRPO 提交脚本（有 n=1 bug）
├── submit_7b_ppo.sh            # 7B PPO 提交脚本
├── train_1.5b_grpo.slurm       # 1.5B GRPO（n=8，正确）
├── train_1.5b_ppo.slurm
├── train_1.5b_ppo_coldstart.slurm
├── train_3b_ppo.slurm
├── eval_checkpoints.py         # 评估脚本
└── generate_sft_data.py        # SFT 数据生成

# Scratch 上的实际 checkpoint 路径
/mnt/scratch/j/jingwenl/TinyZero_checkpoints/
├── tinyzero-7b-grpo/7b-grpo-h100-96/actor/global_step_50/   # 7B GRPO（唯一）
└── TinyZero/
    ├── grpo-1.5b/
    ├── ppo-1.5b/
    ├── ppo-1.5b-coldstart/
    └── ppo-3b/
```
