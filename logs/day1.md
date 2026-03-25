# Day 1 复盘：提交第一个 SLURM Job

## 完成情况 ✅

- 确认 conda 环境（tinyzero）和数据（data/countdown/）都 ready
- 写好 train.slurm，提交到 A100-40GB 节点
- wandb 配置完成，曲线正常上传
- critic/score/mean 从 0.02 → 0.08，Step 22 后明显上升 → 模型在学习

---

## 整体流程

```
Mac (VS Code)
    ↓ SSH
xlogin1（登录节点）← 只提交任务/看日志，严禁跑程序
    ↓ sbatch
A100 计算节点 ← 后台跑训练，不用守着
    ↓
logs/ + wandb（浏览器看曲线）
```

---

## train.slurm 三块结构

```bash
# 第一块：告诉 SLURM 要什么资源
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1   # 指定 GPU
#SBATCH --mem=64G              # 系统内存，必须写！
#SBATCH --time=02:00:00

# 第二块：准备环境
source ~/.bashrc               # 让 conda 可用
conda activate tinyzero
cd ~/TinyZero

# 第三块：设置变量，启动训练
export BASE_MODEL=Qwen/Qwen2-0.5B
export DATA_DIR=./data/countdown
bash ./scripts/train_tiny_zero.sh
```

本质：一封给集群的信，说清楚要什么资源、怎么启动。

---

## 踩坑记录

**坑1：VRAM OOM**
Ray 同时加载 Actor + Critic + Ref 三份权重，原始脚本按多卡设计，单卡装不下。
解法：`train_batch_size 256→64`，`response_length 1024→512`，`gpu_memory_utilization 0.4→0.3`

**坑2：系统内存 OOM**
SLURM 默认 RAM 极少，Ray 多进程被 kill。
解法：`#SBATCH --mem=64G` 必须显式写。

**坑3：conda 在 login 节点不可用**
scratch 只挂载在计算节点，login 节点没有。
解法：用 srun 进计算节点再操作，或 sbatch 里 `source ~/.bashrc`。

**坑4：wandb 需要先登录**
sbatch job 无法交互，需要提前 `srun ... wandb login` 一次，key 存到 `~/.netrc`。

---

## wandb 关键曲线解读

| 曲线 | 含义 | 看什么 |
|------|------|--------|
| `critic/score/mean` | 模型回答得分均值 | **最重要，要上升** |
| `actor/ppo_kl` | 当前策略和初始策略的差距 | 保持稳定，不要爆掉 |
| `actor/pg_loss` | Policy Gradient 损失 | 波动正常 |

**Aha Moment**：Step 22 附近曲线突然起飞，是模型开始学会"先思考再回答"格式的信号。

---

## Day 2 任务

读懂"这条 score 曲线的分数是怎么算出来的"：

1. `examples/data_preprocess/countdown.py` — 数据从哪来，prompt 怎么构造
2. `verl/protocol.py` — DataProto 数据格式
3. `verl/utils/reward_score/countdown.py` — 奖励函数怎么打分
