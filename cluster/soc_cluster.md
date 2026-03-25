# SoC Compute Cluster 使用手册

## 登录
```bash
ssh jingwenl@xlogin1.comp.nus.edu.sg  # 或 xlogin0, xlogin2
```
文档：https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/

---

## 节点架构

```
xlogin（登录节点）← 只提交任务/看日志，严禁跑程序
    ↓ sbatch / srun
计算节点 ← 真正跑训练的地方
```

### GPU 节点一览

| 节点 | 显卡 | 显存 | 适合任务 |
|------|------|------|----------|
| xgpc/xgpd/xgpe/xgpf | Titan V / nv | 12GB | 测试、小实验 |
| xgpg/xgph | A100-40GB | 40GB | 0.5B-3B 训练（主力） |
| xgpj | A100-80GB | 80GB | 大模型，竞争激烈 |
| xgpi | H100-47GB / H100-96GB | 47/96GB | 7B+ 训练 |
| （新）H200 | H200-141GB | 141GB | 超大模型 |

### 当前负载参考（2026-03-19 登录时）
- 整体：CPU 10.2%，Memory 14.2%，**GPU 41.1%**
- A100-40：53.6% 占用（有空位，推荐用）
- A100-80：**90.9%**（基本满了，别抢）
- H100-96：80.0%（紧张）
- H200-141：25.0%（新卡，相对空）

---

## 存储结构

| 路径 | 用途 | 备注 |
|------|------|------|
| `/home/j/jingwenl/` | 主目录 | 容量小，放代码/脚本 |
| `/mnt/scratch/j/jingwenl/` | 大文件存储 | conda/模型缓存放这里 |

**重要**：scratch 只挂载在计算节点，login 节点访问不到 → 所以 conda 在 login 节点不可用。

---

## 标准 sbatch 模板

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1     # 指定 GPU 型号，不写就随机分配
#SBATCH --mem=64G                 # 必须写！不写只给几 GB，Ray 会 OOM
#SBATCH --time=02:00:00          # 最大运行时间
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/.bashrc
conda activate tinyzero

cd ~/TinyZero
mkdir -p logs

# 你的训练命令
bash ./scripts/train_tiny_zero.sh
```

**长时间任务用 `gpu-long` 分区：**
```bash
#SBATCH --partition=gpu-long
```

---

## 常用命令

```bash
# 查看所有 GPU 节点空闲情况
sinfo -o "%P %G %N %t" | grep idle

# 查看我的任务
squeue -u jingwenl

# 取消任务
scancel <JOBID>

# 看实时日志
tail -f logs/slurm-<JOBID>.out

# 交互式调试（借一张卡进去跑命令）
srun -p gpu -G a100-40:1 --mem=32G --time=01:00:00 --pty bash
```

---

## 踩坑记录

### 坑1：conda 在 login 节点不可用
scratch 没挂载，`conda: command not found`。
**解法**：用 `srun` 进计算节点再操作，或者在 sbatch 里 `source ~/.bashrc`。

### 坑2：VRAM OOM
Ray 同时加载 Actor + Critic + Ref 三份权重，原始 batch_size=256 是按多卡设计的。
**解法**：单卡跑时调小参数：
```
data.train_batch_size=64
data.max_response_length=512
actor_rollout_ref.actor.ppo_micro_batch_size=4
actor_rollout_ref.rollout.gpu_memory_utilization=0.3
```

### 坑3：系统内存 OOM
SLURM 默认给的 RAM 很少，Ray 多进程会被 kill。
**解法**：`#SBATCH --mem=64G` 必须显式写。

### 坑4：wandb 需要 API key
sbatch job 里无法交互式登录。
**解法**：先用 srun 交互节点 `wandb login` 一次，key 存到 `~/.netrc`，之后 sbatch 自动读取。

### 坑5：srun 里 conda 找不到
`bash -c "conda activate ..."` 不会自动 source bashrc。
**解法**：`bash -c "source ~/.bashrc && conda activate tinyzero && ..."`

### 坑6：计算节点下载 HuggingFace 模型
计算节点 ping 不通外网（ICMP 被墙），但 HTTPS 可以。
**解法**：在计算节点（srun 进去后）直接用：
```bash
hf download Qwen/Qwen2.5-3B
```
模型缓存在 `~/.cache/huggingface/hub/`，home 目录空间够用（26T 可用）。
`python -c "snapshot_download(...)"` 会卡住，用 cli 代替。

### 坑7：xgpj0（A100-80）架构不兼容
xgpj0 的 CPU 架构与 miniforge3 安装节点不同，sbatch 脚本报 `Exec format error`，conda 也报 `syntax error near unexpected token sys.argv`。
**解法**：改用 H100-96 节点（xgpi[0-1]，96GB 显存，架构兼容）：
```bash
#SBATCH --gres=gpu:h100-96:1
```

### 坑8：sbatch 脚本里 conda activate 失败
某些节点上 `source ~/.bashrc && conda activate tinyzero` 不生效。
**解法**：直接用 python 全路径，跳过 conda activate：
```bash
/mnt/scratch/j/jingwenl/miniforge3/envs/tinyzero/bin/python3 -m verl.trainer.main_ppo
```

### 坑9：H100-47 是 MIG 模式，FSDP 不兼容
xgpi[10-20] 的 H100-47 被切成 MIG 实例，verl/FSDP 需要完整 GPU 做跨卡通信，MIG 模式下卡间通信受限，训练进程启动后卡死，3h 内一个 step 都跑不了。
**解法**：改用 H100-96（xgpi[0-9]），完整卡，兼容 FSDP。

### 坑10：7B PPO 在 2张 H100-96 上 OOM
Actor + Critic + Ref 三份 7B 权重 + optimizer state，2张 H100-96（192GB）不够，每次都在 `adamw.py` optimizer step 时炸。`optimizer_offload` 参数在这个版本的 verl 里不生效，offload 方案无效。
**解法**：改用 GRPO（去掉 Critic），或者多节点多卡。

### 坑11：7B 模型加载时间过长
7B 模型在 2张 H100-96 上加载需要将近 2 小时，8h job 实际只剩 6h 跑训练。
**解法**：时间设 3 天（`3-00:00:00`），或者提前下载好模型缓存。

### 坑12：checkpoint 默认存在 home 目录，导致磁盘爆满
verl 默认把 checkpoint 存到项目目录下，7B 模型每个 checkpoint 约 14GB，`save_freq=50` 存几个就把 home 目录撑满（501GB）。
**解法**：提交 job 时加 `trainer.default_local_dir=/mnt/scratch/j/jingwenl/checkpoints`，把 checkpoint 存到 scratch。

---

## GPU 配额限制（QOS: normal）

通过 `sacctmgr show qos normal format=name,MaxTRESPU%120` 查到的 per-user 上限：

| GPU 型号 | 上限 |
|----------|------|
| A100-40 | 8 张 |
| A100-80 | 5 张 |
| H100-47 | 6 张 |
| **H100-96** | **2 张** |
| H200-141 | 1 张 |

超限会报 `QOSMaxGRESPerUser`，job 进入 PD 状态但不会跑。

---

## TinyZero 单卡 A100 能力边界

| 任务 | A100-40（40GB）| A100-80（80GB）| H100-96（96GB）|
|------|---------------|---------------|----------------|
| 0.5B PPO | ✅ | ✅ | ✅ |
| 1.5B PPO | ❌ OOM（实测）| ✅（xgpj0 架构问题）| ✅ 实测可用 |
| 3B PPO | ❌ OOM | ✅（xgpj0 架构问题）| ✅ 实测可用 |
| 7B LoRA SFT | ⚠️ | ⚠️ | ✅ |
| 7B 全参数 PPO | ❌ | ❌ | ⚠️ 需多卡 |

> 1.5B 及以上用 H100-96（xgpi[0-1]）最稳，A100-80 唯一节点 xgpj0 有架构兼容性问题。
