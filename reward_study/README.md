# TinyZero Reward 设计研究

> 系统性研究：在 PPO 训练中，reward 函数的设计如何塑造小模型的推理行为。
>
> **基座模型**：Qwen2.5-3B · **任务**：Countdown · **框架**：veRL + PPO · **硬件**：4 × NVIDIA L40S

---

## 这个仓库里有什么

本文件夹包含**所有 reward 函数、PPO trainer 入口、PRM 代码、启动脚本、评估工具**。原始 TinyZero baseline 保留为 `v1`；之后每个版本**只改一个变量**，便于分离出各个改动的效果。

```
reward_study/
├── README.md                    <- 你在这里
├── requirements.txt             <- Python 依赖
├── rewards/                     <- 7 个 reward 函数 + 1 个对抗测试
├── trainers/                    <- PPO 入口（每个 reward 对应一个）
├── prm/                         <- Process Reward Model（仅 v3 使用）
├── scripts/                     <- 启动每个实验的 shell 脚本
└── eval/                        <- 统一 exact-match 评估（vLLM greedy 解码）
```

---

## 实验全景速览

| 代号 | 一句话意图 | 相对上一版的 reward 改动 | 关键文件 |
|---|---|---|---|
| **v1** | TinyZero 原版（离散 outcome reward） | — | `rewards/countdown_v1.py` |
| **v2** | 连续 outcome reward（接近度分） | 加 proximity 0~0.4 | `rewards/countdown_v2.py` |
| **v3** | v2 + Process Reward Model（步级稠密） | `<think>` 每步打分 | `rewards/countdown_v2.py` + `prm/` |
| **v4a** | 数字使用分权重 ×3（0.05 → 0.15） | 单个标量改动 | `rewards/countdown_v4a.py` |
| **v4b** | 数字使用分权重 ×6（0.05 → 0.30） | 单个标量改动 | `rewards/countdown_v4b.py` |
| **v5b** | 新增乘除法奖励（+0.5，含 reward hacking 漏洞） | 有意义 `*`/`/` +0.5 | `rewards/countdown_v5b.py` |
| **v5c** | v5b + 硬约束加固（防 hacking） | +硬约束层 + mul bonus 锚定 `numbers_valid` | `rewards/countdown_v5c.py` |
| **v2-no-proximity** | 因果消融：从 v2 移除 proximity | proximity_score = 0 | `rewards/countdown_v2_no_proximity.py` |

---

## 快速上手：复现一个实验

### 前置要求

1. **veRL 框架**：`git clone https://github.com/volcengine/verl && cd verl && pip install -e .`
2. **Qwen2.5-3B 权重** → 下到本地路径（下面 `BASE_MODEL`）
3. **Countdown-Tasks-3to4 数据集**：先 `huggingface-cli download Jiayi-Pan/Countdown-Tasks-3to4 --repo-type dataset`，再用 veRL 自带的预处理脚本转 parquet：
   ```bash
   python verl/examples/data_preprocess/countdown.py --local_dir $DATA_DIR
   ```
   预处理后会得到 `train.parquet`、`test.parquet`，列名 `prompt`、`reward_model`（含 `ground_truth.target` 和 `ground_truth.numbers`）
4. **GPU**：4 张 ≥24GB（实测 4×L40S 46GB）
5. **（仅 v3 / v3_random 需要）训一个 PRM checkpoint**：
   ```bash
   cd reward_study/prm
   python generate_data_v2.py    # 生成训练数据 → data/train_v2.jsonl
   python train_v2.py            # 训出 checkpoints_v1_improved/best_model.pt（约 30 分钟）
   ```
   v1/v2/v4/v5 系列都不需要 PRM，跳过这一步即可。

### 运行一个实验（以 v4a 为例）

```bash
# 0. 装依赖
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# 1. 把 rewards/ 和 trainers/ 注入到你的 veRL checkout
bash install_into_verl.sh /path/to/verl

# 2. 编辑启动脚本——设置 BASE_MODEL / DATA_DIR / CUDA_VISIBLE_DEVICES
vim scripts/run_v4a.sh

# 3. 启动
bash scripts/run_v4a.sh
```

每个 trainer 入口都是对 veRL `RayPPOTrainer` 的极小封装——实验间**唯一变的**是导入哪个 `countdown_vXX.compute_score`（下一节会解释为什么能这么简单）。

---

## 目录逐一说明

### `rewards/` — 7 个 reward 函数变体 + 1 个对抗测试

所有 reward 文件共享**相同的函数签名**，即可互相 drop-in 替换：

```python
def compute_score(solution_str, ground_truth, method='strict',
                  format_score=0.1, score=1.) -> float:
    """
    对从 solution_str 提取的最终等式打分，返回标量 reward。
    ground_truth = {'target': int, 'numbers': List[int]}
    """
```

每个文件的改动要点：

| 文件 | 改了什么（相对上一版） | 为什么这么改 |
|------|-----------------------|------------|
| `countdown_v1.py` | — | TinyZero 原版：3 档离散（0 / 0.1 / 1.0） |
| `countdown_v2.py` | 新增 proximity、format、数字使用、complexity 各项 | 连续梯度信号；"越接近目标越好" |
| `countdown_v4a.py` | 数字使用分权重 0.05 → **0.15** | 81.66% 错误答案是"算对数值但用错数字"→ 加强数字约束 |
| `countdown_v4b.py` | 数字使用分权重 0.05 → **0.30** | v4a 的激进版（结果塌缩） |
| `countdown_v5b.py` | 新增 +0.5 乘除法奖励（对错都给，数字合法度折扣）| 尝试恢复乘除法探索 |
| `countdown_v5c.py` | v5b + 硬约束"不许出现题外数字" + mul bonus 必须 `numbers_valid=True` | 修复 v5b 的 hacking 漏洞 |
| `countdown_v2_no_proximity.py` | v2 的 proximity 直接设为 0 | 因果消融：proximity 真的压制乘除法了吗？ |
| `test_countdown_v5c.py` | 对抗测试套件 | 训练前必跑：验证所有 hacking 模板得分 ≤ 0.1 |

**重要教训**：v5b 看起来 val_score 有 0.924，但真实正确率是 0%——模型学会了 `(given_a - given_b) / given_c + TARGET` 这个 reward hacking 模板。**训练前务必跑 `test_countdown_vXX.py` 式的对抗测试**。

### `trainers/` — PPO 入口

每个文件约 180 行，大部分是 veRL 的模板代码；实验间的差异只有 2 行：
- `from verl.utils.reward_score import countdown_vXX`
- `_select_rm_score_fn()` 里 `return countdown_vXX.compute_score`

**新增实验时**，复制最接近的 trainer，改这两行即可。

### `prm/` — Process Reward Model（仅 v3 使用）

PRM 是一个手搓的小 Transformer，给 `<think>` 里每个中间算术步骤打分。仅在 v3 实验中使用。

| 文件 | 作用 |
|---|---|
| `model.py` | v1 PRM 架构——字符级 Transformer，约 812K 参数 |
| `model_v2.py` | v2 PRM（带 NumberValueEncoder）——**失败**，保留用于复现失败故事 |
| `train.py` | 训练 PRM v1 |
| `train_v2.py` | 训练 PRM v2 / v1_improved（同样的 trainer，不同超参） |
| `generate_data.py` | v1 数据生成器：简单随机扰动负样本 |
| `generate_data_v2.py` | v2 数据生成器：6 种困难负样本（off-by-1、off-by-10、符号错等） |
| `tokenizer.py` / `tokenizer_v2.py` | 字符级 tokenizer |
| `integrate.py` | PRM → PPO 奖励集成：从 `<think>` 提取 `a op b = c` 步骤，在对应 token 注入 `prm_weight × (2p-1)` |
| `integrate_random.py` | 随机权重 PRM 变体，用于消融（"PRM 信号是真的还是噪声？"） |

**PRM 训练**：跑 `python prm/generate_data_v2.py && python prm/train_v2.py` 即可得到 v3 用的 checkpoint（`checkpoints_v1_improved/best_model.pt`）。`model_v2.py` + `train_v2.py` 的 `--use_value_encoder` 路径是失败实验，保留代码但默认不启用。

### `scripts/` — 启动脚本

每个 `run_vXX.sh` 是自包含的 bash 脚本：
1. 激活 `verl` conda 环境
2. 设置 `CUDA_VISIBLE_DEVICES`（改成你的 GPU 编号）
3. 导出 `BASE_MODEL`、`DATA_DIR`、`ROLLOUT_TP_SIZE`
4. 启动 `python3 -m verl.trainer.main_ppo_vXX` 并传入所有 PPO 超参

**适配你的环境**，编辑脚本顶部的 export：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export BASE_MODEL=/path/to/Qwen2.5-3B
export DATA_DIR=/path/to/countdown
export ROLLOUT_TP_SIZE=2
```

其他 PPO 超参（lr、KL 系数、batch size 等）实验间应保持一致——**唯一变量是 reward 函数**。

### `eval/` — 统一 exact-match 评估

**为什么必须独立评估**：训练时记录的 `val/test_score` 是**平均奖励分数**，不是 exact-match 正确率。由于不同实验的 reward 函数上限不同（v2 max=1.2, v5b max=1.7, v2-no-proximity max=0.8），**`val_score` 跨实验不可直接比较**。

`eval_unified.py` 用 vLLM greedy 解码为每个测试 prompt 生成一个响应，然后用严格规则分类：

```
exact_match:                严格正确——数字用对 AND 结果 = target
right_value_wrong_numbers:  结果 = target 但用了题外数字或少用
wrong_value:                结果 ≠ target
no_equation:                提取不出 <answer>...</answer>
```

这样得到**跨实验可比的单一指标**（exact-match 正确率）。

用法：
```bash
python eval/eval_unified.py \
    --model /path/to/checkpoint/actor/global_step_1600 \
    --tag v4a \
    --test_parquet /path/to/countdown/test.parquet \
    --out_dir ./eval_results \
    --tp_size 1 \
    --gpu_mem 0.5
```

输出：
- `eval_results/v4a_summary.json` — 汇总指标
- `eval_results/v4a_records.jsonl` — 每个测试样本一行，包含提取的等式、结果、分类

---

## 如何设计一个新的 reward 变体

如果你想加自己的实验（假设叫 `vX`）：

1. **先想清楚再写代码**，写下：
   - 这个 reward 测的是什么假设？
   - 相对最接近的现有版本改了什么？
   - 什么结果算支持假设？什么算反驳？

2. **复制最接近的 reward**：
   ```bash
   cp rewards/countdown_v2.py rewards/countdown_vX.py
   ```
   做最小改动。

3. **训练前写对抗测试**（参考 `test_countdown_v5c.py`）：
   - 列至少 10 个可能的 hacking 模板
   - 断言它们都得分 ≤ 0.1，再启动 PPO
   - **这一步当初能省下 27 小时的 v5b 训练浪费**

4. **复制 trainer**：
   ```bash
   cp trainers/main_ppo_v2.py trainers/main_ppo_vX.py
   # 改 2 行：import 和 score function 派发
   ```

5. **复制启动脚本**：
   ```bash
   cp scripts/run_v2.sh scripts/run_vX.sh
   # 改 EXPERIMENT_NAME 和 main_ppo_vX 入口名
   ```

6. **跑完后评估**：
   ```bash
   bash scripts/run_vX.sh
   # 训练完成后：
   python eval/eval_unified.py --model .../global_step_1920 --tag vX
   ```

---

## 核心教训（如果没时间看别的就看这节）

1. **`val_score` ≠ 正确率**。不同实验的 reward 函数不同，原始 reward 数字跨实验不可比。务必用 `eval/eval_unified.py` 做 exact-match 评估。

2. **最大收益来自单个标量改动**。v4a 把 `num_usage_weight: 0.05 → 0.15` 带来 +14.17pp——比第 1、2 章所有架构改动加起来还多。

3. **Reward hacking 极易意外产生**。v5b 加了一个看似合理的 +0.5 乘除法奖励，模型立刻收敛到 `(a-b)/c + target` 的 hacking 模板，真实正确率 0%。**训练前必跑对抗测试**。

4. **架构归纳偏置必须匹配任务判别粒度**。PRM v2 的 NumberValueEncoder 让 44 和 45 在 embedding 空间接近——恰恰破坏了模型识别 off-by-1 错误的能力。"看起来更聪明"的设计可能反向有害。

5. **有时正确的修复是收紧约束而不是加激励**。v4a 只是加紧数字约束，就让乘除法尝试率从 0% 回到 6.64%——没加任何乘除法专属激励。**约束搜索空间比指导搜索方向更有效**。

---

## 引用与致谢

- 基础项目：[Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- 训练框架：[veRL (Volcano Engine RL)](https://github.com/volcengine/verl)
- 基座模型：[Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)
- 数据集：[Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)

---

## 文件一览

```
reward_study/
├── README.md                                        (本文件)
├── requirements.txt                                  (Python 依赖)
├── rewards/
│   ├── countdown_v1.py                              (baseline — 离散 ORM)
│   ├── countdown_v2.py                              (连续 ORM)
│   ├── countdown_v4a.py                             (num_weight=0.15)
│   ├── countdown_v4b.py                             (num_weight=0.30)
│   ├── countdown_v5b.py                             (朴素 mul bonus，含 hacking)
│   ├── countdown_v5c.py                             (加固 mul bonus)
│   ├── countdown_v2_no_proximity.py                 (proximity 消融)
│   └── test_countdown_v5c.py                        (对抗测试)
├── trainers/
│   ├── main_ppo_v2.py  v3.py  v3_random.py  v4a.py  v4b.py  v5b.py  v5c.py  v2_no_proximity.py
├── prm/
│   ├── model.py  model_v2.py
│   ├── train.py  train_v2.py
│   ├── generate_data.py  generate_data_v2.py        (简单 vs 困难负样本)
│   ├── tokenizer.py  tokenizer_v2.py
│   ├── integrate.py                                  (PRM → PPO 奖励注入)
│   └── integrate_random.py                          (随机权重 PRM 消融)
├── scripts/
│   ├── run_v4a.sh  run_v4b.sh                       (v4 系列)
│   ├── run_v5b.sh  run_v5c.sh                       (v5 系列)
│   ├── run_v2_no_proximity.sh                       (proximity 消融)
│   └── run_v3_random.sh                             (随机 PRM 消融)
└── eval/
    └── eval_unified.py                               (exact-match vLLM 评估)
```
