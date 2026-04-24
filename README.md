# TinyZero Experiments

A systematic reproduction and deep-dive analysis of [DeepSeek-R1-Zero](https://github.com/deepseek-ai/DeepSeek-R1) on Countdown tasks, built on [TinyZero](https://github.com/Jiayi-Pan/TinyZero) and [veRL](https://github.com/volcengine/verl).

The goal was not just to run the numbers — it was to understand *why* training succeeds or fails, and to chase down the root cause when it doesn't.

---

## 📊 Interactive Experiment Reports

| | |
|---|---|
| **[Part I: Scale · Batch · Entropy Collapse](https://limpidjunewind.github.io/tinyzero-experiments/webshowcase/chapter1.html)** | Does model size matter? What triggers the Aha Moment? Why do all runs crash at bs=256? |
| **[Part II: Reward Function Design](https://limpidjunewind.github.io/tinyzero-experiments/webshowcase/chapter2.html)** | Sparse reward is the bottleneck — three reward designs, one variable, two surprises |

---

## Key Findings

**1. Model scale sets a hard floor on reasoning emergence**

0.5B never learns to reason — reward is near-uniform, gradient signal is too weak. 1.5B shows a weak, atypical Aha signal. 3B is the minimum for stable emergence on Countdown. Running at `batch=64` (vs. the paper's `batch=256`) due to GPU constraints is the structural reason our baseline underperforms the original.

**2. Batch size gates the Aha Moment**

Fixing the model at Qwen2.5-3B, `batch=128 / mini=64` triggers a clean Aha Moment (score rebounds at step ~75); `batch=64` does not. Larger batches reduce gradient noise, giving the policy a cleaner signal to extract reasoning behavior from.

**3. Entropy collapse is a train-inference mismatch, not a hyperparameter problem**

All three `bs=256` configs crashed mid-training: response floods to 512 identical `!` tokens, score drops to zero, never recovers. Entropy_coeff ablations (`0.001 / 0.01 / 0.05`) delayed but did not eliminate crashes — ruling out entropy collapse as root cause. Root cause: **veRL's hybrid engine discards vLLM's native log_probs and recomputes them with FSDP, causing π_old ≠ actual sampling distribution** (measured ratio peaks up to 1.99×). The fix — passing vLLM's native log_probs directly — eliminates crashes over a 280-step observation window with no loss in final score (~0.60).

**4. Reward balance is a critical design parameter**

The original sparse reward (exact-match only) systematically suppresses multiplication and division — the model learns to avoid operators that are harder to verify. Redesigning the reward to add number-usage weighting and operator diversity incentives changes *what* the model learns, not just *how fast*.

---

## Repo Structure

```
webshowcase/          Interactive visual presentations (open in browser)
  chapter1.html         Part I: Scale, Batch, Entropy Collapse
  chapter2.html         Part II: Reward Function Design

reports/              Detailed experiment notes
  4.17 - ppo_countdown_scale_ablation_0.5B_1.5B_3B.md
  4.17 - cold_start_on_1.5B.md
  4.20 - batch_size_aha_moment_ablation.md
  4.20 - ppo_3b_crash_and_ablations.md   (entropy_coeff & crash ablations)
  4.24 - train_infer_mismatch.md         (root cause analysis & fix)
  experiment_report_reward.md
  sft-vs-rl-practice.md

assets/               Charts and figures referenced in reports

scripts/              Analysis code
  eval_think_analysis.py     Evaluates <think> chain quality across checkpoints
  logit_analysis.py          Probes token-level log_prob distributions
  slurm/                     Training scripts for each experiment configuration

patches/
  megatron_v4.patch          Framework-level modifications to veRL
```

---

## Setup

```bash
conda create -n zero python=3.9
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 ray
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb matplotlib
```

Data prep:
```bash
python ./examples/data_preprocess/countdown.py --local_dir {path_to_dataset}
```

Training scripts are in `scripts/slurm/` — each `.slurm` file corresponds to a specific experiment configuration (batch size, entropy coeff, KL penalty, etc.). See `reports/` for the hyperparameters and results of each run. Training code is based on [TinyZero](https://github.com/Jiayi-Pan/TinyZero) — clone that repo and apply `patches/megatron_v4.patch` to reproduce the setup.

---

## Related

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero) — original repo this work builds on
- [veRL](https://github.com/volcengine/verl) — training framework
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) — paper being reproduced
- [verl #721](https://github.com/volcengine/verl/issues/721) — community report on NaN / entropy collapse root cause
