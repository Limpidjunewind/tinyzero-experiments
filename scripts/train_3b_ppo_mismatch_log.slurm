#!/bin/bash
#SBATCH --job-name=ppo3b-mismatch-log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100-96:2
#SBATCH --time=03:00:00
#SBATCH --output=/home/j/jingwenl/logs/slurm-%j.out
#SBATCH --error=/home/j/jingwenl/logs/slurm-%j.err

# Short PPO run (50 steps) with mismatch instrumentation.
# Goal: measure the real FSDP-vs-vLLM log_prob divergence on actual training path.
# NOT meant to train to convergence.

set -e
source ~/.bashrc
conda activate tinyzero

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_PROJECT=tinyzero-mismatch
export WANDB_NAME=ppo3b-mismatch-log-${SLURM_JOB_ID}

cd ~/TinyZero

# reuse existing countdown PPO 3B config; only shorten training
python3 -m verl.trainer.main_ppo \
  data.train_files=~/TinyZero/data/countdown/train.parquet \
  data.val_files=~/TinyZero/data/countdown/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=256 \
  data.max_prompt_length=256 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.n=1 \
  critic.optim.lr=1e-5 \
  critic.model.path=Qwen/Qwen2.5-3B \
  critic.ppo_micro_batch_size=8 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['console','wandb'] \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.total_training_steps=50 \
  trainer.project_name=tinyzero-mismatch \
  trainer.experiment_name=ppo3b-mismatch-log
