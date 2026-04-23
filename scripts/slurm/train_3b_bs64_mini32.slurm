#!/bin/bash
#SBATCH --job-name=tinyzero-bs64-mini32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:2
#SBATCH --time=3:00:00
#SBATCH --mem=120G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

cd ~/TinyZero
mkdir -p logs

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Set 1: 固定 updates/rollout=2，缩小 train_batch
# train_batch=64, mini_batch=32, updates/rollout=2
/mnt/scratch/j/jingwenl/miniforge3/envs/tinyzero/bin/python3 -m verl.trainer.main_ppo \
data.train_files=./data/countdown/train.parquet \
data.val_files=./data/countdown/test.parquet \
data.train_batch_size=64 \
data.val_batch_size=1312 \
data.max_prompt_length=256 \
data.max_response_length=512 \
actor_rollout_ref.model.path=Qwen/Qwen2.5-3B \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
critic.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.ppo_micro_batch_size=4 \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
critic.optim.lr=1e-5 \
critic.model.path=Qwen/Qwen2.5-3B \
critic.ppo_mini_batch_size=32 \
critic.ppo_micro_batch_size=4 \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=2 \
trainer.nnodes=1 \
trainer.save_freq=50 \
trainer.test_freq=25 \
trainer.project_name=TinyZero \
trainer.experiment_name=ppo-3b-bs64-mini32 \
trainer.total_epochs=15 2>&1 | tee logs/3b_bs64_mini32.log
