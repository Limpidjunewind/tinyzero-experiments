#!/bin/bash
# Experiment v5c: Hacking-hardened mul/div incentive.
#
# Based on v5b's failure (learned reward hacking template (a-b)/c + target),
# v5c adds hard constraints + mul bonus conditional on numbers_valid.
#
# Adversarial tests passed: 10 hacking templates all score 0.
#
# Running on GPU 0-3 (v2-no-proximity uses 4-7).

source activate verl 2>/dev/null || conda activate verl

export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export BASE_MODEL=/data/fangda/models/Qwen2.5-3B
export DATA_DIR=/data/fangda/data/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-v5c-mul-hardened
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/fangda/tinyzero

python3 -m verl.trainer.main_ppo_v5c \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=512 \
data.max_prompt_length=256 \
data.max_response_length=1024 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size=4 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_micro_batch_size=4 \
critic.model.enable_gradient_checkpointing=True \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=100 \
trainer.test_freq=25 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee /data/fangda/tinyzero/experiments/v5_mul_incentive/train_v5c.log
