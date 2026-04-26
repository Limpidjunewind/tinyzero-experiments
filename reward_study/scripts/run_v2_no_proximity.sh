#!/bin/bash
# Experiment v6: Causal ablation of v2 proximity score (v2-no-proximity).
#
# Hypothesis (C3 from report_0413/experiment_report.md):
#   v2's continuous proximity score CAUSED the drop in mul/div attempt
#   rate from v1's 4.4% to v2's 1.5%, by creating an "add/sub highway"
#   that gave continuous gradient feedback to add/sub but not to mul/div.
#
# Test: train v2 with proximity_score = 0 (everything else identical).
#   Predicted: mul/div rate rises toward v1's ~4.4%, accuracy drops < 50%.
#
# Control: v2 (1.5% mul rate, 60.4% accuracy) — comparison baseline.
# Reference: v1 (4.4% mul rate, 34.3% accuracy) — what we'd see if proximity
#   was the only thing changed v1 → v2.
#
# Using GPU 4-7 (v5b is running on GPU 0-3).

source activate verl 2>/dev/null || conda activate verl

export CUDA_VISIBLE_DEVICES=4,5,6,7
export N_GPUS=4
export BASE_MODEL=/data/fangda/models/Qwen2.5-3B
export DATA_DIR=/data/fangda/data/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-v2-no-proximity
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/fangda/tinyzero

python3 -m verl.trainer.main_ppo_v2_no_proximity \
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
trainer.save_freq=400 \
trainer.test_freq=50 \
trainer.project_name=TinyZero \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=15 2>&1 | tee /data/fangda/tinyzero/experiments/v6_proximity_ablation/train_v2_no_proximity.log
