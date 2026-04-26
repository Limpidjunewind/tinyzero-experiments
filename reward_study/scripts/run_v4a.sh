#!/bin/bash
# Experiment v4a: Number usage weight 0.15 (up from v2's 0.05)
# Hypothesis: Increasing number usage reward weight improves accuracy
#   because the current bottleneck is constraint following (88.8% of
#   wrong answers compute correct value but use wrong numbers).
# Control: v2 (number usage weight 0.05, accuracy 60.4%)

source activate verl 2>/dev/null || conda activate verl

export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export BASE_MODEL=/data/fangda/models/Qwen2.5-3B
export DATA_DIR=/data/fangda/data/countdown
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-v4a-numweight-0.15
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data/fangda/tinyzero

python3 -m verl.trainer.main_ppo_v4a \
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
trainer.total_epochs=15 2>&1 | tee /data/fangda/tinyzero/experiments/v4_num_weight/train_v4a.log
