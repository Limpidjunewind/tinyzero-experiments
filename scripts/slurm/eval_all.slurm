#!/bin/bash
#SBATCH --job-name=tinyzero-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

source ~/.bashrc
conda activate tinyzero
cd ~/TinyZero

BASE=checkpoints/TinyZero
mkdir -p eval_results

CHECKPOINTS=(
    "ppo-3b/actor/global_step_150"
    "ppo-3b/actor/global_step_250"
    "ppo-3b/actor/global_step_300"
    "ppo-1.5b/actor/global_step_350"
    "ppo-1.5b/actor/global_step_650"
    "grpo-1.5b/actor/global_step_150"
)

for CKPT in "${CHECKPOINTS[@]}"; do
    NAME=$(echo $CKPT | tr '/' '-')
    echo "=== Evaluating $CKPT ==="
    python eval_checkpoints.py \
        --checkpoint $BASE/$CKPT \
        --output eval_results/${NAME}.json \
        --n_samples 100
done

echo "=== All done ==="
