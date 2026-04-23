#!/bin/bash
#SBATCH --job-name=tinyzero-eval-cs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/eval-cs-%j.out
#SBATCH --error=logs/eval-cs-%j.err

source ~/.bashrc
conda activate tinyzero
cd ~/TinyZero

BASE=checkpoints/TinyZero
mkdir -p eval_results

CHECKPOINTS=(
    "ppo-1.5b-coldstart/actor/global_step_50"
    "ppo-1.5b-coldstart/actor/global_step_150"
    "ppo-1.5b-coldstart/actor/global_step_250"
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
