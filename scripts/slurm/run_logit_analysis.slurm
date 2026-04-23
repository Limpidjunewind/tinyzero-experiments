#!/bin/bash
#SBATCH --job-name=logit-analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/home/j/jingwenl/TinyZero/logs/logit_analysis_%j.out
#SBATCH --error=/home/j/jingwenl/TinyZero/logs/logit_analysis_%j.err

cd ~/TinyZero

/mnt/scratch/j/jingwenl/miniforge3/envs/tinyzero/bin/python3 logit_analysis.py \
    --step50  checkpoints/TinyZero/ppo-3b-bs128-mini16/actor/global_step_50 \
    --step100 checkpoints/TinyZero/ppo-3b-bs128-mini16/actor/global_step_100 \
    --device cuda
