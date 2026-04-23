#!/bin/bash
#SBATCH --job-name=logit-bs64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/home/j/jingwenl/TinyZero/logs/logit_bs64_%j.out
#SBATCH --error=/home/j/jingwenl/TinyZero/logs/logit_bs64_%j.err

cd ~/TinyZero

/mnt/scratch/j/jingwenl/miniforge3/envs/tinyzero/bin/python3 logit_analysis.py \
    --step50  checkpoints/TinyZero/ppo-3b-bs64-mini32/actor/global_step_50 \
    --step100 checkpoints/TinyZero/ppo-3b-bs64-mini32/actor/global_step_500 \
    --device cuda
