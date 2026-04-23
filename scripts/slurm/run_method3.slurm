#!/bin/bash
#SBATCH --job-name=method3-excl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=/home/j/jingwenl/TinyZero/logs/method3_%j.out
#SBATCH --error=/home/j/jingwenl/TinyZero/logs/method3_%j.err

cd ~/TinyZero
/mnt/scratch/j/jingwenl/miniforge3/envs/tinyzero/bin/python3 method3_exclamation_prob.py
