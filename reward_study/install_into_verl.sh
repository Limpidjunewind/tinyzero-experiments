#!/usr/bin/env bash
# Copy this folder's reward functions and PPO trainer entry points into an
# already-installed veRL checkout, so they become callable as
# `verl.utils.reward_score.countdown_vXX` and `verl.trainer.main_ppo_vXX`.
#
# Usage:
#   bash install_into_verl.sh /path/to/verl
#
# After this runs, you still need to:
#   1. Download Qwen2.5-3B and Countdown-Tasks-3to4
#   2. Edit BASE_MODEL / DATA_DIR / CUDA_VISIBLE_DEVICES at the top of scripts/run_*.sh
#   3. (For v3 only) train a PRM checkpoint via prm/train_v2.py first
set -euo pipefail

VERL_ROOT="${1:-}"
if [[ -z "$VERL_ROOT" ]]; then
    echo "usage: bash install_into_verl.sh /path/to/verl" >&2
    exit 1
fi
if [[ ! -d "$VERL_ROOT/verl/utils/reward_score" ]] || [[ ! -d "$VERL_ROOT/verl/trainer" ]]; then
    echo "error: $VERL_ROOT does not look like a veRL checkout" >&2
    exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "Copying reward functions -> $VERL_ROOT/verl/utils/reward_score/"
cp "$HERE"/rewards/countdown_*.py "$VERL_ROOT/verl/utils/reward_score/"

echo "Copying PPO trainer entry points -> $VERL_ROOT/verl/trainer/"
cp "$HERE"/trainers/main_ppo_*.py "$VERL_ROOT/verl/trainer/"

echo "Done. Launch experiments via:  bash $HERE/scripts/run_vXX.sh"
