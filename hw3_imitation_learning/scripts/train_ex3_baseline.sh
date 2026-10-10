#!/usr/bin/env bash
# Run from hw3_imitation_learning after collecting and preprocessing demonstrations.
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo 'Usage: bash scripts/train_ex3_baseline.sh <processed.zarr> [output.pt] [epochs]' >&2
  exit 2
fi
.venv/bin/python scripts/train.py \
  --policy multitask --zarr "$1" \
  --state-keys state_ee_xyz state_gripper 'state_joints[:5]' \
    'original_pos_cube_red[:3]' 'original_pos_cube_green[:3]' \
    'original_pos_cube_blue[:3]' state_goal goal_pos \
  --action-keys action_ee_xyz action_gripper \
  --chunk-size 4 --d-model 512 --depth 3 --threads 2 \
  --epochs "${3:-100}" \
  --output "${2:-checkpoints/multi_cube/ex3_baseline.pt}"
