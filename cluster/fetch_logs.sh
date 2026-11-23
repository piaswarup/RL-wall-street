#!/usr/bin/env bash
# Run on your LAPTOP. Pulls training logs + checkpoints from Unity into ./logs so you can view them in
# TensorBoard and replay policies locally.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/unity.env"
PROJECT="$(dirname "$HERE")"

mkdir -p "$PROJECT/logs" "$PROJECT/cluster/slurm_out"
rsync -avz "$UNITY_SSH:$UNITY_DIR/vertigo_rl/logs/" "$PROJECT/logs/"
rsync -avz "$UNITY_SSH:$UNITY_DIR/vertigo_rl/cluster/slurm_out/" "$PROJECT/cluster/slurm_out/" || true
echo
echo "Logs are in $PROJECT/logs. View with:  uv run tensorboard --logdir logs/rsl_rl"
