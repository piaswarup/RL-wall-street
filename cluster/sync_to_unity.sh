#!/usr/bin/env bash
# Run on your LAPTOP. Copies this project (code, robot USD, uv.lock) to Unity. Safe to re-run after every edit:
# rsync only sends what changed. Never deletes training logs on Unity.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/unity.env"
PROJECT="$(dirname "$HERE")"

if [[ "$UNITY_DIR" == *CHANGEME* ]]; then
    echo "Edit UNITY_DIR in cluster/unity.env first." >&2
    exit 1
fi

ssh "$UNITY_SSH" "mkdir -p '$UNITY_DIR/vertigo_rl'"
rsync -avz --delete \
    --exclude '.venv/' --exclude 'logs/' --exclude 'outputs/' --exclude '__pycache__/' --exclude '.git/' --exclude 'cluster/slurm_out/' \
    "$PROJECT/" "$UNITY_SSH:$UNITY_DIR/vertigo_rl/"
echo
echo "Synced to $UNITY_SSH:$UNITY_DIR/vertigo_rl"
