#!/bin/bash
# PPO training job. Submit from the project folder on Unity:
#
#   sbatch cluster/train.sbatch                                  # one run, defaults below
#   sbatch --export=ALL,MAX_ITERS=3000 cluster/train.sbatch      # override settings
#   sbatch --array=0-3 cluster/train.sbatch                      # 4 runs with seeds 0..3 (one GPU each)
#   sbatch cluster/train.sbatch env.thrust_max=10.0              # extra args go straight to `isaaclab train`
#
# The job resumes from its own latest checkpoint if Slurm preempts and requeues it.
#SBATCH --job-name=vertigo-train
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=sm_75|sm_80|sm_86|sm_89|sm_90
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=cluster/slurm_out/train-%j.out
#
# Partition notes (Unity):
#   gpu-preempt          most GPUs, starts fastest, may be killed after 2 h (job auto-resumes)  <- default
#   gpu                  same GPUs, never preempted, up to 5 days; queue can be longer
#   gpu-highvram         A100/L40S/H100; max 8 GPUs per user. Not needed: this task uses ~1.5 GB of GPU memory.

set -euo pipefail
source cluster/unity.env
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$(dirname "$PWD")/.uv-cache" UV_PYTHON_INSTALL_DIR="$(dirname "$PWD")/.uv-python"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITERS=${MAX_ITERS:-1500}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-42}}
RUN_NAME="job${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_seed${SEED}"

echo "node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) | run $RUN_NAME"

# resume after preemption: newest checkpoint from an earlier attempt of this same run
RESUME_ARGS=()
ITERS_LEFT=$MAX_ITERS
LATEST=$(ls -v logs/rsl_rl/vertigo_wall_climb/*_"$RUN_NAME"/model_*.pt 2>/dev/null | tail -1 || true)
if [[ -n "$LATEST" ]]; then
    DONE_ITERS=$(basename "$LATEST" .pt | sed 's/model_//')
    ITERS_LEFT=$(( MAX_ITERS - DONE_ITERS ))
    echo "Resuming from $LATEST ($ITERS_LEFT iterations left)"
    RESUME_ARGS=(--checkpoint "$LATEST")
    if (( ITERS_LEFT <= 0 )); then echo "Already finished."; exit 0; fi
fi

uv run --no-sync isaaclab train \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$ITERS_LEFT" \
    --seed "$SEED" \
    --run_name "$RUN_NAME" \
    "${RESUME_ARGS[@]}" \
    "$@"
