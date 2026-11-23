#!/bin/bash
# Quick (~5 min) test job: is the GPU usable, and does the simulation behave? Run this before any training.
#     sbatch cluster/check_gpu.sbatch        then:  cat cluster/slurm_out/check-<jobid>.out
#SBATCH --job-name=vertigo-check
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=sm_75|sm_80|sm_86|sm_89|sm_90
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/slurm_out/check-%j.out

set -euo pipefail
source cluster/unity.env
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$(dirname "$PWD")/.uv-cache" UV_PYTHON_INSTALL_DIR="$(dirname "$PWD")/.uv-python"

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [[ "$TORCH_CUDA" == "cu130" && "$DRIVER_MAJOR" -lt 580 ]]; then
    echo "!! Driver $DRIVER_MAJOR < 580 is too old for CUDA 13 torch. Set TORCH_CUDA=cu128 in cluster/unity.env"
    echo "!! and re-run: bash cluster/setup_unity.sh"
    exit 1
fi

uv run --no-sync python -c "import torch; assert torch.cuda.is_available(); print('torch sees', torch.cuda.get_device_name(0))"
uv run --no-sync python scripts/smoke_test.py --num_envs 64
