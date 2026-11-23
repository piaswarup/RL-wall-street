#!/usr/bin/env bash
# Run ON UNITY (login node is fine), from inside the synced project:
#     cd $UNITY_DIR/vertigo_rl && bash cluster/setup_unity.sh
#
# One-time setup (safe to re-run): installs uv, clones Isaac Lab at the tested commit next to this project,
# and builds the Python environment (.venv). Takes ~10-20 min the first time (downloads ~6 GB).
# Isaac Sim is NOT installed: training uses the Newton physics backend, which only needs a CUDA GPU.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/unity.env"
PROJECT="$(dirname "$HERE")"
PARENT="$(dirname "$PROJECT")"

# keep big caches off the 100 GB home quota
export UV_CACHE_DIR="$PARENT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PARENT/.uv-python"
export UV_LINK_MODE=copy

# 1) uv
if ! command -v uv >/dev/null || [[ "$(uv --version | awk '{print $2}')" != "$UV_VERSION" ]]; then
    echo "==> Installing uv $UV_VERSION into ~/.local/bin"
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2) Isaac Lab source checkout, as a sibling folder (pyproject.toml points at ../IsaacLab)
if [[ ! -d "$PARENT/IsaacLab/.git" ]]; then
    echo "==> Cloning Isaac Lab"
    git clone "$ISAACLAB_REPO" "$PARENT/IsaacLab"
fi
git -C "$PARENT/IsaacLab" fetch --quiet origin
git -C "$PARENT/IsaacLab" checkout --quiet "$ISAACLAB_COMMIT"
echo "==> Isaac Lab at $(git -C "$PARENT/IsaacLab" log -1 --format='%h %s')"

# 3) optional: older NVIDIA driver -> use the CUDA 12.8 torch build
cd "$PROJECT"
if [[ "$TORCH_CUDA" != "cu130" ]]; then
    echo "==> Switching torch to the $TORCH_CUDA build"
    sed -i "s/pytorch-cu130/pytorch-$TORCH_CUDA/g; s#whl/cu130#whl/$TORCH_CUDA#g" pyproject.toml
    uv lock
fi

# 4) Python environment
echo "==> Building the Python environment (uv sync)"
if [[ "$TORCH_CUDA" == "cu130" ]]; then
    uv sync --frozen
else
    uv sync
fi
uv run --no-sync python -c "import torch, isaaclab, vertigo_rl; print('torch', torch.__version__, '| imports OK')"

mkdir -p "$PROJECT/cluster/slurm_out"
echo
echo "Setup complete. Next: sbatch cluster/check_gpu.sbatch"
