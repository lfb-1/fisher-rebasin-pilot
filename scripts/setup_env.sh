#!/bin/bash
# Setup JAX conda environment for fisher-rebasin-pilot
# Run this ONCE as a SLURM job (needs GPU node for JAX CUDA test)
# Usage: sbatch scripts/setup_env.sh

#SBATCH --job-name=jax-setup
#SBATCH --partition=sablab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot/logs/setup_%j.log

set -euo pipefail

REPO_DIR="/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot"
ENV_DIR="/share/sablab/nfs04/users/fl453/fisher-rebasin-pilot/env"

echo "=== Setting up JAX conda environment ==="
echo "Node: $(hostname)"
echo "CUDA devices: $(nvidia-smi -L)"

source ~/miniconda3/etc/profile.d/conda.sh

# Create env
conda create -y -p "$ENV_DIR" python=3.11
conda activate "$ENV_DIR"

# JAX with CUDA 12 support
pip install --upgrade "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Core dependencies (exact versions from git-re-basin requirements)
pip install \
    flax==0.8.5 \
    optax==0.2.3 \
    augmax \
    einops \
    tensorflow-cpu \
    scipy \
    pandas \
    tqdm \
    wandb \
    matplotlib \
    "tensorflow-datasets>=4.0.0"

# Verify JAX GPU
python -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('GPU available:', any(d.platform == 'gpu' for d in jax.devices()))
"

echo "=== Environment setup complete ==="
echo "Activate with: conda activate $ENV_DIR"
