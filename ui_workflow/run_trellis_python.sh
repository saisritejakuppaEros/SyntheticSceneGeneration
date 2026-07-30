#!/usr/bin/env bash
set -eo pipefail

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"

source /devwork/MiniConda/miniconda3/etc/profile.d/conda.sh
conda activate /home/parth_h200/parth/TRELLIS.2/trellis
source /home/parth_h200/parth/TRELLIS.2/env_cuda.sh

export HF_HOME="${TRELLIS_MODEL_CACHE:-/home/parth_h200/parth/TRELLIS.2/models}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TRELLIS_ROOT="${TRELLIS_ROOT:-/home/parth_h200/parth/TRELLIS.2}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

exec python "$@"
