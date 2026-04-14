#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=08:00:00

#SBATCH -J "effnet_train"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=../logs/effnet-train-%j.out
#SBATCH --error=../logs/effnet-train-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/efficient_net/train"
OUT_ROOT="${OUT_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_data}"

mkdir -p "${REPO_ROOT}/efficient_net/logs"
mkdir -p "${OUT_ROOT}/checkpoints"

cd "${SCRIPT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}

if [[ -n "${CONDA_SH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
fi

CONDA_ENV_NAME="${CONDA_ENV:-cs156b-efficientnet}"
if command -v conda >/dev/null 2>&1 && conda run -n "${CONDA_ENV_NAME}" python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
  PYTHON_BIN=(conda run -n "${CONDA_ENV_NAME}" python)
else
  PYTHON_BIN=("${PYTHON_BIN}")
fi

echo "Using Python: ${PYTHON_BIN[*]}"

"${PYTHON_BIN[@]}" train_efficientnet.py \
  --train_csv "${OUT_ROOT}/manifests_preprocessed/train_manifest_preprocessed.csv" \
  --val_csv "${OUT_ROOT}/manifests_preprocessed/val_manifest_preprocessed.csv" \
  --output_dir "${OUT_ROOT}/checkpoints" \
  --model_name efficientnet_b0 \
  --image_size 224 \
  --epochs 15 \
  --batch_size 32 \
  --lr 1e-4 \
  --num_workers 8
