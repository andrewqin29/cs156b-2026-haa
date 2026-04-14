#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=23:59:00

#SBATCH -J "effnet_data_prep"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=data-prep-%j.out
#SBATCH --error=data-prep-%j.err

set -euo pipefail

# ---- Paths ----
# On some clusters, Slurm runs a spooled copy of this script from /var/spool,
# so do NOT derive paths from BASH_SOURCE. Use the submission directory.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/efficient_net/data_preprocessing"
OUT_ROOT="${OUT_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_data}"
LOG_DIR="${REPO_ROOT}/efficient_net/logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_ROOT}"

cd "${SCRIPT_DIR}"

# ---- Python executable ----
# Option 1: set PYTHON_BIN before sbatch
#   sbatch --export=PYTHON_BIN=/path/to/python efficientnet_data_prep.slurm
# Option 2: default to python on PATH
PYTHON_BIN=${PYTHON_BIN:-python}

# ---- Conda (batch-safe) ----
# In Slurm batch mode, `conda activate` can fail unless the shell was initialized
# (i.e., `conda init`). Prefer `conda run`, which does not require shell init.
if [[ -n "${CONDA_SH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
fi

CONDA_ENV_NAME="${CONDA_ENV:-cs156b-efficientnet}"

if command -v conda >/dev/null 2>&1; then
  # If conda run works, use it for all python invocations.
  if conda run -n "${CONDA_ENV_NAME}" python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    PYTHON_BIN=(conda run -n "${CONDA_ENV_NAME}" python)
  else
    echo "WARNING: conda found but 'conda run' failed; running with ${PYTHON_BIN} on PATH"
    PYTHON_BIN=("${PYTHON_BIN}")
  fi
else
  PYTHON_BIN=("${PYTHON_BIN}")
fi

echo "Using Python: ${PYTHON_BIN[*]}"

# ---- Preflight checks ----
"${PYTHON_BIN[@]}" -c "import sys; print('Python:', sys.version); import pandas, numpy, PIL; print('deps ok')"

for p in \
  /resnick/groups/CS156b/from_central/data/train \
  /resnick/groups/CS156b/from_central/data/test \
  /resnick/groups/CS156b/from_central/data/student_labels/train2023.csv \
  /resnick/groups/CS156b/from_central/data/student_labels/test_ids.csv; do
  if [[ ! -e "$p" ]]; then
    echo "Missing required path: $p"
    exit 1
  fi
done

# 1) Build train/val/test manifests (CSVs)
"${PYTHON_BIN[@]}" build_efficientnet_manifests.py \
  --output_root "${OUT_ROOT}" \
  --val_split 0.15 \
  --seed 42

# 2) Build preprocessed image cache + updated manifests with preprocessed_path
"${PYTHON_BIN[@]}" prepare_efficientnet_images.py \
  --manifest_root "${OUT_ROOT}/manifests" \
  --output_root "${OUT_ROOT}" \
  --image_size 224 \
  --image_format jpg \
  --jpg_quality 95

echo "Data prep complete."
echo "Base output root: ${OUT_ROOT}"
echo "Manifests: ${OUT_ROOT}/manifests"
echo "Preprocessed manifests: ${OUT_ROOT}/manifests_preprocessed"
echo "Cached images: ${OUT_ROOT}/preprocessed_images"
