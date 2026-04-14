#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=06:00:00

#SBATCH -J "effnet_data_prep"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/efficient_net/logs/data-prep-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/efficient_net/logs/data-prep-%j.err

set -euo pipefail

# ---- Paths (edit only if your layout changes) ----
REPO_ROOT="/resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa"
SCRIPT_DIR="${REPO_ROOT}/efficient_net/data_preprocessing"
OUT_ROOT="/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data"
LOG_DIR="${REPO_ROOT}/efficient_net/logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_ROOT}"

cd "${SCRIPT_DIR}"

# ---- Python executable ----
# Option 1: set PYTHON_BIN before sbatch
#   sbatch --export=PYTHON_BIN=/path/to/python efficientnet_data_prep.slurm
# Option 2: default to python on PATH
PYTHON_BIN=${PYTHON_BIN:-python}

# Optional conda activation (if provided at submit time)
# Example:
# sbatch --export=ALL,CONDA_SH=/path/miniconda3/etc/profile.d/conda.sh,CONDA_ENV=cs156b efficientnet_data_prep.slurm
if [[ -n "${CONDA_SH:-}" && -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  PYTHON_BIN=${PYTHON_BIN:-python}
fi

echo "Using Python: ${PYTHON_BIN}"

# ---- Preflight checks ----
"${PYTHON_BIN}" -c "import sys; print('Python:', sys.version); import pandas, numpy, PIL; print('deps ok')"

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
"${PYTHON_BIN}" build_efficientnet_manifests.py \
  --output_root "${OUT_ROOT}" \
  --val_split 0.15 \
  --seed 42

# 2) Build preprocessed image cache + updated manifests with preprocessed_path
"${PYTHON_BIN}" prepare_efficientnet_images.py \
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
