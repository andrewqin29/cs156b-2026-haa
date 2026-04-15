#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=expansion
# Prior run hit this limit after train+val+partial test; bump so the rerun can
# finish remaining test images (existing outputs are skipped, not duplicated).
#SBATCH --time=24:00:00

#SBATCH -J "data_prep"
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
SCRIPT_DIR="${REPO_ROOT}"
OUT_ROOT="${OUT_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/front_512_data}"
LOG_DIR="${REPO_ROOT}/efficient_net/logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_ROOT}"

cd "${SCRIPT_DIR}"

# ---- Python executable (batch nodes often lack conda on PATH) ----
# Prefer explicit env interpreter so we never fall back to system python without pandas.
CONDA_ENV_NAME="${CONDA_ENV:-cs156b}"

if [[ -n "${PYTHON_EXE:-}" && -x "${PYTHON_EXE}" ]]; then
  PYTHON_BIN=("${PYTHON_EXE}")
elif [[ -x "${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python")
elif [[ -x "${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python")
elif [[ -x "/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/${CONDA_ENV_NAME}/bin/python")
else
  # Last resort: conda on PATH or after sourcing conda.sh
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  if command -v conda >/dev/null 2>&1; then
    _py="$(conda run -n "${CONDA_ENV_NAME}" python -c "import sys; print(sys.executable)" 2>/dev/null || true)"
    if [[ -n "${_py}" && -x "${_py}" ]]; then
      PYTHON_BIN=("${_py}")
    else
      echo "ERROR: Could not resolve Python for conda env '${CONDA_ENV_NAME}'."
      echo "Set PYTHON_EXE to the env's python, e.g.:"
      echo "  export PYTHON_EXE=\$HOME/.conda/envs/${CONDA_ENV_NAME}/bin/python"
      exit 1
    fi
  else
    echo "ERROR: No conda and no env python at:"
    echo "  \$HOME/miniconda3/envs/${CONDA_ENV_NAME}/bin/python"
    echo "  \$HOME/.conda/envs/${CONDA_ENV_NAME}/bin/python"
    echo "Submit with: sbatch --export=ALL,PYTHON_EXE=/path/to/env/bin/python ..."
    exit 1
  fi
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
"${PYTHON_BIN[@]}" build_manifests.py \
  --output_root "${OUT_ROOT}" \
  --val_split 0.15 \
  --seed 42

# 2) Build preprocessed image cache + updated manifests with preprocessed_path
"${PYTHON_BIN[@]}" prepare_images.py \
  --manifest_root "${OUT_ROOT}/manifests" \
  --output_root "${OUT_ROOT}" \
  --image_size 512 \
  --image_format png

echo "Data prep complete."
echo "Base output root: ${OUT_ROOT}"
echo "Manifests: ${OUT_ROOT}/manifests"
echo "Preprocessed manifests: ${OUT_ROOT}/manifests_preprocessed"
echo "Cached images: ${OUT_ROOT}/preprocessed_images"
