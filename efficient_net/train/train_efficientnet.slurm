#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --partition=gpu
# Give plenty of headroom.
#SBATCH --time=2-00:00:00

#SBATCH -J "effnet_train"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
# Slurm stdout/stderr (job-level). Per-epoch metrics and full console copy: results/.../runs/<run>/train.log
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/results/efficient_net/slurm_logs/effnet-train-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/results/efficient_net/slurm_logs/effnet-train-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/efficient_net/train"
OUT_ROOT="${OUT_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data}"
RESULTS_ROOT="${RESULTS_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/results/efficient_net}"

mkdir -p "${REPO_ROOT}/efficient_net/logs"
mkdir -p "${RESULTS_ROOT}/runs" "${RESULTS_ROOT}/slurm_logs"

cd "${SCRIPT_DIR}"

# Same resolution as data_preprocessing/efficientnet_data_prep.slurm (compute nodes often lack conda on PATH).
CONDA_ENV_NAME="${CONDA_ENV:-cs156b-efficientnet}"
if [[ -n "${PYTHON_EXE:-}" && -x "${PYTHON_EXE}" ]]; then
  PYTHON_BIN=("${PYTHON_EXE}")
elif [[ -x "${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python")
elif [[ -x "${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python")
elif [[ -x "/resnick/groups/CS156b/from_central/2026/haa/askumar/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("/resnick/groups/CS156b/from_central/2026/haa/askumar/miniconda3/envs/${CONDA_ENV_NAME}/bin/python")
else
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
      echo "ERROR: Could not resolve Python for env '${CONDA_ENV_NAME}'. Set PYTHON_EXE."
      exit 1
    fi
  else
    echo "ERROR: Set PYTHON_EXE to your env's python (with torch installed)."
    exit 1
  fi
fi

echo "Using Python: ${PYTHON_BIN[*]}"
"${PYTHON_BIN[@]}" -c "import sys; import torch; print('Python:', sys.version); print('torch:', torch.__version__)"

"${PYTHON_BIN[@]}" train_efficientnet.py \
  --train_csv "${OUT_ROOT}/manifests_preprocessed/train_manifest_preprocessed.csv" \
  --val_csv "${OUT_ROOT}/manifests_preprocessed/val_manifest_preprocessed.csv" \
  --runs_base "${RESULTS_ROOT}/runs" \
  --model_name efficientnet_b3 \
  --uncertain_policy uzeros \
  --use_pos_weight \
  --image_size 224 \
  --epochs 15 \
  --batch_size 24 \
  --lr 1e-4 \
  --num_workers 16
