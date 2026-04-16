#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
# Full test set can be slow on shared filesystem; allow headroom.
#SBATCH --time=12:00:00

#SBATCH -J "densenet_infer"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=densenet-infer-%j.out
#SBATCH --error=densenet-infer-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/dense_net/inference"

DEFAULT_CHECKPOINT="/resnick/groups/CS156b/from_central/2026/haa/dense_net_runs/densenet121_run1/best_model.pt"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT}}"
TEST_CSV="${TEST_CSV:-/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed/test_manifest_preprocessed.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/dense_net/inference/submission.csv}"

mkdir -p "${REPO_ROOT}/dense_net/logs"

cd "${SCRIPT_DIR}"

CONDA_ENV_NAME="${CONDA_ENV:-cs156b}"
if [[ -n "${PYTHON_EXE:-}" && -x "${PYTHON_EXE}" ]]; then
  PYTHON_BIN=("${PYTHON_EXE}")
elif [[ -x "${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python")
elif [[ -x "${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python")
elif [[ -x "/resnick/groups/CS156b/from_central/2026/haa/asqin/miniconda3/envs/cs156b_ready/bin/python" ]]; then
  PYTHON_BIN=("/resnick/groups/CS156b/from_central/2026/haa/asqin/miniconda3/envs/cs156b_ready/bin/python")
elif [[ -x "/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/${CONDA_ENV_NAME}/bin/python")
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

"${PYTHON_BIN[@]}" densenet_inference.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --csv "${TEST_CSV}" \
  --output "${OUTPUT_PATH}" \
  --batch_size 32 \
  --num_workers 8
