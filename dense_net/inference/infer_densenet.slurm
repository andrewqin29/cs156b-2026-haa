#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

#SBATCH -J "densenet_infer_front512"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/results/dense_net/slurm_logs/densenet-infer-front512-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/results/dense_net/slurm_logs/densenet-infer-front512-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/dense_net/inference"
DATA_ROOT="${DATA_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/front_512_data}"
RESULTS_ROOT="${RESULTS_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/results/dense_net}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TEST_CSV="${TEST_CSV:-${DATA_ROOT}/manifests_preprocessed/test_manifest_preprocessed.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${RESULTS_ROOT}/inference/submission.csv}"
MODEL_NAME="${MODEL_NAME:-}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/resnick/groups/CS156b/from_central/2026/haa/asqin/.config/matplotlib}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: Set CHECKPOINT_PATH to a trained DenseNet checkpoint."
  exit 1
fi

mkdir -p "${RESULTS_ROOT}/inference" "${RESULTS_ROOT}/slurm_logs"
mkdir -p "${MPLCONFIGDIR}"
export MPLCONFIGDIR

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

CMD=(
  "${PYTHON_BIN[@]}" densenet_inference.py
  --checkpoint "${CHECKPOINT_PATH}"
  --csv "${TEST_CSV}"
  --output "${OUTPUT_PATH}"
  --batch_size 12
  --num_workers 16
)

if [[ -n "${MODEL_NAME}" ]]; then
  CMD+=(--model_name "${MODEL_NAME}")
fi

if [[ -n "${IMAGE_SIZE}" ]]; then
  CMD+=(--image_size "${IMAGE_SIZE}")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
