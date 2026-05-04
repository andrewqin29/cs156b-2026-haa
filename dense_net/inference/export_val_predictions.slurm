#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

#SBATCH -J "densenet_val_export"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/asqin/dense_net_results/slurm_logs/densenet-val-export-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/asqin/dense_net_results/slurm_logs/densenet-val-export-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/dense_net/inference"
DATA_ROOT="${DATA_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/full_512_nans_-999_JPG}"
RESULTS_ROOT="${RESULTS_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/asqin/dense_net_results}"
RUN_NAME="${RUN_NAME:-densenet169_full512_scaled_mse_mask0_baseline}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RESULTS_ROOT}/runs/${RUN_NAME}/best_model.pt}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/manifests_preprocessed/val_manifest_preprocessed.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${RESULTS_ROOT}/val_predictions/${RUN_NAME}_val_predictions.csv}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
MODEL_NAME="${MODEL_NAME:-}"
DROPOUT="${DROPOUT:-}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/resnick/groups/CS156b/from_central/2026/haa/asqin/.config/matplotlib}"

mkdir -p "${RESULTS_ROOT}/val_predictions" "${RESULTS_ROOT}/slurm_logs" "${MPLCONFIGDIR}"
export MPLCONFIGDIR

cd "${SCRIPT_DIR}"

CONDA_ENV_NAME="${CONDA_ENV:-cs156b}"
if [[ -n "${PYTHON_EXE:-}" && -x "${PYTHON_EXE}" ]]; then
  PYTHON_BIN=("${PYTHON_EXE}")
elif [[ -x "/resnick/groups/CS156b/from_central/2026/haa/asqin/miniconda3/envs/cs156b_ready/bin/python" ]]; then
  PYTHON_BIN=("/resnick/groups/CS156b/from_central/2026/haa/asqin/miniconda3/envs/cs156b_ready/bin/python")
elif [[ -x "${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python")
elif [[ -x "${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python")
else
  echo "ERROR: Set PYTHON_EXE to your env's python (with torch installed)."
  exit 1
fi

echo "Using Python: ${PYTHON_BIN[*]}"
"${PYTHON_BIN[@]}" -c "import sys; import torch; print('Python:', sys.version); print('torch:', torch.__version__)"

CMD=(
  "${PYTHON_BIN[@]}" export_val_predictions.py
  --checkpoint "${CHECKPOINT_PATH}"
  --csv "${VAL_CSV}"
  --output "${OUTPUT_PATH}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --run_name "${RUN_NAME}"
  --limit_rows "${LIMIT_ROWS}"
)

if [[ -n "${IMAGE_SIZE}" ]]; then
  CMD+=(--image_size "${IMAGE_SIZE}")
fi
if [[ -n "${MODEL_NAME}" ]]; then
  CMD+=(--model_name "${MODEL_NAME}")
fi
if [[ -n "${DROPOUT}" ]]; then
  CMD+=(--dropout "${DROPOUT}")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
