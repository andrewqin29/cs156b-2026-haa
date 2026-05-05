#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH -J "efficientnet_multiview_mse"
#SBATCH --mail-user=askuamr@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_results/slurm_logs/efficientnet-multiview-train-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_results/slurm_logs/efficientnet-multiview-train-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/efficient_net/train"
DATA_ROOT="${DATA_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/full_512_nans_-999_JPG}"
RESULTS_ROOT="${RESULTS_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_results}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/manifests_preprocessed/train_manifest_preprocessed.csv}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/manifests_preprocessed/val_manifest_preprocessed.csv}"
MODEL_NAME="${MODEL_NAME:-efficientnet_b0}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-5e-5}"
DROPOUT="${DROPOUT:-0.4}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-6}"
PATIENCE="${PATIENCE:-7}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CONSISTENCY_LAMBDA="${CONSISTENCY_LAMBDA:-0.0}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/multiview_runs/${MODEL_NAME}_full512_scaled_mse}"

mkdir -p "${RESULTS_ROOT}/slurm_logs" "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

CONDA_ENV_NAME="${CONDA_ENV:-cs156b}"
if [[ -n "${PYTHON_EXE:-}" && -x "${PYTHON_EXE}" ]]; then
  PYTHON_BIN=("${PYTHON_EXE}")
elif [[ -x "/resnick/groups/CS156b/from_central/2026/haa/askumar/miniconda3/envs/cs156b/bin/python" ]]; then
  PYTHON_BIN=("/resnick/groups/CS156b/from_central/2026/haa/askumar/miniconda3/envs/cs156b/bin/python")
elif [[ -x "${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python")
elif [[ -x "${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
  PYTHON_BIN=("${HOME}/.conda/envs/${CONDA_ENV_NAME}/bin/python")
else
  echo "ERROR: Set PYTHON_EXE to your env's python (with torch installed)."
  exit 1
fi

CMD=(
  "${PYTHON_BIN[@]}" train_multiview.py
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --dropout "${DROPOUT}"
  --model_name "${MODEL_NAME}"
  --img_size "${IMAGE_SIZE}"
  --unfreeze_epoch "${UNFREEZE_EPOCH}"
  --patience "${PATIENCE}"
  --num_workers "${NUM_WORKERS}"
  --consistency_lambda "${CONSISTENCY_LAMBDA}"
  --output_dir "${OUTPUT_DIR}"
  --train_csv "${TRAIN_CSV}"
  --val_csv "${VAL_CSV}"
)

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
