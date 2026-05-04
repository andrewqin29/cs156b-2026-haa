#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00

#SBATCH -J "densenet169_full_scaled_mse"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/asqin/dense_net_results/slurm_logs/densenet-train-full512-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/asqin/dense_net_results/slurm_logs/densenet-train-full512-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/dense_net/train"
DATA_ROOT="${DATA_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/full_512_nans_-999_JPG}"
RESULTS_ROOT="${RESULTS_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/asqin/dense_net_results}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/manifests_preprocessed/train_manifest_preprocessed.csv}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/manifests_preprocessed/val_manifest_preprocessed.csv}"
MODEL_NAME="${MODEL_NAME:-densenet169}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
EPOCHS="${EPOCHS:-15}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LR="${LR:-1e-4}"
HEAD_LR_MULTIPLIER="${HEAD_LR_MULTIPLIER:-10.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DROPOUT="${DROPOUT:-0.3}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-densenet169_full512_scaled_mse_mask0_baseline}"
TORCH_HOME_DIR="${TORCH_HOME_DIR:-/resnick/groups/CS156b/from_central/2026/haa/asqin/.torch_test_cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/resnick/groups/CS156b/from_central/2026/haa/asqin/.config/matplotlib}"

mkdir -p "${RESULTS_ROOT}/runs" "${RESULTS_ROOT}/slurm_logs" "${MPLCONFIGDIR}"
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
  "${PYTHON_BIN[@]}" train_densenet.py
  --train_csv "${TRAIN_CSV}"
  --val_csv "${VAL_CSV}"
  --runs_base "${RESULTS_ROOT}/runs"
  --run_name "${RUN_NAME}"
  --model_name "${MODEL_NAME}"
  --image_size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --head_lr_multiplier "${HEAD_LR_MULTIPLIER}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_workers "${NUM_WORKERS}"
  --dropout "${DROPOUT}"
  --patience "${PATIENCE}"
  --seed "${SEED}"
  --torch_home "${TORCH_HOME_DIR}"
)

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
