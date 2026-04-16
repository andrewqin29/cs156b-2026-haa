#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --partition=gpu
# Give plenty of headroom.
#SBATCH --time=2-00:00:00

#SBATCH -J "densenet_train"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=densenet-train-%j.out
#SBATCH --error=densenet-train-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/dense_net/train"
DEFAULT_RUNS_ROOT="/resnick/groups/CS156b/from_central/2026/haa/dense_net_runs"
TRAIN_CSV="${TRAIN_CSV:-/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed/train_manifest_preprocessed.csv}"
VAL_CSV="${VAL_CSV:-/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed/val_manifest_preprocessed.csv}"
TORCH_HOME_DIR="${TORCH_HOME_DIR:-/resnick/groups/CS156b/from_central/2026/haa/asqin/.torch_test_cache}"
MODEL_NAME="${MODEL_NAME:-densenet121}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
EPOCHS="${EPOCHS:-15}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
HEAD_LR_MULTIPLIER="${HEAD_LR_MULTIPLIER:-10.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DROPOUT="${DROPOUT:-0.3}"
PATIENCE="${PATIENCE:-3}"
SEED="${SEED:-42}"

if [[ -z "${RUN_NAME:-}" ]]; then
  RUN_NAME="${MODEL_NAME}_img${IMAGE_SIZE}_lr${LR}_w${WARMUP_EPOCHS}_p${PATIENCE}_bs${BATCH_SIZE}"
fi

OUT_ROOT="${OUT_ROOT:-${DEFAULT_RUNS_ROOT}/${RUN_NAME}}"

mkdir -p "${REPO_ROOT}/dense_net/logs"
mkdir -p "${OUT_ROOT}"
mkdir -p "${TORCH_HOME_DIR}"

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

echo "Run name: ${RUN_NAME}"
echo "Output dir: ${OUT_ROOT}"
echo "Model: ${MODEL_NAME}"
echo "Image size: ${IMAGE_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Warmup epochs: ${WARMUP_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR: ${LR}"
echo "Patience: ${PATIENCE}"

"${PYTHON_BIN[@]}" train_densenet.py \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --output_dir "${OUT_ROOT}" \
  --torch_home "${TORCH_HOME_DIR}" \
  --model_name "${MODEL_NAME}" \
  --image_size "${IMAGE_SIZE}" \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --head_lr_multiplier "${HEAD_LR_MULTIPLIER}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --dropout "${DROPOUT}" \
  --patience "${PATIENCE}" \
  --seed "${SEED}"
