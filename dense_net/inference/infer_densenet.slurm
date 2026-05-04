#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

#SBATCH -J "densenet169_infer_full"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/asqin/dense_net_results/slurm_logs/densenet-infer-full512-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/asqin/dense_net_results/slurm_logs/densenet-infer-full512-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/dense_net/inference"
DATA_ROOT="${DATA_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/full_512_nans_-999_JPG}"
RESULTS_ROOT="${RESULTS_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/asqin/dense_net_results}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RESULTS_ROOT}/runs/densenet169_full512_scaled_mse_mask0_baseline/best_model.pt}"
TEST_CSV="${TEST_CSV:-${DATA_ROOT}/manifests_preprocessed/test_manifest_preprocessed.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${RESULTS_ROOT}/inference/densenet169_full512_scaled_mse_submission.csv}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/resnick/groups/CS156b/from_central/2026/haa/asqin/.config/matplotlib}"

mkdir -p "${RESULTS_ROOT}/inference" "${RESULTS_ROOT}/slurm_logs" "${MPLCONFIGDIR}"
export MPLCONFIGDIR

cd "${SCRIPT_DIR}"

if [[ -n "${PYTHON_EXE:-}" && -x "${PYTHON_EXE}" ]]; then
  PYTHON_BIN=("${PYTHON_EXE}")
elif [[ -x "/resnick/groups/CS156b/from_central/2026/haa/asqin/miniconda3/envs/cs156b_ready/bin/python" ]]; then
  PYTHON_BIN=("/resnick/groups/CS156b/from_central/2026/haa/asqin/miniconda3/envs/cs156b_ready/bin/python")
else
  echo "ERROR: Set PYTHON_EXE to your env's python (with torch installed)."
  exit 1
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

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
