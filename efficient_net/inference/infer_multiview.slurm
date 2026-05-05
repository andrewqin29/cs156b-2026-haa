#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

#SBATCH -J "efficientnet_multiview_infer"
#SBATCH --mail-user=askuamr@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_results/slurm_logs/efficientnet-multiview-infer-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_results/slurm_logs/efficientnet-multiview-infer-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/full_512_nans_-999_JPG}"
RESULTS_ROOT="${RESULTS_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_results}"
CKPT_DIR="${CKPT_DIR:-${RESULTS_ROOT}/multiview_runs/efficientnet_b0_full512_scaled_mse}"
INFER_DIR="${REPO_ROOT}/efficient_net/inference"
TEST_CSV="${TEST_CSV:-${DATA_ROOT}/manifests_preprocessed/test_manifest_preprocessed.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-${RESULTS_ROOT}/inference/efficientnet_b0_multiview_submission.csv}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${RESULTS_ROOT}/slurm_logs" "${RESULTS_ROOT}/inference"

cd "${INFER_DIR}"

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
  "${PYTHON_BIN[@]}" infer_multiview.py
  --frontal_checkpoint "${CKPT_DIR}/best_frontal.pt"
  --lateral_checkpoint "${CKPT_DIR}/best_lateral.pt"
  --csv "${TEST_CSV}"
  --output "${OUTPUT_PATH}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
)

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
