#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

#SBATCH -J "densenet_prep"
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=densenet-prep-%j.out
#SBATCH --error=densenet-prep-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/dense_net/data_preprocessing"
MANIFEST_ROOT="${MANIFEST_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/resnick/groups/CS156b/from_central/2026/haa/dense_net_data}"
IMAGE_SIZE="${IMAGE_SIZE:-320}"
IMAGE_FORMAT="${IMAGE_FORMAT:-jpg}"
JPG_QUALITY="${JPG_QUALITY:-95}"

mkdir -p "${OUTPUT_ROOT}"

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
  echo "ERROR: Set PYTHON_EXE to your env's python (with pandas and pillow installed)."
  exit 1
fi

echo "Using Python: ${PYTHON_BIN[*]}"
echo "Manifest root: ${MANIFEST_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Image size: ${IMAGE_SIZE}"
echo "Image format: ${IMAGE_FORMAT}"

"${PYTHON_BIN[@]}" prepare_densenet_images.py \
  --manifest_root "${MANIFEST_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --image_size "${IMAGE_SIZE}" \
  --image_format "${IMAGE_FORMAT}" \
  --jpg_quality "${JPG_QUALITY}"
