#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

#SBATCH --time=00:30:00

#SBATCH -J "resnet_inference"
#SBATCH --mail-user=hgaston@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"

mkdir -p logs

PY_SCRIPT="$SUBMIT_DIR/resnet_inference.py"
if [ ! -f "$PY_SCRIPT" ]; then
  echo "ERROR: resnet_inference.py not found at $PY_SCRIPT"
  echo "Tip: run sbatch from the directory containing resnet_inference.py (resnet/inference)."
  exit 1
fi

/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/cs156b/bin/python "$PY_SCRIPT" \
  --checkpoint /resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/resnet/checkpoints_resnet_alex_manifests/best_model.pt \
  --csv /resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed/test_manifest_preprocessed.csv \
  --output /resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/resnet/inference/submission.csv