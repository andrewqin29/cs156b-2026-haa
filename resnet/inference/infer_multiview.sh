#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=00:30:00

#SBATCH -J "resnet_inference"
#SBATCH --mail-user=hgaston@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs/slurm-infer-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs/slurm-infer-%j.err

LOG_DIR=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs

mkdir -p $LOG_DIR

PYTHON_BIN=/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/cs156b/bin/python
CKPT_DIR=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/checkpoints_v3
INFER_DIR=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/inference

cd $INFER_DIR

$PYTHON_BIN infer_multiview.py \
    --frontal_checkpoint $CKPT_DIR/best_frontal.pt \
    --csv /resnick/groups/CS156b/from_central/2026/haa/full_512_nans_-999_JPG/manifests_preprocessed/test_manifest_preprocessed.csv \
    --output $INFER_DIR/submission_v3.csv \
    --batch_size 256 \
    --num_workers 8