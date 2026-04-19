#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=gpu

#SBATCH --time=12:00:00

#SBATCH -J "resnet_multiview_mse"
#SBATCH --mail-user=hgaston@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

LOG_DIR=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs/slurm-multiview-%j.out
#SBATCH --error=/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/logs/slurm-multiview-%j.err

mkdir -p $LOG_DIR

cd /resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/train

PYTHON_BIN=/resnick/groups/CS156b/from_central/2026/haa/hgaston/miniconda3/envs/cs156b/bin/python

$PYTHON_BIN train_multiview.py \
    --epochs 40 \
    --batch_size 256 \
    --lr 3e-4 \
    --dropout 0.4 \
    --img_size 512 \
    --unfreeze_epoch 5 \
    --frontal_weight 0.75 \
    --patience 7 \
    --num_workers 16 \
    --output_dir /resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/checkpoints_mse_tanh_512 \
    --train_csv /resnick/groups/CS156b/from_central/2026/haa/full_512_nans_0_FIXED/manifests_preprocessed/train_manifest_preprocessed.csv \
    --val_csv /resnick/groups/CS156b/from_central/2026/haa/full_512_nans_0_FIXED/manifests_preprocessed/val_manifest_preprocessed.csv